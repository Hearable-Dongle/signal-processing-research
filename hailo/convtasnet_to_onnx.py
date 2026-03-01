import torch
import torch.nn as nn
from typing import List, Tuple
import argparse
from asteroid.models import ConvTasNet
from asteroid.masknn import norms, TDConvNet

# --- MONKEY PATCHES ---

NORM_MODE = "channel"
NORM_VAR_FLOOR = 1e-6
NORM_OUT_CLAMP = 12.0
PRELU_OUT_CLAMP = 12.0


def _str_to_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


class ChannelLNExport(nn.Module):
    """Export-safe channel-wise LN for both 3D (N,C,T) and 4D (N,C,1,L)."""

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, eps: float):
        super().__init__()
        self.gamma = nn.Parameter(gamma.clone().detach(), requires_grad=True)
        self.beta = nn.Parameter(beta.clone().detach(), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        if x.dim() == 3:
            mean = x.mean(dim=1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=1, keepdim=True)
            normed = (x - mean) / torch.sqrt(torch.clamp(var + self.eps, min=NORM_VAR_FLOOR))
            weight = self.gamma.view(1, -1, 1)
            bias = self.beta.view(1, -1, 1)
            return torch.clamp(normed * weight + bias, min=-NORM_OUT_CLAMP, max=NORM_OUT_CLAMP)
        if x.dim() == 4:
            mean = x.mean(dim=1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=1, keepdim=True)
            normed = (x - mean) / torch.sqrt(torch.clamp(var + self.eps, min=NORM_VAR_FLOOR))
            weight = self.gamma.view(1, -1, 1, 1)
            bias = self.beta.view(1, -1, 1, 1)
            return torch.clamp(normed * weight + bias, min=-NORM_OUT_CLAMP, max=NORM_OUT_CLAMP)
        raise ValueError(f"ChannelLNExport expects 3D or 4D input, got shape={tuple(x.shape)}")


class AffineNormExport(nn.Module):
    """Export-safe affine-only norm replacement: y = gamma * x + beta."""

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor):
        super().__init__()
        self.gamma = nn.Parameter(gamma.clone().detach(), requires_grad=True)
        self.beta = nn.Parameter(beta.clone().detach(), requires_grad=True)

    def forward(self, x):
        if x.dim() == 3:
            weight = self.gamma.view(1, -1, 1)
            bias = self.beta.view(1, -1, 1)
            return torch.clamp(x * weight + bias, min=-NORM_OUT_CLAMP, max=NORM_OUT_CLAMP)
        if x.dim() == 4:
            weight = self.gamma.view(1, -1, 1, 1)
            bias = self.beta.view(1, -1, 1, 1)
            return torch.clamp(x * weight + bias, min=-NORM_OUT_CLAMP, max=NORM_OUT_CLAMP)
        raise ValueError(f"AffineNormExport expects 3D or 4D input, got shape={tuple(x.shape)}")


def _glob_norm_4d(x, eps: float = 1e-8):
    # Input x: [N, C, 1, L]
    if NORM_MODE == "global":
        # Original GlobLN behavior: normalize over channels and time.
        mean = x.mean(dim=(1, 3), keepdim=True)  # [N, 1, 1, 1]
        var = (x - mean).pow(2).mean(dim=(1, 3), keepdim=True)
    else:
        # Channel-wise fallback for Hailo export stability:
        # keep channel axis in stats to reduce problematic feature broadcasts.
        mean = x.mean(dim=3, keepdim=True)  # [N, C, 1, 1]
        var = (x - mean).pow(2).mean(dim=3, keepdim=True)
    # Keep denominator bounded away from zero to avoid unstable large slopes
    # during quantization.
    denom = torch.sqrt(torch.clamp(var + eps, min=NORM_VAR_FLOOR))
    normalized = (x - mean) / denom
    return torch.clamp(normalized, min=-NORM_OUT_CLAMP, max=NORM_OUT_CLAMP)


norms._glob_norm = _glob_norm_4d

def globln_forward_4d(self, x, EPS: float = 1e-8):
    # x is [B, C, 1, L]
    value = norms._glob_norm(x, eps=EPS)
    
    if hasattr(self, 'gamma') and hasattr(self, 'beta'):
        C = self.gamma.shape[0]
        # Broadcast gamma/beta to [1, C, 1, 1]
        weight = self.gamma.view(1, C, 1, 1)
        bias = self.beta.view(1, C, 1, 1)
        scaled = value * weight + bias
        return torch.clamp(scaled, min=-NORM_OUT_CLAMP, max=NORM_OUT_CLAMP)
    return value

norms.GlobLN.forward = globln_forward_4d

# 2. TDConvNet Forward: Handle 4D input [B, C, 1, L]
def tdconvnet_forward_4d(self, mixture_w):
    # mixture_w is [Batch, C, 1, L]
    s = mixture_w.size()
    
    if len(s) == 4:
        batch, _, _, n_frames = s
    else:
        # Fallback if 3D
        batch, _, n_frames = s
        mixture_w = mixture_w.unsqueeze(2) # Insert H=1 at dim 2

    output = self.bottleneck(mixture_w)
    disable_skip = bool(getattr(self, "_hailo_disable_skip", False))
    truncate_k_blocks = int(getattr(self, "_hailo_truncate_k_blocks", 0))
    skip_connection = None

    for idx, layer in enumerate(self.TCN):
        if truncate_k_blocks > 0 and idx >= truncate_k_blocks:
            break
        tcn_out = layer(output)
        if self.skip_chan and not disable_skip:
            residual, skip = tcn_out
            skip_connection = skip if skip_connection is None else (skip_connection + skip)
        else:
            residual = tcn_out[0] if isinstance(tcn_out, tuple) else tcn_out
        output = output + residual

    if self.skip_chan and not disable_skip and skip_connection is not None:
        mask_inp = skip_connection
    else:
        mask_inp = output
    score = self.mask_net(mask_inp)
    
    # Score is [B, n_src*out_chan, 1, L]
    # No view, keep concatenated
    est_mask = self.output_act(score)
    return est_mask

TDConvNet.forward = tdconvnet_forward_4d


def convert_model_to_4d(model):
    """Recursively replace Conv1d/ConvTranspose1d with Conv2d/ConvTranspose2d (Horizontal)."""
    for name, module in model.named_children():
        print(f"Visiting: {name} ({type(module)})")
        
        if isinstance(module, nn.Conv1d):
            print(f"Converting Conv1d: {name}")
            # Create Conv2d (Horizontal: 1xK)
            new_layer = nn.Conv2d(
                module.in_channels,
                module.out_channels,
                (1, module.kernel_size[0]),
                stride=(1, module.stride[0]),
                padding=(0, module.padding[0]),
                dilation=(1, module.dilation[0]),
                groups=module.groups,
                bias=(module.bias is not None)
            )
            # Copy weights: [Out, In, K] -> [Out, In, 1, K]
            new_layer.weight.data = module.weight.data.unsqueeze(2)
            if module.bias is not None:
                new_layer.bias.data = module.bias.data
            setattr(model, name, new_layer)
            
        elif isinstance(module, nn.ConvTranspose1d):
            # Create ConvTranspose2d (Horizontal)
            new_layer = nn.ConvTranspose2d(
                module.in_channels,
                module.out_channels,
                (1, module.kernel_size[0]),
                stride=(1, module.stride[0]),
                padding=(0, module.padding[0]),
                dilation=(1, module.dilation[0]),
                groups=module.groups,
                bias=(module.bias is not None)
            )
            # Copy weights: [In, Out, K] -> [In, Out, 1, K]
            new_layer.weight.data = module.weight.data.unsqueeze(2)
            if module.bias is not None:
                new_layer.bias.data = module.bias.data
            setattr(model, name, new_layer)

        elif "Encoder" in str(type(module)) and hasattr(module, "filterbank"):
            print(f"Converting Asteroid Encoder: {name}")
            w = module.filterbank._filters # [Out, In, K]
            stride = module.stride
            padding = module.padding
            if isinstance(stride, int): stride = (1, stride)
            else: stride = (1, stride[0])
            
            if isinstance(padding, int): padding = (0, padding)
            else: padding = (0, padding[0])

            new_layer = nn.Conv2d(
                w.size(1), # In
                w.size(0), # Out
                (1, w.size(2)), # 1xK
                stride=stride,
                padding=padding,
                bias=False
            )
            new_layer.weight.data = w.unsqueeze(2) # [Out, In, 1, K]
            setattr(model, name, new_layer)

        elif "Decoder" in str(type(module)) and hasattr(module, "filterbank"):
            print(f"Converting Asteroid Decoder: {name}")
            w = module.filterbank._filters # [In, Out, K]
            stride = module.stride
            padding = module.padding
            if isinstance(stride, int): stride = (1, stride)
            else: stride = (1, stride[0])
            
            if isinstance(padding, int): padding = (0, padding)
            else: padding = (0, padding[0])
            
            new_layer = nn.ConvTranspose2d(
                w.size(0), # In
                w.size(1), # Out
                (1, w.size(2)), # 1xK
                stride=stride,
                padding=padding,
                bias=False
            )
            new_layer.weight.data = w.unsqueeze(2) # [In, Out, 1, K]
            setattr(model, name, new_layer)

        else:
            convert_model_to_4d(module)


class HailoExportWrapper(nn.Module):
    def __init__(
        self,
        original_model,
        mask_mul_mode: str = "normal",
        force_n_src_1: bool = False,
        bypass_concat: bool = False,
        skip_topology_mode: str = "concat",
        deconv_mode: str = "grouped",
    ):
        super().__init__()
        self.encoder = original_model.encoder
        self.masker = original_model.masker
        self.enc_activation = original_model.enc_activation
        self.mask_mul_mode = mask_mul_mode
        self.bypass_concat = bypass_concat
        self.skip_topology_mode = skip_topology_mode
        self.deconv_mode = deconv_mode
        self.decoder_pre = nn.Identity()

        # Handle Decoder with groups for n_src
        old_decoder = original_model.decoder
        self.model_n_src = self.masker.n_src
        self.export_n_src = 1 if force_n_src_1 else self.model_n_src
        self.source_projector = None

        if self.export_n_src > 1:
            if self.skip_topology_mode == "project":
                in_ch = old_decoder.in_channels
                out_ch = in_ch * self.export_n_src
                proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
                with torch.no_grad():
                    proj.weight.zero_()
                    for src_idx in range(self.export_n_src):
                        start = src_idx * in_ch
                        for ch in range(in_ch):
                            proj.weight[start + ch, ch, 0, 0] = 1.0
                self.source_projector = proj

            in_total = old_decoder.in_channels * self.export_n_src
            out_total = old_decoder.out_channels * self.export_n_src

            if self.deconv_mode == "ungrouped_blockdiag":
                # Build a groups=1 deconvolution with block-diagonal weights that
                # is mathematically equivalent to grouped deconvolution.
                new_decoder = nn.ConvTranspose2d(
                    in_total,
                    out_total,
                    old_decoder.kernel_size,
                    stride=old_decoder.stride,
                    padding=old_decoder.padding,
                    groups=1,
                    bias=(old_decoder.bias is not None),
                )
                with torch.no_grad():
                    new_decoder.weight.zero_()
                    in_ch = old_decoder.in_channels
                    out_per_src = old_decoder.out_channels
                    for src_idx in range(self.export_n_src):
                        in_start = src_idx * in_ch
                        out_start = src_idx * out_per_src
                        new_decoder.weight[
                            in_start : in_start + in_ch,
                            out_start : out_start + out_per_src,
                            :,
                            :,
                        ] = old_decoder.weight
                    if old_decoder.bias is not None:
                        new_decoder.bias.copy_(torch.cat([old_decoder.bias] * self.export_n_src, dim=0))
                self.decoder = new_decoder
            elif self.deconv_mode in {"reduced_deconv_128", "reduced_deconv_64"}:
                reduced_ch = 128 if self.deconv_mode.endswith("128") else 64
                self.decoder_pre = nn.Conv2d(in_total, reduced_ch, kernel_size=1, bias=False)
                self.decoder = nn.ConvTranspose2d(
                    reduced_ch,
                    out_total,
                    old_decoder.kernel_size,
                    stride=old_decoder.stride,
                    padding=old_decoder.padding,
                    groups=1,
                    bias=(old_decoder.bias is not None),
                )
            elif self.deconv_mode == "conv1x1_head":
                # Minimal debug head to reduce allocator pressure; no time upsampling.
                self.decoder = nn.Conv2d(in_total, out_total, kernel_size=1, bias=True)
            else:
                # Create grouped decoder
                # old_decoder is ConvTranspose2d [In, Out, 1, K]
                new_decoder = nn.ConvTranspose2d(
                    in_total,
                    out_total,  # 1 * n_src
                    old_decoder.kernel_size,
                    stride=old_decoder.stride,
                    padding=old_decoder.padding,
                    groups=self.export_n_src,
                    bias=(old_decoder.bias is not None),
                )

                # Copy weights: [In, Out, 1, K]. In=512. Out=1.
                # Target: [1024, 1, 1, K] for 2 sources.
                new_decoder.weight.data = torch.cat([old_decoder.weight.data] * self.export_n_src, dim=0)
                if old_decoder.bias is not None:
                    new_decoder.bias.data = torch.cat([old_decoder.bias.data] * self.export_n_src, dim=0)

                self.decoder = new_decoder
        else:
            # Single-source export path.
            if self.deconv_mode == "conv1x1_head":
                self.decoder = nn.Conv2d(old_decoder.in_channels, old_decoder.out_channels, kernel_size=1, bias=True)
            elif self.deconv_mode in {"reduced_deconv_128", "reduced_deconv_64"}:
                reduced_ch = 128 if self.deconv_mode.endswith("128") else 64
                self.decoder_pre = nn.Conv2d(old_decoder.in_channels, reduced_ch, kernel_size=1, bias=False)
                self.decoder = nn.ConvTranspose2d(
                    reduced_ch,
                    old_decoder.out_channels,
                    old_decoder.kernel_size,
                    stride=old_decoder.stride,
                    padding=old_decoder.padding,
                    groups=1,
                    bias=(old_decoder.bias is not None),
                )
            else:
                self.decoder = old_decoder

    def forward(self, wav_4d):
        # wav_4d is [1, 1, 1, 16000]
        print(f"wav_4d shape: {wav_4d.shape}")
        
        # Encoder (Conv2d Horizontal)
        tf_rep = self.encoder(wav_4d)
        print(f"tf_rep after encoder: {tf_rep.shape}")
        
        if self.enc_activation is not None:
            tf_rep = self.enc_activation(tf_rep)
        
        # Masker (TDConvNet 4D Horizontal)
        est_masks = self.masker(tf_rep)
        # est_masks is [B, n_src*out_chan, 1, L]

        if self.export_n_src == 1:
            out_chan = tf_rep.shape[1]
            est_masks = est_masks[:, :out_chan, :, :]
            tf_rep_repeated = tf_rep
        else:
            # tf_rep is [B, out_chan, 1, L]
            # Repeat on channel dim to match est_masks
            if self.skip_topology_mode == "project":
                tf_rep_repeated = self.source_projector(tf_rep)
            elif self.bypass_concat:
                # Use repeat() path to avoid explicit concat op in ONNX for topology isolation.
                tf_rep_repeated = tf_rep.repeat(1, self.export_n_src, 1, 1)
            else:
                tf_rep_repeated = torch.cat([tf_rep] * self.export_n_src, dim=1)

        if self.mask_mul_mode == "bypass":
            masked_tf_rep = tf_rep_repeated
        else:
            masked_tf_rep = est_masks * tf_rep_repeated
        
        # Decoder (Grouped)
        # masked_tf_rep is [B, n_src*out_chan, 1, L]
        decoded = self.decoder(self.decoder_pre(masked_tf_rep))
        # decoded is [B, n_src, 1, T_out]
        
        # Reshape back to [B, n_src, T_out]
        decoded = decoded.squeeze(2)
        
        return decoded

class PReLU4D(nn.Module):
    def __init__(self, original_prelu):
        super().__init__()
        self.weight = original_prelu.weight

    def forward(self, x):
        # x: [B, C, 1, L]
        # weight: [C] -> [1, C, 1, 1]
        w = self.weight.view(1, -1, 1, 1)
        # PReLU(x) = max(0, x) + w * min(0, x)
        out = torch.max(torch.zeros_like(x), x) + w * torch.min(torch.zeros_like(x), x)
        return torch.clamp(out, min=-PRELU_OUT_CLAMP, max=PRELU_OUT_CLAMP)


def replace_prelu_with_custom(model):
    for name, module in model.named_children():
        if isinstance(module, nn.PReLU):
            print(f"Replacing PReLU {name} with PReLU4D")
            setattr(model, name, PReLU4D(module))
        else:
            replace_prelu_with_custom(module)


def replace_prelu_with_relu(model) -> int:
    count = 0
    for name, module in model.named_children():
        if isinstance(module, nn.PReLU):
            setattr(model, name, nn.ReLU(inplace=False))
            count += 1
        else:
            count += replace_prelu_with_relu(module)
    return count


def replace_globln_with_cln(model, eps: float) -> int:
    count = 0
    for name, module in model.named_children():
        if isinstance(module, norms.GlobLN):
            repl = ChannelLNExport(module.gamma, module.beta, eps=eps)
            setattr(model, name, repl)
            count += 1
        else:
            count += replace_globln_with_cln(module, eps=eps)
    return count


def replace_globln_with_affine(model) -> int:
    count = 0
    for name, module in model.named_children():
        if isinstance(module, norms.GlobLN):
            repl = AffineNormExport(module.gamma, module.beta)
            setattr(model, name, repl)
            count += 1
        else:
            count += replace_globln_with_affine(module)
    return count


def replace_globln_with_identity(model) -> int:
    count = 0
    for name, module in model.named_children():
        if isinstance(module, norms.GlobLN):
            setattr(model, name, nn.Identity())
            count += 1
        else:
            count += replace_globln_with_identity(module)
    return count


def apply_export_profile(args: argparse.Namespace) -> Tuple[str, str]:
    act_replace = args.act_replace
    norm_replace = args.norm_replace
    if args.export_profile == "hailo_safe":
        if act_replace == "none":
            act_replace = "relu"
        if norm_replace == "none":
            norm_replace = "identity"
    return act_replace, norm_replace

def main():
    parser = argparse.ArgumentParser(description="Export ConvTasNet to ONNX")
    parser.add_argument("output", nargs="?", default="hailo/convtas_hailo_ready.onnx", help="Output ONNX file path")
    parser.add_argument(
        "--norm_mode",
        choices=["channel", "global"],
        default="channel",
        help="Normalization mode for export. 'channel' is preferred for Hailo compile stability.",
    )
    parser.add_argument(
        "--export_profile",
        choices=["hailo_safe", "baseline"],
        default="hailo_safe",
        help="Export compatibility profile. 'hailo_safe' enables conservative activation/norm substitutions.",
    )
    parser.add_argument(
        "--act_replace",
        choices=["none", "relu"],
        default="relu",
        help="Replace PReLU modules during export.",
    )
    parser.add_argument(
        "--norm_replace",
        choices=["none", "cln", "affine", "identity"],
        default="identity",
        help="Replace GlobLN modules during export. 'identity' is strongest compile-focused fallback.",
    )
    parser.add_argument(
        "--norm_eps",
        type=float,
        default=1e-8,
        help="Epsilon for export-time normalization replacements.",
    )
    parser.add_argument(
        "--disable_skip",
        choices=["true", "false"],
        default="false",
        help="Disable skip-connection accumulation in TDConvNet forward for export debugging.",
    )
    parser.add_argument(
        "--mask_mul_mode",
        choices=["normal", "bypass"],
        default="normal",
        help="Control masker application in wrapper. 'bypass' skips est_masks * tf_rep multiplication.",
    )
    parser.add_argument(
        "--force_n_src_1",
        choices=["true", "false"],
        default="false",
        help="Export a single-source decode path for topology isolation.",
    )
    parser.add_argument(
        "--bypass_concat",
        choices=["true", "false"],
        default="false",
        help="Use tensor repeat instead of explicit cat when expanding encoder features across sources.",
    )
    parser.add_argument(
        "--skip_topology_mode",
        choices=["concat", "project"],
        default="concat",
        help="How to build source expansion topology. 'project' replaces skip-concat fan-in with 1x1 projection.",
    )
    parser.add_argument(
        "--deconv_mode",
        choices=["grouped", "ungrouped_blockdiag", "reduced_deconv_128", "reduced_deconv_64", "conv1x1_head"],
        default="grouped",
        help="Decoder mode fallback selector for compile debugging and allocator pressure reduction.",
    )
    parser.add_argument(
        "--truncate_k_blocks",
        type=int,
        default=0,
        help="If > 0, run only first K Conv1DBlocks in TDConvNet during export tracing.",
    )
    args = parser.parse_args()
    global NORM_MODE, NORM_VAR_FLOOR
    NORM_MODE = args.norm_mode
    NORM_VAR_FLOOR = max(args.norm_eps, 1e-12)
    disable_skip = _str_to_bool(args.disable_skip)
    force_n_src_1 = _str_to_bool(args.force_n_src_1)
    bypass_concat = _str_to_bool(args.bypass_concat)
    truncate_k_blocks = max(args.truncate_k_blocks, 0)
    act_replace, norm_replace = apply_export_profile(args)

    # 1. Load trained model
    pretrained_model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM!_sepclean")
    print(f"Model Config: {pretrained_model.masker.get_config()}")

    prelu_replaced = 0
    globln_replaced = 0
    if act_replace == "relu":
        prelu_replaced = replace_prelu_with_relu(pretrained_model)
    if norm_replace == "cln":
        globln_replaced = replace_globln_with_cln(pretrained_model, eps=args.norm_eps)
    elif norm_replace == "affine":
        globln_replaced = replace_globln_with_affine(pretrained_model)
    elif norm_replace == "identity":
        globln_replaced = replace_globln_with_identity(pretrained_model)
    print(
        "Export substitutions: "
        f"profile={args.export_profile}, act_replace={act_replace}, norm_replace={norm_replace}, "
        f"prelu_replaced={prelu_replaced}, globln_replaced={globln_replaced}"
    )

    # 2. Convert to 4D (Horizontal)
    print("Converting model to 4D (Conv2d Horizontal)...")
    convert_model_to_4d(pretrained_model)

    pretrained_model.masker._hailo_disable_skip = disable_skip
    pretrained_model.masker._hailo_truncate_k_blocks = truncate_k_blocks
    print(
        "Topology debug options: "
        f"disable_skip={disable_skip}, mask_mul_mode={args.mask_mul_mode}, "
        f"force_n_src_1={force_n_src_1}, bypass_concat={bypass_concat}, "
        f"skip_topology_mode={args.skip_topology_mode}, deconv_mode={args.deconv_mode}, "
        f"truncate_k_blocks={truncate_k_blocks}"
    )

    # 3. Wrap
    export_model = HailoExportWrapper(
        pretrained_model,
        mask_mul_mode=args.mask_mul_mode,
        force_n_src_1=force_n_src_1,
        bypass_concat=bypass_concat,
        skip_topology_mode=args.skip_topology_mode,
        deconv_mode=args.deconv_mode,
    )
    export_model.eval()
    
    # 4. Dummy Input [1, 1, 1, 16000]
    dummy_input = torch.randn(1, 1, 1, 16000) 
    
    output_path = args.output
    
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"Export done to {output_path}")

if __name__ == "__main__":
    main()
