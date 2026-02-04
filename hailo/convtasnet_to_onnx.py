import torch
import torch.nn as nn
from typing import List
import os
from asteroid.models import ConvTasNet
from asteroid.masknn import norms, TDConvNet
from asteroid.masknn.convolutional import Conv1DBlock
import asteroid.masknn.activations as activations

# --- MONKEY PATCHES ---

# 1. Norms: Handle 4D input [B, C, 1, L] (Horizontal)
def _glob_norm_4d(x, eps: float = 1e-8):
    # Input x: [N, C, 1, L]
    # Sequential reduction
    # Mean over L (dim 3)
    m1 = x.mean(dim=3, keepdim=True) # [N, C, 1, 1]
    # Mean over C (dim 1)
    mean = m1.mean(dim=1, keepdim=True) # [N, 1, 1, 1]
    
    # Var
    d = x - mean
    d2 = d.pow(2)
    v1 = d2.mean(dim=3, keepdim=True)
    var = v1.mean(dim=1, keepdim=True)
    
    return d / torch.sqrt(var + eps)

norms._glob_norm = _glob_norm_4d

def globln_forward_4d(self, x, EPS: float = 1e-8):
    # x is [B, C, 1, L]
    value = norms._glob_norm(x, eps=EPS)
    
    if hasattr(self, 'gamma') and hasattr(self, 'beta'):
        C = self.gamma.shape[0]
        # Broadcast gamma/beta to [1, C, 1, 1]
        weight = self.gamma.view(1, C, 1, 1)
        bias = self.beta.view(1, C, 1, 1)
        return value * weight + bias
    return value

norms.GlobLN.forward = globln_forward_4d

# 2. TDConvNet Forward: Handle 4D input [B, C, 1, L]
def tdconvnet_forward_4d(self, mixture_w):
    # mixture_w is [Batch, C, 1, L]
    s = mixture_w.size()
    print(f"TDConvNet input size tuple: {s}")
    
    if len(s) == 4:
        batch, _, _, n_frames = s
    else:
        # Fallback if 3D
        batch, _, n_frames = s
        mixture_w = mixture_w.unsqueeze(2) # Insert H=1 at dim 2

    output = self.bottleneck(mixture_w)
    skip_connection = torch.tensor([0.0], device=output.device)
    for layer in self.TCN:
        tcn_out = layer(output)
        if self.skip_chan:
            residual, skip = tcn_out
            skip_connection = skip_connection + skip
        else:
            residual = tcn_out
        output = output + residual
        
    mask_inp = skip_connection if self.skip_chan else output
    score = self.mask_net(mask_inp)
    
    # Score is [B, n_src*out_chan, 1, L]
    # No view, keep concatenated
    est_mask = self.output_act(score)
    return est_mask

TDConvNet.forward = tdconvnet_forward_4d

# --------------------

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
    def __init__(self, original_model):
        super().__init__()
        self.encoder = original_model.encoder
        self.masker = original_model.masker
        self.enc_activation = original_model.enc_activation

        # Handle Decoder with groups for n_src
        old_decoder = original_model.decoder
        n_src = self.masker.n_src
        self.n_src = n_src

        # Create grouped decoder
        # old_decoder is ConvTranspose2d [In, Out, 1, K]
        new_decoder = nn.ConvTranspose2d(
            old_decoder.in_channels * n_src,
            old_decoder.out_channels * n_src, # 1 * n_src
            old_decoder.kernel_size,
            stride=old_decoder.stride,
            padding=old_decoder.padding,
            groups=n_src,
            bias=(old_decoder.bias is not None)
        )
        
        # Copy weights: [In, Out, 1, K]. In=512. Out=1.
        # Target: [1024, 1, 1, K].
        new_decoder.weight.data = torch.cat([old_decoder.weight.data] * n_src, dim=0)
        if old_decoder.bias is not None:
             new_decoder.bias.data = torch.cat([old_decoder.bias.data] * n_src, dim=0)
        
        self.decoder = new_decoder

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
        
        # Application
        # tf_rep is [B, out_chan, 1, L]
        # Repeat on channel dim to match est_masks
        tf_rep_repeated = torch.cat([tf_rep] * self.n_src, dim=1)
        
        masked_tf_rep = est_masks * tf_rep_repeated
        
        # Decoder (Grouped)
        # masked_tf_rep is [B, n_src*out_chan, 1, L]
        decoded = self.decoder(masked_tf_rep)
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
        return torch.max(torch.zeros_like(x), x) + w * torch.min(torch.zeros_like(x), x)

def replace_prelu_with_custom(model):
    for name, module in model.named_children():
        if isinstance(module, nn.PReLU):
            print(f"Replacing PReLU {name} with PReLU4D")
            setattr(model, name, PReLU4D(module))
        else:
            replace_prelu_with_custom(module)

def main():
    # 1. Load trained model
    pretrained_model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM!_sepclean")
    print(f"Model Config: {pretrained_model.masker.get_config()}")
    
    # 2. Convert to 4D (Horizontal)
    print("Converting model to 4D (Conv2d Horizontal)...")
    convert_model_to_4d(pretrained_model)
    
    # Replace PReLU with Custom PReLU (Max/Min/Mul/Add)
    replace_prelu_with_custom(pretrained_model)
    
    # 3. Wrap
    export_model = HailoExportWrapper(pretrained_model)
    export_model.eval()
    
    # 4. Dummy Input [1, 1, 1, 16000]
    dummy_input = torch.randn(1, 1, 1, 16000) 
    
    output_path = "hailo/convtas_hailo_ready.onnx"
    
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
