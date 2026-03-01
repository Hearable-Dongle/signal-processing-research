import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
HAILO_ASTEROID_ROOT = REPO_ROOT / "asteroid"
if str(HAILO_ASTEROID_ROOT) not in sys.path:
    sys.path.insert(0, str(HAILO_ASTEROID_ROOT))

from asteroid.masknn.hailo_activations import get_hailo_activation
from asteroid.masknn.hailo_convolutional import HailoConv1DBlock2D, HailoConv1DBlockAsTensor, HailoTDConvNet2D
from asteroid.masknn.hailo_norms import HailoChannelAffineNorm2D, HailoIdentityNorm2D
from asteroid.models.hailo_conv_tasnet import HailoConvTasNet, HailoDecoderConv1x1Head, HailoEncoder2D
from asteroid.models.hailo_conv_tasnet_submodules import (
    HailoConv1x1PartialBlock,
    HailoDecoderHeadSingleSrc,
    HailoDecoderPreConvOrIdentity1x1,
    HailoDecoderPreSlice,
    HailoMaskerBottleneckBlock,
    HailoMaskerBottleneckOnly,
    HailoMaskerFirstTCNBlockAsTensor,
    HailoMaskerHeadOnly,
    HailoMaskerHeadBlock,
    HailoMaskerTCN0DepthBlock,
    HailoMaskerTCN0InConvBlock,
    HailoMaskerTCN0ResBlock,
    HailoMaskerTCN0SkipBlock,
    HailoSourceProjectorSlice,
)


class ActivationWrapper(nn.Module):
    def __init__(self, act: nn.Module):
        super().__init__()
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


def _maybe_extract_state_dict(blob):
    if isinstance(blob, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in blob and isinstance(blob[key], dict):
                return blob[key]
    return blob


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        normalized[nk] = v
    return normalized


def _load_hailo_convtas_weights(model: HailoConvTasNet, state_dict_path: str, strict: bool) -> None:
    ckpt = torch.load(state_dict_path, map_location="cpu")
    state = _maybe_extract_state_dict(ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported state dict payload in {state_dict_path}")
    state = _normalize_state_dict_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict:
        print(f"[WEIGHTS] loaded strictly from {state_dict_path}")
        return
    print(
        "[WEIGHTS] loaded non-strictly from "
        f"{state_dict_path} (missing={len(missing)}, unexpected={len(unexpected)})"
    )


def _build_hailo_convtas(args: argparse.Namespace) -> HailoConvTasNet:
    model = HailoConvTasNet(
        n_src=args.n_src,
        n_filters=args.n_filters,
        kernel_size=args.encdec_kernel_size,
        stride=args.encdec_stride,
        bn_chan=args.bn_chan,
        hid_chan=args.hid_chan,
        skip_chan=args.skip_chan,
        n_blocks=args.n_blocks,
        n_repeats=args.n_repeats,
        conv_kernel_size=args.kernel_size,
        mask_act=args.mask_act,
        norm_mode=args.norm_mode,
        mask_mul_mode=args.mask_mul_mode,
        force_n_src_1=args.force_n_src_1,
        skip_topology_mode=args.skip_topology_mode,
        decoder_mode=args.decoder_mode,
        truncate_k_blocks=args.truncate_k_blocks,
    )
    if args.state_dict_path:
        _load_hailo_convtas_weights(model, args.state_dict_path, strict=bool(args.state_dict_strict))
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dump_op_inventory(onnx_path: Path) -> None:
    model = onnx.load(str(onnx_path))
    counts = Counter(node.op_type for node in model.graph.node)
    print("[ONNX_OPS] " + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))


def build_module(args: argparse.Namespace):
    t = args.time_len
    if args.module == "norm":
        module = HailoChannelAffineNorm2D(args.in_chan) if args.norm_mode == "affine" else HailoIdentityNorm2D(args.in_chan)
        dummy = torch.randn(args.batch, args.in_chan, 1, t)
        return module, dummy

    if args.module == "activation":
        module = ActivationWrapper(get_hailo_activation(args.mask_act))
        dummy = torch.randn(args.batch, args.in_chan, 1, t)
        return module, dummy

    if args.module == "conv1d_block":
        padding = ((args.kernel_size - 1) * args.dilation) // 2
        block = HailoConv1DBlock2D(
            in_chan=args.in_chan,
            hid_chan=args.hid_chan,
            skip_out_chan=args.skip_chan,
            kernel_size=args.kernel_size,
            padding=padding,
            dilation=args.dilation,
            norm_mode=args.norm_mode,
        )
        module = HailoConv1DBlockAsTensor(block)
        dummy = torch.randn(args.batch, args.in_chan, 1, t)
        return module, dummy

    if args.module == "tdconvnet":
        module = HailoTDConvNet2D(
            in_chan=args.in_chan,
            n_src=args.n_src,
            out_chan=args.out_chan,
            n_blocks=args.n_blocks,
            n_repeats=args.n_repeats,
            bn_chan=args.bn_chan,
            hid_chan=args.hid_chan,
            skip_chan=args.skip_chan,
            conv_kernel_size=args.kernel_size,
            mask_act=args.mask_act,
            norm_mode=args.norm_mode,
            disable_skip=args.disable_skip,
            truncate_k_blocks=args.truncate_k_blocks,
        )
        dummy = torch.randn(args.batch, args.in_chan, 1, t)
        return module, dummy

    if args.module == "encoder":
        module = HailoEncoder2D(n_filters=args.n_filters, kernel_size=args.encdec_kernel_size, stride=args.encdec_stride)
        dummy = torch.randn(args.batch, 1, 1, t)
        return module, dummy

    if args.module == "encoder_conv_only":
        module = HailoEncoder2D(n_filters=args.n_filters, kernel_size=args.encdec_kernel_size, stride=args.encdec_stride).conv
        dummy = torch.randn(args.batch, 1, 1, t)
        return module, dummy

    if args.module == "decoder":
        module = HailoDecoderConv1x1Head(args.in_chan, args.n_src)
        dummy = torch.randn(args.batch, args.in_chan, 1, t)
        return module, dummy

    if args.module == "hailo_convtasnet":
        module = _build_hailo_convtas(args)
        dummy = torch.randn(args.batch, 1, 1, t)
        return module, dummy

    if args.module == "convtas_encoder_only":
        full = _build_hailo_convtas(args)
        module = full.encoder
        dummy = torch.randn(args.batch, 1, 1, t)
        return module, dummy

    if args.module == "convtas_masker_only":
        full = _build_hailo_convtas(args)
        module = full.masker
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.n_filters, 1, latent_t)
        return module, dummy

    if args.module == "convtas_masker_bottleneck_only":
        full = _build_hailo_convtas(args)
        module = HailoMaskerBottleneckOnly(full.masker)
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.n_filters, 1, latent_t)
        return module, dummy

    if args.module == "convtas_masker_tcn_block0_only":
        full = _build_hailo_convtas(args)
        module = HailoMaskerFirstTCNBlockAsTensor(full.masker)
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.bn_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_masker_mask_head_only":
        full = _build_hailo_convtas(args)
        module = HailoMaskerHeadOnly(full.masker)
        latent_t = max(1, t // max(1, args.encdec_stride))
        if args.disable_skip or args.skip_chan == 0:
            head_in_chan = args.bn_chan
        else:
            head_in_chan = args.skip_chan
        dummy = torch.randn(args.batch, head_in_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_masker_bottleneck_block":
        full = _build_hailo_convtas(args)
        module = HailoMaskerBottleneckBlock(
            full.masker,
            out_block_idx=args.out_block_idx,
            in_block_idx=args.in_block_idx,
            block_chan=args.block_chan,
        )
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.block_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_masker_tcn0_inconv_block":
        full = _build_hailo_convtas(args)
        module = HailoMaskerTCN0InConvBlock(
            full.masker,
            out_block_idx=args.out_block_idx,
            in_block_idx=args.in_block_idx,
            block_chan=args.block_chan,
        )
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.block_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_masker_tcn0_depth_block":
        full = _build_hailo_convtas(args)
        module = HailoMaskerTCN0DepthBlock(
            full.masker,
            depth_block_idx=args.depth_block_idx,
            block_chan=args.block_chan,
        )
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.block_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_masker_tcn0_res_block":
        full = _build_hailo_convtas(args)
        module = HailoMaskerTCN0ResBlock(
            full.masker,
            out_block_idx=args.out_block_idx,
            in_block_idx=args.in_block_idx,
            block_chan=args.block_chan,
        )
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.block_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_masker_tcn0_skip_block":
        full = _build_hailo_convtas(args)
        module = HailoMaskerTCN0SkipBlock(
            full.masker,
            out_block_idx=args.out_block_idx,
            in_block_idx=args.in_block_idx,
            block_chan=args.block_chan,
        )
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.block_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_masker_head_block":
        full = _build_hailo_convtas(args)
        module = HailoMaskerHeadBlock(
            full.masker,
            out_block_idx=args.out_block_idx,
            in_block_idx=args.in_block_idx,
            block_chan=args.block_chan,
        )
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.block_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_source_projector_only":
        full = _build_hailo_convtas(args)
        if full.source_projector is None:
            raise ValueError("convtas_source_projector_only requires n_src > 1 and force_n_src_1 disabled")
        module = full.source_projector
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.n_filters, 1, latent_t)
        return module, dummy

    if args.module == "convtas_source_projector_out0":
        full = _build_hailo_convtas(args)
        if full.source_projector is None:
            raise ValueError("convtas_source_projector_out0 requires source_projector")
        module = HailoSourceProjectorSlice(full.source_projector, out_index=0, slice_width=args.n_filters)
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.n_filters, 1, latent_t)
        return module, dummy

    if args.module == "convtas_source_projector_out1":
        full = _build_hailo_convtas(args)
        if full.source_projector is None:
            raise ValueError("convtas_source_projector_out1 requires source_projector")
        module = HailoSourceProjectorSlice(full.source_projector, out_index=1, slice_width=args.n_filters)
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.n_filters, 1, latent_t)
        return module, dummy

    if args.module == "convtas_decoder_pre_only":
        full = _build_hailo_convtas(args)
        export_n_src = 1 if args.force_n_src_1 else args.n_src
        in_total = args.n_filters * export_n_src
        module = HailoDecoderPreConvOrIdentity1x1(full.decoder_pre, in_total)
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, in_total, 1, latent_t)
        return module, dummy

    if args.module == "convtas_decoder_pre_half0":
        full = _build_hailo_convtas(args)
        export_n_src = 1 if args.force_n_src_1 else args.n_src
        in_total = args.n_filters * export_n_src
        module = HailoDecoderPreSlice(full.decoder_pre, in_total, out_index=0, slice_width=args.n_filters)
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, in_total, 1, latent_t)
        return module, dummy

    if args.module == "convtas_decoder_pre_half1":
        full = _build_hailo_convtas(args)
        export_n_src = 1 if args.force_n_src_1 else args.n_src
        in_total = args.n_filters * export_n_src
        module = HailoDecoderPreSlice(full.decoder_pre, in_total, out_index=1, slice_width=args.n_filters)
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, in_total, 1, latent_t)
        return module, dummy

    if args.module == "convtas_decoder_only":
        full = _build_hailo_convtas(args)
        module = full.decoder
        export_n_src = 1 if args.force_n_src_1 else args.n_src
        if args.decoder_mode == "conv1x1_head":
            in_chan = args.n_filters * export_n_src
        else:
            in_chan = 64 if args.decoder_mode.endswith("64") else 128
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, in_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_decoder_head_src0":
        full = _build_hailo_convtas(args)
        decoder_conv = full.decoder.conv if hasattr(full.decoder, "conv") else full.decoder
        if not isinstance(decoder_conv, nn.Conv2d):
            raise ValueError("convtas_decoder_head_src0 requires decoder_mode=conv1x1_head")
        module = HailoDecoderHeadSingleSrc(decoder_conv, src_index=0)
        export_n_src = 1 if args.force_n_src_1 else args.n_src
        in_chan = args.n_filters * export_n_src
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, in_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_decoder_head_src1":
        full = _build_hailo_convtas(args)
        decoder_conv = full.decoder.conv if hasattr(full.decoder, "conv") else full.decoder
        if not isinstance(decoder_conv, nn.Conv2d):
            raise ValueError("convtas_decoder_head_src1 requires decoder_mode=conv1x1_head")
        module = HailoDecoderHeadSingleSrc(decoder_conv, src_index=1)
        export_n_src = 1 if args.force_n_src_1 else args.n_src
        in_chan = args.n_filters * export_n_src
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, in_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_source_projector_block":
        full = _build_hailo_convtas(args)
        if full.source_projector is None:
            raise ValueError("convtas_source_projector_block requires source_projector")
        src_idx = args.proj_src_idx
        if src_idx < 0 or src_idx >= args.n_src:
            raise ValueError("proj_src_idx out of range")
        in_start = args.in_block_idx * args.block_chan
        out_start = (src_idx * args.n_filters) + (args.out_block_idx * args.block_chan)
        module = HailoConv1x1PartialBlock(
            source_conv=full.source_projector,
            in_start=in_start,
            in_len=args.block_chan,
            out_start=out_start,
            out_len=args.block_chan,
            include_bias=(args.in_block_idx == 0),
        )
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.block_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_decoder_pre_block":
        full = _build_hailo_convtas(args)
        export_n_src = 1 if args.force_n_src_1 else args.n_src
        in_total = args.n_filters * export_n_src
        pre = HailoDecoderPreConvOrIdentity1x1(full.decoder_pre, in_total).pre
        if not isinstance(pre, nn.Conv2d):
            raise ValueError("convtas_decoder_pre_block requires conv-compatible decoder_pre")
        half_idx = args.half_idx
        if half_idx < 0 or half_idx >= export_n_src:
            raise ValueError("half_idx out of range")
        in_start = args.in_block_idx * args.block_chan
        out_start = (half_idx * args.n_filters) + (args.out_block_idx * args.block_chan)
        module = HailoConv1x1PartialBlock(
            source_conv=pre,
            in_start=in_start,
            in_len=args.block_chan,
            out_start=out_start,
            out_len=args.block_chan,
            include_bias=(args.in_block_idx == 0),
        )
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.block_chan, 1, latent_t)
        return module, dummy

    if args.module == "convtas_decoder_head_block":
        full = _build_hailo_convtas(args)
        decoder_conv = full.decoder.conv if hasattr(full.decoder, "conv") else full.decoder
        if not isinstance(decoder_conv, nn.Conv2d):
            raise ValueError("convtas_decoder_head_block requires decoder_mode=conv1x1_head")
        if args.head_src_idx < 0 or args.head_src_idx >= decoder_conv.out_channels:
            raise ValueError("head_src_idx out of range")
        in_start = args.in_block_idx * args.block_chan
        module = HailoConv1x1PartialBlock(
            source_conv=decoder_conv,
            in_start=in_start,
            in_len=args.block_chan,
            out_start=args.head_src_idx,
            out_len=1,
            include_bias=(args.in_block_idx == 0),
        )
        latent_t = max(1, t // max(1, args.encdec_stride))
        dummy = torch.randn(args.batch, args.block_chan, 1, latent_t)
        return module, dummy

    if args.module == "plain_conv1x1":
        module = nn.Conv2d(args.in_chan, args.out_chan, kernel_size=1, bias=True)
        dummy = torch.randn(args.batch, args.in_chan, 1, t)
        return module, dummy

    raise ValueError(f"Unsupported module: {args.module}")


def main():
    parser = argparse.ArgumentParser(description="Export Hailo module to ONNX")
    parser.add_argument(
        "--module",
        required=True,
        choices=[
            "norm",
            "activation",
            "conv1d_block",
            "tdconvnet",
            "encoder",
            "encoder_conv_only",
            "decoder",
            "hailo_convtasnet",
            "convtas_encoder_only",
            "convtas_masker_only",
            "convtas_masker_bottleneck_only",
            "convtas_masker_tcn_block0_only",
            "convtas_masker_mask_head_only",
            "convtas_masker_bottleneck_block",
            "convtas_masker_tcn0_inconv_block",
            "convtas_masker_tcn0_depth_block",
            "convtas_masker_tcn0_res_block",
            "convtas_masker_tcn0_skip_block",
            "convtas_masker_head_block",
            "convtas_source_projector_only",
            "convtas_source_projector_out0",
            "convtas_source_projector_out1",
            "convtas_decoder_pre_only",
            "convtas_decoder_pre_half0",
            "convtas_decoder_pre_half1",
            "convtas_decoder_only",
            "convtas_decoder_head_src0",
            "convtas_decoder_head_src1",
            "convtas_source_projector_block",
            "convtas_decoder_pre_block",
            "convtas_decoder_head_block",
            "plain_conv1x1",
        ],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--time_len", type=int, default=16000)
    parser.add_argument("--in_chan", type=int, default=128)
    parser.add_argument("--out_chan", type=int, default=128)
    parser.add_argument("--n_src", type=int, default=2)
    parser.add_argument("--n_filters", type=int, default=256)
    parser.add_argument("--bn_chan", type=int, default=128)
    parser.add_argument("--hid_chan", type=int, default=256)
    parser.add_argument("--skip_chan", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--mask_act", default="sigmoid")
    parser.add_argument("--norm_mode", default="affine", choices=["affine", "identity"])
    parser.add_argument("--disable_skip", action="store_true")
    parser.add_argument("--truncate_k_blocks", type=int, default=0)
    parser.add_argument("--mask_mul_mode", default="bypass", choices=["normal", "bypass"])
    parser.add_argument("--force_n_src_1", action="store_true")
    parser.add_argument("--skip_topology_mode", default="project", choices=["project", "repeat"])
    parser.add_argument("--decoder_mode", default="conv1x1_head", choices=["conv1x1_head", "reduced_deconv_64", "reduced_deconv_128"])
    parser.add_argument("--encdec_kernel_size", type=int, default=16)
    parser.add_argument("--encdec_stride", type=int, default=8)
    parser.add_argument("--block_chan", type=int, default=64)
    parser.add_argument("--in_block_idx", type=int, default=0)
    parser.add_argument("--out_block_idx", type=int, default=0)
    parser.add_argument("--proj_src_idx", type=int, default=0)
    parser.add_argument("--half_idx", type=int, default=0)
    parser.add_argument("--head_src_idx", type=int, default=0)
    parser.add_argument("--depth_block_idx", type=int, default=0)
    parser.add_argument("--state_dict_path", default="")
    parser.add_argument("--state_dict_strict", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    module, dummy = build_module(args)
    module.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy,
            str(out),
            input_names=["input"],
            output_names=["output"],
            opset_version=12,
            do_constant_folding=True,
        )

    print(f"[OK] Exported {args.module} ONNX to {out}")
    dump_op_inventory(out)


if __name__ == "__main__":
    main()
