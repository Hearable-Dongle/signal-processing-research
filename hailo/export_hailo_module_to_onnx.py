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
HAILO_ASTEROID_ROOT = REPO_ROOT / "hailo-asteroid"
if str(HAILO_ASTEROID_ROOT) not in sys.path:
    sys.path.insert(0, str(HAILO_ASTEROID_ROOT))

from asteroid.masknn.hailo_activations import get_hailo_activation
from asteroid.masknn.hailo_convolutional import HailoConv1DBlock2D, HailoConv1DBlockAsTensor, HailoTDConvNet2D
from asteroid.masknn.hailo_norms import HailoChannelAffineNorm2D, HailoIdentityNorm2D
from asteroid.models.hailo_conv_tasnet import HailoConvTasNet, HailoDecoderConv1x1Head, HailoEncoder2D


class ActivationWrapper(nn.Module):
    def __init__(self, act: nn.Module):
        super().__init__()
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


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

    if args.module == "decoder":
        module = HailoDecoderConv1x1Head(args.in_chan, args.n_src)
        dummy = torch.randn(args.batch, args.in_chan, 1, t)
        return module, dummy

    if args.module == "hailo_convtasnet":
        module = HailoConvTasNet(
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
        dummy = torch.randn(args.batch, 1, 1, t)
        return module, dummy

    raise ValueError(f"Unsupported module: {args.module}")


def main():
    parser = argparse.ArgumentParser(description="Export Hailo module to ONNX")
    parser.add_argument("--module", required=True, choices=["norm", "activation", "conv1d_block", "tdconvnet", "encoder", "decoder", "hailo_convtasnet"])
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
