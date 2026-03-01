import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
HAILO_ASTEROID_ROOT = REPO_ROOT / "hailo-asteroid"
if str(HAILO_ASTEROID_ROOT) not in sys.path:
    sys.path.insert(0, str(HAILO_ASTEROID_ROOT))

from asteroid.models.hailo_conv_tasnet import HailoConvTasNet
from asteroid.models.hailo_conv_tasnet_submodules import HailoDecoderPreConvOrIdentity1x1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def conv1x1_block(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    in_start: int,
    in_len: int,
    out_start: int,
    out_len: int,
    include_bias: bool,
) -> torch.Tensor:
    w = weight[out_start : out_start + out_len, in_start : in_start + in_len, :, :]
    b = None
    if include_bias and bias is not None:
        b = bias[out_start : out_start + out_len]
    x_blk = x[:, in_start : in_start + in_len, :, :]
    return F.conv2d(x_blk, w, b)


def tile_indices(total_w: int, tile_w: int):
    s = 0
    while s < total_w:
        e = min(total_w, s + tile_w)
        yield s, e
        s = e


def reconstruct_source_projector_tiled(
    x: torch.Tensor,
    conv: torch.nn.Conv2d,
    n_src: int = 2,
    n_filters: int = 256,
    block: int = 64,
    tile_w: int = 256,
) -> torch.Tensor:
    n, _, h, w = x.shape
    assert h == 1
    outs = []
    for s, e in tile_indices(w, tile_w):
        xt = x[:, :, :, s:e]
        src_chunks = []
        for src_idx in range(n_src):
            out_chunks = []
            for ob in range(n_filters // block):
                acc = None
                for ib in range(n_filters // block):
                    y = conv1x1_block(
                        xt,
                        conv.weight,
                        conv.bias,
                        in_start=ib * block,
                        in_len=block,
                        out_start=(src_idx * n_filters) + (ob * block),
                        out_len=block,
                        include_bias=(ib == 0),
                    )
                    acc = y if acc is None else (acc + y)
                out_chunks.append(acc)
            src_chunks.append(torch.cat(out_chunks, dim=1))
        outs.append(torch.cat(src_chunks, dim=1))
    return torch.cat(outs, dim=3)


def reconstruct_decoder_pre_tiled(
    x: torch.Tensor,
    conv: torch.nn.Conv2d,
    n_src: int = 2,
    n_filters: int = 256,
    block: int = 64,
    tile_w: int = 256,
) -> torch.Tensor:
    _, in_c, h, w = x.shape
    assert h == 1
    assert in_c == n_src * n_filters
    in_blocks = in_c // block
    outs = []
    for s, e in tile_indices(w, tile_w):
        xt = x[:, :, :, s:e]
        half_chunks = []
        for half_idx in range(n_src):
            out_chunks = []
            for ob in range(n_filters // block):
                acc = None
                for ib in range(in_blocks):
                    y = conv1x1_block(
                        xt,
                        conv.weight,
                        conv.bias,
                        in_start=ib * block,
                        in_len=block,
                        out_start=(half_idx * n_filters) + (ob * block),
                        out_len=block,
                        include_bias=(ib == 0),
                    )
                    acc = y if acc is None else (acc + y)
                out_chunks.append(acc)
            half_chunks.append(torch.cat(out_chunks, dim=1))
        outs.append(torch.cat(half_chunks, dim=1))
    return torch.cat(outs, dim=3)


def reconstruct_decoder_head_tiled(
    x: torch.Tensor,
    conv: torch.nn.Conv2d,
    n_src: int = 2,
    block: int = 64,
    tile_w: int = 256,
) -> torch.Tensor:
    _, in_c, h, w = x.shape
    assert h == 1
    assert in_c % block == 0
    in_blocks = in_c // block
    outs = []
    for s, e in tile_indices(w, tile_w):
        xt = x[:, :, :, s:e]
        src_out = []
        for src_idx in range(n_src):
            acc = None
            for ib in range(in_blocks):
                y = conv1x1_block(
                    xt,
                    conv.weight,
                    conv.bias,
                    in_start=ib * block,
                    in_len=block,
                    out_start=src_idx,
                    out_len=1,
                    include_bias=(ib == 0),
                )
                acc = y if acc is None else (acc + y)
            src_out.append(acc)
        outs.append(torch.cat(src_out, dim=1))
    return torch.cat(outs, dim=3)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run_once(args):
    set_seed(args.seed)
    model = HailoConvTasNet(
        n_src=args.n_src,
        n_filters=args.n_filters,
        kernel_size=args.encdec_kernel_size,
        stride=args.encdec_stride,
        bn_chan=args.bn_chan,
        hid_chan=args.hid_chan,
        skip_chan=args.skip_chan,
        n_blocks=1,
        n_repeats=1,
        conv_kernel_size=3,
        mask_act="sigmoid",
        norm_mode="affine",
        mask_mul_mode="bypass",
        force_n_src_1=False,
        skip_topology_mode="project",
        decoder_mode="conv1x1_head",
        truncate_k_blocks=1,
    ).eval()

    latent_w = args.latent_w
    x = torch.randn(args.batch, args.n_filters, 1, latent_w)

    source_conv = model.source_projector
    if source_conv is None:
        raise RuntimeError("source_projector is None; expected n_src=2 with skip_topology_mode=project")
    pre_conv = HailoDecoderPreConvOrIdentity1x1(model.decoder_pre, args.n_filters * args.n_src).pre
    dec_conv = model.decoder.conv if hasattr(model.decoder, "conv") else model.decoder

    with torch.no_grad():
        proj_direct = source_conv(x)
        proj_tiled = reconstruct_source_projector_tiled(
            x,
            source_conv,
            n_src=args.n_src,
            n_filters=args.n_filters,
            block=args.block_chan,
            tile_w=args.tile_w,
        )

        pre_direct = pre_conv(proj_direct)
        pre_tiled = reconstruct_decoder_pre_tiled(
            proj_direct,
            pre_conv,
            n_src=args.n_src,
            n_filters=args.n_filters,
            block=args.block_chan,
            tile_w=args.tile_w,
        )

        head_direct = dec_conv(pre_direct)
        head_tiled = reconstruct_decoder_head_tiled(
            pre_direct,
            dec_conv,
            n_src=args.n_src,
            block=args.block_chan,
            tile_w=args.tile_w,
        )

    metrics = {
        "tile_w": args.tile_w,
        "latent_w": args.latent_w,
        "source_projector_max_abs": max_abs(proj_direct, proj_tiled),
        "decoder_pre_max_abs": max_abs(pre_direct, pre_tiled),
        "decoder_head_max_abs": max_abs(head_direct, head_tiled),
        "all_close_tol": args.tol,
    }
    metrics["all_close"] = (
        metrics["source_projector_max_abs"] <= args.tol
        and metrics["decoder_pre_max_abs"] <= args.tol
        and metrics["decoder_head_max_abs"] <= args.tol
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate temporal tiling + blockwise decoder-path reconstruction.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n_src", type=int, default=2)
    parser.add_argument("--n_filters", type=int, default=256)
    parser.add_argument("--bn_chan", type=int, default=128)
    parser.add_argument("--hid_chan", type=int, default=256)
    parser.add_argument("--skip_chan", type=int, default=128)
    parser.add_argument("--encdec_kernel_size", type=int, default=16)
    parser.add_argument("--encdec_stride", type=int, default=8)
    parser.add_argument("--latent_w", type=int, default=2000)
    parser.add_argument("--tile_w", type=int, default=256)
    parser.add_argument("--block_chan", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    metrics = run_once(args)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(f"[OK] wrote {out}")
    print(json.dumps(metrics, sort_keys=True))
    if not metrics["all_close"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
