import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
HAILO_ASTEROID_ROOT = REPO_ROOT / "hailo-asteroid"
if str(HAILO_ASTEROID_ROOT) not in sys.path:
    sys.path.insert(0, str(HAILO_ASTEROID_ROOT))

from asteroid.models.hailo_conv_tasnet import HailoConvTasNet
from hailo.hef_tiled_decoder_path import (
    HailoRuntimeExecutor,
    TorchProxyExecutor,
    build_manifest,
    reconstruct_decoder_from_projected_with_executor,
    reconstruct_with_executor,
)
from hailo.masker_tiled_path import (
    HailoRuntimeMaskerExecutor,
    TorchProxyMaskerExecutor,
    parse_manifest as parse_masker_manifest,
    reconstruct_masker_tiled,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        mask_mul_mode="normal",
        force_n_src_1=False,
        skip_topology_mode="project",
        decoder_mode="conv1x1_head",
        truncate_k_blocks=1,
    ).eval()

    wav = torch.randn(args.batch, 1, 1, args.wav_t)

    with torch.no_grad():
        tf_ref = model.encoder(wav)
        mask_ref = model.masker(tf_ref)
        tf_exp_ref = model._expand_tf(tf_ref)
        masked_ref = mask_ref * tf_exp_ref
        dec_ref = model.decoder(model.decoder_pre(masked_ref))
        out_ref = dec_ref.squeeze(2)

    with torch.no_grad():
        if args.backend == "torch_proxy":
            masker_ex = TorchProxyMaskerExecutor(model, block_chan=args.block_chan)
            dec_ex = TorchProxyExecutor(model, block_chan=args.block_chan)
        else:
            dec_manifest = build_manifest(Path(args.decoder_summary_tsv))
            masker_manifest = parse_masker_manifest(Path(args.masker_summary_tsv))
            dec_ex = HailoRuntimeExecutor(dec_manifest)
            masker_ex = HailoRuntimeMaskerExecutor(model, manifest=masker_manifest, block_chan=args.block_chan)

        mask_parts = reconstruct_masker_tiled(tf_ref, masker_ex, tile_w=args.tile_w)
        mask_tiled = mask_parts["est_mask"]
        tf_exp_tiled, _, _ = reconstruct_with_executor(
            tf_ref,
            dec_ex,
            n_src=args.n_src,
            n_filters=args.n_filters,
            block=args.block_chan,
            tile_w=args.tile_w,
        )
        masked_tiled = mask_tiled * tf_exp_tiled
        pre_tiled, dec_tiled = reconstruct_decoder_from_projected_with_executor(
            masked_tiled,
            dec_ex,
            n_src=args.n_src,
            n_filters=args.n_filters,
            block=args.block_chan,
            tile_w=args.tile_w,
        )
        out_tiled = dec_tiled.squeeze(2)

    metrics = {
        "backend": args.backend,
        "wav_t": args.wav_t,
        "latent_w": int(tf_ref.shape[-1]),
        "tile_w": args.tile_w,
        "masker_output_max_abs": max_abs(mask_ref, mask_tiled),
        "masked_rep_max_abs": max_abs(masked_ref, masked_tiled),
        "decoder_output_max_abs": max_abs(dec_ref, dec_tiled),
        "final_output_max_abs": max_abs(out_ref, out_tiled),
        "tol": args.tol,
    }
    metrics["all_close"] = (
        metrics["masker_output_max_abs"] <= max(args.tol, 2e-5)
        and metrics["masked_rep_max_abs"] <= max(args.tol, 2e-5)
        and metrics["decoder_output_max_abs"] <= max(args.tol, 2e-5)
        and metrics["final_output_max_abs"] <= max(args.tol, 2e-5)
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Full-chain tiled/blockwise parity check (torch proxy backend).")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--backend", choices=["torch_proxy", "hailo_runtime"], default="torch_proxy")
    parser.add_argument("--decoder_summary_tsv", default="hailo/module_runs/20260223_052929/allocator_mapping_fixes_summary.tsv")
    parser.add_argument("--masker_summary_tsv", default="hailo/module_runs/20260223_082226/masker_allocator_fixes_summary.tsv")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n_src", type=int, default=2)
    parser.add_argument("--n_filters", type=int, default=256)
    parser.add_argument("--bn_chan", type=int, default=128)
    parser.add_argument("--hid_chan", type=int, default=256)
    parser.add_argument("--skip_chan", type=int, default=128)
    parser.add_argument("--encdec_kernel_size", type=int, default=16)
    parser.add_argument("--encdec_stride", type=int, default=8)
    parser.add_argument("--wav_t", type=int, default=16000)
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
