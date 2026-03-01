import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
HAILO_ASTEROID_ROOT = REPO_ROOT / "hailo-asteroid"
if str(HAILO_ASTEROID_ROOT) not in sys.path:
    sys.path.insert(0, str(HAILO_ASTEROID_ROOT))

from asteroid.models.hailo_conv_tasnet import HailoConvTasNet
from asteroid.models.hailo_conv_tasnet_submodules import HailoDecoderPreConvOrIdentity1x1
from hailo.hailo_runtime_runner import HailoHEFRunner


@dataclass
class BlockManifest:
    source_blocks: Dict[Tuple[int, int, int], Path]
    decpre_blocks: Dict[Tuple[int, int, int], Path]
    dechead_blocks: Dict[Tuple[int, int], Path]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tile_indices(total_w: int, tile_w: int):
    s = 0
    while s < total_w:
        e = min(total_w, s + tile_w)
        yield s, e
        s = e


def _conv1x1_partial(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    in_start: int,
    in_len: int,
    out_start: int,
    out_len: int,
    include_bias: bool,
) -> torch.Tensor:
    w_blk = w[out_start : out_start + out_len, in_start : in_start + in_len, :, :]
    b_blk = None
    if include_bias and b is not None:
        b_blk = b[out_start : out_start + out_len]
    return F.conv2d(x[:, in_start : in_start + in_len, :, :], w_blk, b_blk)


def _parse_tag(tag: str):
    # allocfix_full_source_s{src}_o{ob}_i{ib}_w256
    # allocfix_block_decpre_h{half}_o{ob}_i{ib}_w256
    # allocfix_block_dechead_s{src}_i{ib}_w256
    if tag.startswith("allocfix_full_source_"):
        p = tag.split("_")
        src = int(p[4][1:])
        ob = int(p[5][1:])
        ib = int(p[6][1:])
        return ("source", src, ob, ib)
    if tag.startswith("allocfix_block_decpre_"):
        p = tag.split("_")
        half = int(p[3][1:])
        ob = int(p[4][1:])
        ib = int(p[5][1:])
        return ("decpre", half, ob, ib)
    if tag.startswith("allocfix_block_dechead_"):
        p = tag.split("_")
        src = int(p[3][1:])
        ib = int(p[4][1:])
        return ("dechead", src, ib)
    return None


def build_manifest(summary_tsv: Path) -> BlockManifest:
    source: Dict[Tuple[int, int, int], Path] = {}
    decpre: Dict[Tuple[int, int, int], Path] = {}
    dechead: Dict[Tuple[int, int], Path] = {}
    with summary_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("hef_success") != "true":
                continue
            tag = row["run_tag"]
            parsed = _parse_tag(tag)
            if parsed is None:
                continue
            kind = parsed[0]
            hef_path = Path(row["hef_path"])
            if kind == "source":
                _, src, ob, ib = parsed
                source[(src, ob, ib)] = hef_path
            elif kind == "decpre":
                _, half, ob, ib = parsed
                decpre[(half, ob, ib)] = hef_path
            else:
                _, src, ib = parsed
                dechead[(src, ib)] = hef_path
    return BlockManifest(source_blocks=source, decpre_blocks=decpre, dechead_blocks=dechead)


class BlockExecutor:
    def run_source(self, src: int, ob: int, ib: int, x_block: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def run_decpre(self, half: int, ob: int, ib: int, x_block: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def run_dechead(self, src: int, ib: int, x_block: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TorchProxyExecutor(BlockExecutor):
    def __init__(self, model: HailoConvTasNet, block_chan: int = 64):
        self.block = block_chan
        self.n_filters = model.encoder.conv.out_channels
        self.n_src = model.model_n_src
        self.source_conv = model.source_projector
        if self.source_conv is None:
            raise RuntimeError("source_projector is None")
        self.pre_conv = HailoDecoderPreConvOrIdentity1x1(model.decoder_pre, self.n_filters * self.n_src).pre
        dec = model.decoder.conv if hasattr(model.decoder, "conv") else model.decoder
        if not isinstance(dec, torch.nn.Conv2d):
            raise RuntimeError("decoder must be conv1x1 for TorchProxyExecutor")
        self.dec_conv = dec

    def run_source(self, src: int, ob: int, ib: int, x_block: torch.Tensor) -> torch.Tensor:
        out_start = (src * self.n_filters) + ob * self.block
        return _conv1x1_partial(
            x_block,
            self.source_conv.weight,
            self.source_conv.bias,
            in_start=ib * self.block,
            in_len=self.block,
            out_start=out_start,
            out_len=self.block,
            include_bias=(ib == 0),
        )

    def run_decpre(self, half: int, ob: int, ib: int, x_block: torch.Tensor) -> torch.Tensor:
        out_start = (half * self.n_filters) + ob * self.block
        return _conv1x1_partial(
            x_block,
            self.pre_conv.weight,
            self.pre_conv.bias,
            in_start=ib * self.block,
            in_len=self.block,
            out_start=out_start,
            out_len=self.block,
            include_bias=(ib == 0),
        )

    def run_dechead(self, src: int, ib: int, x_block: torch.Tensor) -> torch.Tensor:
        return _conv1x1_partial(
            x_block,
            self.dec_conv.weight,
            self.dec_conv.bias,
            in_start=ib * self.block,
            in_len=self.block,
            out_start=src,
            out_len=1,
            include_bias=(ib == 0),
        )


class HailoRuntimeExecutor(BlockExecutor):
    def __init__(self, manifest: BlockManifest):
        self.manifest = manifest
        self._source_runners: Dict[Tuple[int, int, int], HailoHEFRunner] = {}
        self._decpre_runners: Dict[Tuple[int, int, int], HailoHEFRunner] = {}
        self._dechead_runners: Dict[Tuple[int, int], HailoHEFRunner] = {}

    @staticmethod
    def _infer(runner: HailoHEFRunner, x_block: torch.Tensor) -> torch.Tensor:
        y = runner.infer_nchw(x_block)
        return torch.from_numpy(y)

    def _get_source(self, src: int, ob: int, ib: int) -> HailoHEFRunner:
        key = (src, ob, ib)
        if key not in self.manifest.source_blocks:
            raise RuntimeError(f"Missing source HEF in manifest for key={key}")
        if key not in self._source_runners:
            self._source_runners[key] = HailoHEFRunner(self.manifest.source_blocks[key])
        return self._source_runners[key]

    def _get_decpre(self, half: int, ob: int, ib: int) -> HailoHEFRunner:
        key = (half, ob, ib)
        if key not in self.manifest.decpre_blocks:
            raise RuntimeError(f"Missing decoder_pre HEF in manifest for key={key}")
        if key not in self._decpre_runners:
            self._decpre_runners[key] = HailoHEFRunner(self.manifest.decpre_blocks[key])
        return self._decpre_runners[key]

    def _get_dechead(self, src: int, ib: int) -> HailoHEFRunner:
        key = (src, ib)
        if key not in self.manifest.dechead_blocks:
            raise RuntimeError(f"Missing decoder_head HEF in manifest for key={key}")
        if key not in self._dechead_runners:
            self._dechead_runners[key] = HailoHEFRunner(self.manifest.dechead_blocks[key])
        return self._dechead_runners[key]

    def run_source(self, src: int, ob: int, ib: int, x_block: torch.Tensor) -> torch.Tensor:
        return self._infer(self._get_source(src, ob, ib), x_block)

    def run_decpre(self, half: int, ob: int, ib: int, x_block: torch.Tensor) -> torch.Tensor:
        return self._infer(self._get_decpre(half, ob, ib), x_block)

    def run_dechead(self, src: int, ib: int, x_block: torch.Tensor) -> torch.Tensor:
        return self._infer(self._get_dechead(src, ib), x_block)


def reconstruct_with_executor(
    x: torch.Tensor,
    executor: BlockExecutor,
    n_src: int,
    n_filters: int,
    block: int,
    tile_w: int,
):
    n, _, h, w = x.shape
    assert h == 1
    in_blocks = n_filters // block

    proj_tiles = []
    pre_tiles = []
    head_tiles = []

    for s, e in tile_indices(w, tile_w):
        xt = x[:, :, :, s:e]

        src_chunks = []
        for src in range(n_src):
            out_chunks = []
            for ob in range(n_filters // block):
                acc = None
                for ib in range(in_blocks):
                    y = executor.run_source(src, ob, ib, xt)
                    acc = y if acc is None else (acc + y)
                out_chunks.append(acc)
            src_chunks.append(torch.cat(out_chunks, dim=1))
        proj_t = torch.cat(src_chunks, dim=1)
        proj_tiles.append(proj_t)

        pre_half_chunks = []
        decpre_in_blocks = proj_t.shape[1] // block
        for half in range(n_src):
            out_chunks = []
            for ob in range(n_filters // block):
                acc = None
                for ib in range(decpre_in_blocks):
                    y = executor.run_decpre(half, ob, ib, proj_t)
                    acc = y if acc is None else (acc + y)
                out_chunks.append(acc)
            pre_half_chunks.append(torch.cat(out_chunks, dim=1))
        pre_t = torch.cat(pre_half_chunks, dim=1)
        pre_tiles.append(pre_t)

        head_chunks = []
        dechead_in_blocks = pre_t.shape[1] // block
        for src in range(n_src):
            acc = None
            for ib in range(dechead_in_blocks):
                y = executor.run_dechead(src, ib, pre_t)
                acc = y if acc is None else (acc + y)
            head_chunks.append(acc)
        head_t = torch.cat(head_chunks, dim=1)
        head_tiles.append(head_t)

    return torch.cat(proj_tiles, dim=3), torch.cat(pre_tiles, dim=3), torch.cat(head_tiles, dim=3)


def reconstruct_decoder_from_projected_with_executor(
    proj_x: torch.Tensor,
    executor: BlockExecutor,
    n_src: int,
    n_filters: int,
    block: int,
    tile_w: int,
):
    """Run only decoder_pre + decoder_head from already source-projected representation."""
    n, _, h, w = proj_x.shape
    assert h == 1

    pre_tiles = []
    head_tiles = []
    for s, e in tile_indices(w, tile_w):
        pt = proj_x[:, :, :, s:e]

        pre_half_chunks = []
        decpre_in_blocks = pt.shape[1] // block
        for half in range(n_src):
            out_chunks = []
            for ob in range(n_filters // block):
                acc = None
                for ib in range(decpre_in_blocks):
                    y = executor.run_decpre(half, ob, ib, pt)
                    acc = y if acc is None else (acc + y)
                out_chunks.append(acc)
            pre_half_chunks.append(torch.cat(out_chunks, dim=1))
        pre_t = torch.cat(pre_half_chunks, dim=1)
        pre_tiles.append(pre_t)

        head_chunks = []
        dechead_in_blocks = pre_t.shape[1] // block
        for src in range(n_src):
            acc = None
            for ib in range(dechead_in_blocks):
                y = executor.run_dechead(src, ib, pre_t)
                acc = y if acc is None else (acc + y)
            head_chunks.append(acc)
        head_t = torch.cat(head_chunks, dim=1)
        head_tiles.append(head_t)

    return torch.cat(pre_tiles, dim=3), torch.cat(head_tiles, dim=3)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main():
    parser = argparse.ArgumentParser(description="Run tiled decoder-path stitching with torch proxy or HEF runtime backend.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--summary_tsv", default="hailo/module_runs/20260223_052929/allocator_mapping_fixes_summary.tsv")
    parser.add_argument("--backend", choices=["torch_proxy", "hailo_runtime"], default="torch_proxy")
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

    x = torch.randn(args.batch, args.n_filters, 1, args.latent_w)
    source_conv = model.source_projector
    if source_conv is None:
        raise RuntimeError("source_projector unavailable")
    pre_conv = HailoDecoderPreConvOrIdentity1x1(model.decoder_pre, args.n_filters * args.n_src).pre
    dec_conv = model.decoder.conv if hasattr(model.decoder, "conv") else model.decoder

    with torch.no_grad():
        proj_ref = source_conv(x)
        pre_ref = pre_conv(proj_ref)
        head_ref = dec_conv(pre_ref)

    manifest = build_manifest(Path(args.summary_tsv))
    manifest_counts = {
        "source_blocks": len(manifest.source_blocks),
        "decpre_blocks": len(manifest.decpre_blocks),
        "dechead_blocks": len(manifest.dechead_blocks),
    }

    if args.backend == "torch_proxy":
        executor = TorchProxyExecutor(model, block_chan=args.block_chan)
    else:
        executor = HailoRuntimeExecutor(manifest)

    with torch.no_grad():
        proj_out, pre_out, head_out = reconstruct_with_executor(
            x,
            executor,
            n_src=args.n_src,
            n_filters=args.n_filters,
            block=args.block_chan,
            tile_w=args.tile_w,
        )

    metrics = {
        "backend": args.backend,
        "latent_w": args.latent_w,
        "tile_w": args.tile_w,
        "manifest_counts": manifest_counts,
        "source_projector_max_abs": max_abs(proj_ref, proj_out),
        "decoder_pre_max_abs": max_abs(pre_ref, pre_out),
        "decoder_head_max_abs": max_abs(head_ref, head_out),
        "tol": args.tol,
    }
    metrics["all_close"] = (
        metrics["source_projector_max_abs"] <= args.tol
        and metrics["decoder_pre_max_abs"] <= args.tol
        and metrics["decoder_head_max_abs"] <= args.tol
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"[OK] wrote {out}")
    print(json.dumps(metrics, sort_keys=True))

    if not metrics["all_close"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
