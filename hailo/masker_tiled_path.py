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
from hailo.hailo_runtime_runner import HailoHEFRunner


@dataclass
class MaskerManifest:
    bottleneck_blocks: Dict[Tuple[int, int], Path]
    tcn0_in_blocks: Dict[Tuple[int, int], Path]
    tcn0_depth_blocks: Dict[int, Path]
    tcn0_res_blocks: Dict[Tuple[int, int], Path]
    tcn0_skip_blocks: Dict[Tuple[int, int], Path]
    head_blocks: Dict[Tuple[int, int], Path]


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


def parse_manifest(summary_tsv: Path) -> MaskerManifest:
    bneck: Dict[Tuple[int, int], Path] = {}
    tcn_in: Dict[Tuple[int, int], Path] = {}
    depth: Dict[int, Path] = {}
    res: Dict[Tuple[int, int], Path] = {}
    skip: Dict[Tuple[int, int], Path] = {}
    head: Dict[Tuple[int, int], Path] = {}
    with summary_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("hef_success") != "true":
                continue
            tag = row["run_tag"]
            hef_path = Path(row["hef_path"])
            parts = tag.split("_")
            if tag.startswith("allocfix_masker_bneck_o"):
                bneck[(int(parts[3][1:]), int(parts[4][1:]))] = hef_path
            elif tag.startswith("allocfix_masker_tcn0_in_o"):
                tcn_in[(int(parts[4][1:]), int(parts[5][1:]))] = hef_path
            elif tag.startswith("allocfix_masker_tcn0_depth_b"):
                depth[int(parts[4][1:])] = hef_path
            elif tag.startswith("allocfix_masker_tcn0_res_o"):
                res[(int(parts[4][1:]), int(parts[5][1:]))] = hef_path
            elif tag.startswith("allocfix_masker_tcn0_skip_o"):
                skip[(int(parts[4][1:]), int(parts[5][1:]))] = hef_path
            elif tag.startswith("allocfix_masker_head_o"):
                head[(int(parts[3][1:]), int(parts[4][1:]))] = hef_path
    return MaskerManifest(
        bottleneck_blocks=bneck,
        tcn0_in_blocks=tcn_in,
        tcn0_depth_blocks=depth,
        tcn0_res_blocks=res,
        tcn0_skip_blocks=skip,
        head_blocks=head,
    )


def conv1x1_partial(
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
    return F.conv2d(x[:, in_start : in_start + in_len, :, :], w, b)


def depthwise_partial_tiled(
    x: torch.Tensor,
    depth_conv: torch.nn.Conv2d,
    block_idx: int,
    block_chan: int,
    tile_w: int,
) -> torch.Tensor:
    start = block_idx * block_chan
    end = start + block_chan
    xch = x[:, start:end, :, :]
    w = depth_conv.weight[start:end, :, :, :]
    b = depth_conv.bias[start:end] if depth_conv.bias is not None else None
    k = depth_conv.kernel_size[1]
    d = depth_conv.dilation[1]
    radius = ((k - 1) * d) // 2
    n, c, h, wtot = xch.shape
    assert h == 1
    outs = []
    for s, e in tile_indices(wtot, tile_w):
        l = max(0, s - radius)
        r = min(wtot, e + radius)
        xs = xch[:, :, :, l:r]
        ys = F.conv2d(
            xs,
            w,
            b,
            stride=depth_conv.stride,
            padding=depth_conv.padding,
            dilation=depth_conv.dilation,
            groups=block_chan,
        )
        crop_l = s - l
        crop_r = crop_l + (e - s)
        outs.append(ys[:, :, :, crop_l:crop_r])
    return torch.cat(outs, dim=3)


class TorchProxyMaskerExecutor:
    def __init__(self, model: HailoConvTasNet, block_chan: int = 64):
        self.model = model
        self.masker = model.masker
        self.block = block_chan
        self.bn_chan = self.masker.bn_chan
        self.hid_chan = self.masker.hid_chan
        self.skip_chan = self.masker.skip_chan
        self.n_filters = model.encoder.conv.out_channels
        self.n_src = model.model_n_src
        self.block0 = self.masker.tcn[0]

    def bneck(self, x: torch.Tensor, ob: int, ib: int) -> torch.Tensor:
        x_norm = self.masker.bottleneck_norm(x)
        return conv1x1_partial(
            x_norm,
            self.masker.bottleneck_conv.weight,
            self.masker.bottleneck_conv.bias,
            in_start=ib * self.block,
            in_len=self.block,
            out_start=ob * self.block,
            out_len=self.block,
            include_bias=(ib == 0),
        )

    def tcn0_in(self, x: torch.Tensor, ob: int, ib: int) -> torch.Tensor:
        return conv1x1_partial(
            x,
            self.block0.in_conv.weight,
            self.block0.in_conv.bias,
            in_start=ib * self.block,
            in_len=self.block,
            out_start=ob * self.block,
            out_len=self.block,
            include_bias=(ib == 0),
        )

    def tcn0_depth(self, x: torch.Tensor, db: int, tile_w: int) -> torch.Tensor:
        return depthwise_partial_tiled(x, self.block0.depth_conv, db, self.block, tile_w=tile_w)

    def tcn0_res(self, x: torch.Tensor, ob: int, ib: int) -> torch.Tensor:
        return conv1x1_partial(
            x,
            self.block0.res_conv.weight,
            self.block0.res_conv.bias,
            in_start=ib * self.block,
            in_len=self.block,
            out_start=ob * self.block,
            out_len=self.block,
            include_bias=(ib == 0),
        )

    def tcn0_skip(self, x: torch.Tensor, ob: int, ib: int) -> torch.Tensor:
        if self.block0.skip_conv is None:
            raise RuntimeError("skip conv unavailable")
        return conv1x1_partial(
            x,
            self.block0.skip_conv.weight,
            self.block0.skip_conv.bias,
            in_start=ib * self.block,
            in_len=self.block,
            out_start=ob * self.block,
            out_len=self.block,
            include_bias=(ib == 0),
        )

    def head(self, x: torch.Tensor, ob: int, ib: int) -> torch.Tensor:
        return conv1x1_partial(
            x,
            self.masker.mask_head.weight,
            self.masker.mask_head.bias,
            in_start=ib * self.block,
            in_len=self.block,
            out_start=ob * self.block,
            out_len=self.block,
            include_bias=(ib == 0),
        )


class HailoRuntimeMaskerExecutor:
    def __init__(self, model: HailoConvTasNet, manifest: MaskerManifest, block_chan: int = 64):
        self.model = model
        self.masker = model.masker
        self.block = block_chan
        self.bn_chan = self.masker.bn_chan
        self.hid_chan = self.masker.hid_chan
        self.skip_chan = self.masker.skip_chan
        self.n_filters = model.encoder.conv.out_channels
        self.n_src = model.model_n_src
        self.block0 = self.masker.tcn[0]
        self.manifest = manifest
        self._runners_bneck: Dict[Tuple[int, int], HailoHEFRunner] = {}
        self._runners_tcn_in: Dict[Tuple[int, int], HailoHEFRunner] = {}
        self._runners_tcn_depth: Dict[int, HailoHEFRunner] = {}
        self._runners_tcn_res: Dict[Tuple[int, int], HailoHEFRunner] = {}
        self._runners_tcn_skip: Dict[Tuple[int, int], HailoHEFRunner] = {}
        self._runners_head: Dict[Tuple[int, int], HailoHEFRunner] = {}

    @staticmethod
    def _infer(runner: HailoHEFRunner, x: torch.Tensor) -> torch.Tensor:
        y = runner.infer_nchw(x)
        return torch.from_numpy(y)

    def _get_runner(self, cache: Dict, manifest_dict: Dict, key):
        if key not in manifest_dict:
            raise RuntimeError(f"Missing HEF in masker manifest for key={key}")
        if key not in cache:
            cache[key] = HailoHEFRunner(manifest_dict[key])
        return cache[key]

    def bneck(self, x: torch.Tensor, ob: int, ib: int) -> torch.Tensor:
        runner = self._get_runner(
            self._runners_bneck,
            self.manifest.bottleneck_blocks,
            (ob, ib),
        )
        return self._infer(runner, x)

    def tcn0_in(self, x: torch.Tensor, ob: int, ib: int) -> torch.Tensor:
        runner = self._get_runner(
            self._runners_tcn_in,
            self.manifest.tcn0_in_blocks,
            (ob, ib),
        )
        return self._infer(runner, x)

    def tcn0_depth(self, x: torch.Tensor, db: int, tile_w: int) -> torch.Tensor:
        # depthwise HEFs are already tile-safe; tile_w is accepted for interface compatibility.
        _ = tile_w
        runner = self._get_runner(
            self._runners_tcn_depth,
            self.manifest.tcn0_depth_blocks,
            db,
        )
        return self._infer(runner, x)

    def tcn0_res(self, x: torch.Tensor, ob: int, ib: int) -> torch.Tensor:
        runner = self._get_runner(
            self._runners_tcn_res,
            self.manifest.tcn0_res_blocks,
            (ob, ib),
        )
        return self._infer(runner, x)

    def tcn0_skip(self, x: torch.Tensor, ob: int, ib: int) -> torch.Tensor:
        runner = self._get_runner(
            self._runners_tcn_skip,
            self.manifest.tcn0_skip_blocks,
            (ob, ib),
        )
        return self._infer(runner, x)

    def head(self, x: torch.Tensor, ob: int, ib: int) -> torch.Tensor:
        runner = self._get_runner(
            self._runners_head,
            self.manifest.head_blocks,
            (ob, ib),
        )
        return self._infer(runner, x)


def reconstruct_masker_tiled(
    x: torch.Tensor,
    ex,
    tile_w: int = 256,
):
    block = ex.block
    n_filters = ex.n_filters
    bn_chan = ex.bn_chan
    hid_chan = ex.hid_chan
    skip_chan = ex.skip_chan
    head_out_chan = ex.n_src * ex.masker.out_chan

    # bottleneck
    bneck_chunks = []
    for ob in range(bn_chan // block):
        acc = None
        for ib in range(n_filters // block):
            y = ex.bneck(x, ob, ib)
            acc = y if acc is None else (acc + y)
        bneck_chunks.append(acc)
    bneck = torch.cat(bneck_chunks, dim=1)

    # tcn0 inconv (linear contributions, then global act+norm)
    in_chunks = []
    for ob in range(hid_chan // block):
        acc = None
        for ib in range(bn_chan // block):
            y = ex.tcn0_in(bneck, ob, ib)
            acc = y if acc is None else (acc + y)
        in_chunks.append(acc)
    tcn_in_linear = torch.cat(in_chunks, dim=1)
    tcn_in = ex.block0.in_norm(ex.block0.in_act(tcn_in_linear))

    # tcn0 depthwise (linear contributions, then global act+norm)
    depth_chunks = [ex.tcn0_depth(tcn_in, db, tile_w=tile_w) for db in range(hid_chan // block)]
    tcn_depth_linear = torch.cat(depth_chunks, dim=1)
    tcn_depth = ex.block0.depth_norm(ex.block0.depth_act(tcn_depth_linear))

    # tcn0 residual
    res_chunks = []
    for ob in range(bn_chan // block):
        acc = None
        for ib in range(hid_chan // block):
            y = ex.tcn0_res(tcn_depth, ob, ib)
            acc = y if acc is None else (acc + y)
        res_chunks.append(acc)
    residual = torch.cat(res_chunks, dim=1)

    # tcn0 skip
    skip_chunks = []
    for ob in range(skip_chan // block):
        acc = None
        for ib in range(hid_chan // block):
            y = ex.tcn0_skip(tcn_depth, ob, ib)
            acc = y if acc is None else (acc + y)
        skip_chunks.append(acc)
    skip = torch.cat(skip_chunks, dim=1)

    block_out = bneck + residual

    # head + output activation
    head_chunks = []
    for ob in range(head_out_chan // block):
        acc = None
        for ib in range(skip_chan // block):
            y = ex.head(skip, ob, ib)
            acc = y if acc is None else (acc + y)
        head_chunks.append(acc)
    score = torch.cat(head_chunks, dim=1)
    est_mask = ex.masker.output_act(score)

    return {
        "bneck": bneck,
        "tcn_in": tcn_in,
        "tcn_depth": tcn_depth,
        "residual": residual,
        "skip": skip,
        "block_out": block_out,
        "score": score,
        "est_mask": est_mask,
    }


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
    m = model.masker
    block0 = m.tcn[0]
    x = torch.randn(args.batch, args.n_filters, 1, args.latent_w)

    with torch.no_grad():
        bneck_ref = m.bottleneck_conv(m.bottleneck_norm(x))
        t_in_ref = block0.in_norm(block0.in_act(block0.in_conv(bneck_ref)))
        t_depth_ref = block0.depth_norm(block0.depth_act(block0.depth_conv(t_in_ref)))
        residual_ref = block0.res_conv(t_depth_ref)
        skip_ref = block0.skip_conv(t_depth_ref) if block0.skip_conv is not None else None
        block_out_ref = bneck_ref + residual_ref
        mask_input_ref = skip_ref if skip_ref is not None else block_out_ref
        score_ref = m.mask_head(mask_input_ref)
        est_mask_ref = m.output_act(score_ref)

    manifest = parse_manifest(Path(args.summary_tsv))
    manifest_counts = {
        "bottleneck_blocks": len(manifest.bottleneck_blocks),
        "tcn0_in_blocks": len(manifest.tcn0_in_blocks),
        "tcn0_depth_blocks": len(manifest.tcn0_depth_blocks),
        "tcn0_res_blocks": len(manifest.tcn0_res_blocks),
        "tcn0_skip_blocks": len(manifest.tcn0_skip_blocks),
        "head_blocks": len(manifest.head_blocks),
    }

    if args.backend == "torch_proxy":
        ex = TorchProxyMaskerExecutor(model, block_chan=args.block_chan)
    else:
        ex = HailoRuntimeMaskerExecutor(model, manifest=manifest, block_chan=args.block_chan)
    out = reconstruct_masker_tiled(x, ex, tile_w=args.tile_w)

    metrics = {
        "backend": args.backend,
        "latent_w": args.latent_w,
        "tile_w": args.tile_w,
        "manifest_counts": manifest_counts,
        "bottleneck_max_abs": max_abs(bneck_ref, out["bneck"]),
        "tcn0_in_max_abs": max_abs(t_in_ref, out["tcn_in"]),
        "tcn0_depth_max_abs": max_abs(t_depth_ref, out["tcn_depth"]),
        "tcn0_residual_max_abs": max_abs(residual_ref, out["residual"]),
        "tcn0_skip_max_abs": max_abs(skip_ref, out["skip"]) if skip_ref is not None else 0.0,
        "mask_score_max_abs": max_abs(score_ref, out["score"]),
        "mask_output_max_abs": max_abs(est_mask_ref, out["est_mask"]),
        "tol": args.tol,
    }
    metrics["all_close"] = (
        metrics["bottleneck_max_abs"] <= args.tol
        and metrics["tcn0_in_max_abs"] <= args.tol
        and metrics["tcn0_depth_max_abs"] <= max(args.tol, 2e-5)
        and metrics["tcn0_residual_max_abs"] <= args.tol
        and metrics["tcn0_skip_max_abs"] <= args.tol
        and metrics["mask_score_max_abs"] <= args.tol
        and metrics["mask_output_max_abs"] <= max(args.tol, 2e-5)
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate tiled/blockwise masker path reconstruction.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--summary_tsv", required=True)
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
