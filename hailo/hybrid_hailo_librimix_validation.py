import argparse
import contextlib
import io
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent
for asteroid_root in (REPO_ROOT / "hailo-asteroid", REPO_ROOT / "asteroid"):
    if (asteroid_root / "asteroid").is_dir() and str(asteroid_root) not in sys.path:
        sys.path.insert(0, str(asteroid_root))

from asteroid.models import ConvTasNet
from asteroid.models.hailo_conv_tasnet import HailoConvTasNet
from general_utils.constants import LIBRIMIX_PATH
from hailo.hef_tiled_decoder_path import (
    HailoRuntimeExecutor,
    TorchProxyExecutor,
    build_manifest,
    reconstruct_with_executor,
    validate_manifest_coverage,
)
from hailo.masker_tiled_path import (
    HailoRuntimeMaskerExecutor,
    TorchProxyMaskerExecutor,
    parse_manifest as parse_masker_manifest,
    reconstruct_masker_tiled,
    validate_masker_manifest_coverage,
)
from hailo.hailo_runtime_runner import HailoRuntimeUnavailable


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav.astype(np.float32)
    return resample_poly(wav, up=dst_sr, down=src_sr).astype(np.float32)


def _find_default_sanity_mix() -> Path:
    p = REPO_ROOT / "hailo" / "sanity_librimix3" / "sanity_mix.wav"
    if p.exists():
        return p
    libri3 = Path(LIBRIMIX_PATH) / "Libri3Mix"
    for mix in sorted(libri3.glob("**/mix_clean/*.wav")):
        s1 = Path(str(mix).replace("/mix_clean/", "/s1/"))
        s2 = Path(str(mix).replace("/mix_clean/", "/s2/"))
        s3 = Path(str(mix).replace("/mix_clean/", "/s3/"))
        if s1.exists() and s2.exists() and s3.exists():
            return mix
    raise FileNotFoundError("No Libri3Mix mix_clean sample found")


def _best_perm_metrics(est: torch.Tensor, ref: torch.Tensor) -> tuple[float, list[int]]:
    # est/ref: [n_src, T]
    n = int(min(est.shape[0], ref.shape[0]))
    est = est[:n]
    ref = ref[:n]
    best = float("inf")
    best_p = list(range(n))
    for p in itertools.permutations(range(n)):
        aligned = ref[list(p), :]
        mse = float(torch.mean((est - aligned) ** 2).item())
        if mse < best:
            best = mse
            best_p = list(p)
    return best, best_p


def _best_assignment_metrics(est: torch.Tensor, ref: torch.Tensor) -> tuple[float, list[int], list[int]]:
    # est/ref: [n_src, T]
    ne = int(est.shape[0])
    nr = int(ref.shape[0])
    n = int(min(ne, nr))
    best = float("inf")
    best_est = list(range(n))
    best_ref = list(range(n))

    est_subsets = [tuple(range(ne))] if ne == n else list(itertools.combinations(range(ne), n))
    ref_subsets = [tuple(range(nr))] if nr == n else list(itertools.combinations(range(nr), n))

    for e_idx in est_subsets:
        est_sel = est[list(e_idx), :]
        for r_idx in ref_subsets:
            ref_sel = ref[list(r_idx), :]
            mse, perm = _best_perm_metrics(est_sel, ref_sel)
            if mse < best:
                best = mse
                best_est = list(e_idx)
                best_ref = [int(r_idx[p]) for p in perm]
    return best, best_est, best_ref


def _load_gt_sources_from_mix(mix_path: Path, target_sr: int) -> Optional[torch.Tensor]:
    # Case 1: direct copied sanity assets under hailo/sanity_librimix3
    if mix_path.name == "sanity_mix.wav":
        sdir = mix_path.parent
        paths = [sdir / f"sanity_voice_{i}.wav" for i in [1, 2, 3]]
        if all(p.exists() for p in paths):
            srcs = []
            for p in paths:
                wav, sr = _load_wav(p)
                wav = _resample(wav, sr, target_sr)
                srcs.append(torch.from_numpy(wav))
            t = min(s.shape[-1] for s in srcs)
            return torch.stack([s[:t] for s in srcs], dim=0)

    # Case 2: canonical LibriMix mix_clean path layout
    mix_str = str(mix_path)
    if "/mix_clean/" not in mix_str:
        return None
    paths = [Path(mix_str.replace("/mix_clean/", f"/s{i}/")) for i in [1, 2, 3]]
    if not all(p.exists() for p in paths):
        return None
    srcs = []
    for p in paths:
        wav, sr = _load_wav(p)
        wav = _resample(wav, sr, target_sr)
        srcs.append(torch.from_numpy(wav))
    t = min(s.shape[-1] for s in srcs)
    return torch.stack([s[:t] for s in srcs], dim=0)


def _align_to_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        return x[..., :target_len]
    pad = torch.zeros(*x.shape[:-1], target_len - x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def _decode_masked_to_wave(
    masked_4d: torch.Tensor,
    encoder: torch.nn.Module,
    n_src: int,
    n_filters: int,
    target_len: int,
) -> torch.Tensor:
    # Decode masked latent features back to waveform on CPU using encoder basis.
    # [B, n_src*n_filters, 1, W] -> [B*n_src, n_filters, W] -> conv_transpose1d -> [B, n_src, T]
    b, c, _, w = masked_4d.shape
    if c != n_src * n_filters:
        raise RuntimeError(f"Unexpected masked channels={c}, expected {n_src * n_filters}")
    masked = masked_4d.view(b, n_src, n_filters, w).reshape(b * n_src, n_filters, w)
    conv = encoder.conv
    weight = conv.weight.squeeze(2)
    out = F.conv_transpose1d(
        masked,
        weight,
        bias=None,
        stride=conv.stride[-1],
        padding=conv.padding[-1],
    ).view(b, n_src, -1)
    return _align_to_len(out, target_len)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid Hailo validation on real LibriMix sample with latency and WAV outputs")
    parser.add_argument("--backend", choices=["torch_proxy", "hailo_runtime"], default="hailo_runtime")
    parser.add_argument("--decoder_summary_tsv", default="hailo/module_runs/20260223_052929/allocator_mapping_fixes_summary.tsv")
    parser.add_argument("--masker_summary_tsv", default="hailo/module_runs/20260223_082226/masker_allocator_fixes_summary.tsv")
    parser.add_argument("--mix_wav", default="")
    parser.add_argument("--model_id", default="JorisCos/ConvTasNet_Libri3Mix_sepclean_8k")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--n_src", type=int, default=2)
    parser.add_argument("--n_filters", type=int, default=256)
    parser.add_argument("--bn_chan", type=int, default=128)
    parser.add_argument("--hid_chan", type=int, default=256)
    parser.add_argument("--skip_chan", type=int, default=128)
    parser.add_argument("--encdec_kernel_size", type=int, default=16)
    parser.add_argument("--encdec_stride", type=int, default=8)
    parser.add_argument("--tile_w", type=int, default=256)
    parser.add_argument("--block_chan", type=int, default=64)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--max_seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / f"hailo/module_runs/{ts}_hybrid_librimix")
    out_dir.mkdir(parents=True, exist_ok=True)

    mix_path = Path(args.mix_wav) if args.mix_wav else _find_default_sanity_mix()
    mix_np, sr = _load_wav(mix_path)
    mix_np = _resample(mix_np, sr, args.sample_rate)
    max_samples = max(1, int(args.sample_rate * args.max_seconds))
    mix_np = mix_np[:max_samples]
    mix = torch.from_numpy(mix_np).view(1, 1, 1, -1)

    ref_model = ConvTasNet.from_pretrained(args.model_id).eval()
    ref_sr = int(getattr(ref_model, "sample_rate", args.sample_rate) or args.sample_rate)
    ref_mix_np = _resample(mix_np, args.sample_rate, ref_sr)
    ref_mix = torch.from_numpy(ref_mix_np).view(1, 1, -1)
    with torch.no_grad():
        ref_out = ref_model(ref_mix).cpu()[0]
    if ref_sr != args.sample_rate:
        ref_out = torch.stack(
            [torch.from_numpy(_resample(ref_out[i].numpy(), ref_sr, args.sample_rate)) for i in range(ref_out.shape[0])],
            dim=0,
        )
    ref_out = _align_to_len(ref_out, mix.shape[-1])

    from hailo.build_hailo_state_dict_from_pretrained import transfer_pretrained_to_hailo
    from hailo.convtasnet_to_onnx import convert_model_to_4d, replace_globln_with_affine, replace_prelu_with_relu

    src_for_transfer = ConvTasNet.from_pretrained(args.model_id).eval()
    replace_prelu_with_relu(src_for_transfer)
    replace_globln_with_affine(src_for_transfer)
    with contextlib.redirect_stdout(io.StringIO()):
        convert_model_to_4d(src_for_transfer)

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
    transfer_pretrained_to_hailo(src_for_transfer, model)

    effective_backend = args.backend
    fallback_reason = None
    dec_manifest = build_manifest(Path(args.decoder_summary_tsv))
    masker_manifest = parse_masker_manifest(Path(args.masker_summary_tsv))
    strict_runtime_safe = args.backend == "hailo_runtime"

    if args.backend == "hailo_runtime":
        try:
            validate_manifest_coverage(
                dec_manifest,
                n_src=args.n_src,
                n_filters=args.n_filters,
                block_chan=args.block_chan,
            )
            validate_masker_manifest_coverage(
                masker_manifest,
                n_filters=args.n_filters,
                bn_chan=args.bn_chan,
                hid_chan=args.hid_chan,
                skip_chan=args.skip_chan,
                n_src=args.n_src,
                out_chan=model.masker.out_chan,
                block_chan=args.block_chan,
            )
        except RuntimeError as e:
            effective_backend = "torch_proxy_fallback"
            fallback_reason = f"Manifest coverage fallback: {e}"

    def _build_executors(backend: str):
        if backend == "torch_proxy" or backend == "torch_proxy_fallback":
            return (
                TorchProxyMaskerExecutor(model, block_chan=args.block_chan),
                TorchProxyExecutor(model, block_chan=args.block_chan),
            )
        return (
            HailoRuntimeMaskerExecutor(
                model,
                manifest=masker_manifest,
                block_chan=args.block_chan,
                strict_runtime_safe=strict_runtime_safe,
            ),
            HailoRuntimeExecutor(
                dec_manifest,
                block_chan=args.block_chan,
                strict_runtime_safe=strict_runtime_safe,
            ),
        )

    masker_ex, dec_ex = _build_executors(effective_backend)
    cpu_masker_ex = TorchProxyMaskerExecutor(model, block_chan=args.block_chan)
    cpu_dec_ex = TorchProxyExecutor(model, block_chan=args.block_chan)

    t0 = time.perf_counter()
    with torch.no_grad():
        tf = model.encoder(mix)
    t1 = time.perf_counter()
    with torch.no_grad():
        cpu_mask_parts = reconstruct_masker_tiled(tf, cpu_masker_ex, tile_w=args.tile_w)
        cpu_mask = cpu_mask_parts["est_mask"]
        tf_exp_cpu, _, _ = reconstruct_with_executor(
            tf,
            cpu_dec_ex,
            n_src=args.n_src,
            n_filters=args.n_filters,
            block=args.block_chan,
            tile_w=args.tile_w,
        )
        masked_cpu = cpu_mask * tf_exp_cpu
        pytorch_proxy_out = _decode_masked_to_wave(
            masked_cpu,
            model.encoder,
            n_src=args.n_src,
            n_filters=args.n_filters,
            target_len=mix.shape[-1],
        ).cpu()[0]
    t_cpu = time.perf_counter()
    with torch.no_grad():
        mask_parts = reconstruct_masker_tiled(tf, masker_ex, tile_w=args.tile_w)
        mask = mask_parts["est_mask"]
    t2 = time.perf_counter()
    try:
        with torch.no_grad():
            tf_exp, _, _ = reconstruct_with_executor(
                tf,
                dec_ex,
                n_src=args.n_src,
                n_filters=args.n_filters,
                block=args.block_chan,
                tile_w=args.tile_w,
            )
            masked = mask * tf_exp
            hybrid_out = _decode_masked_to_wave(
                masked,
                model.encoder,
                n_src=args.n_src,
                n_filters=args.n_filters,
                target_len=mix.shape[-1],
            ).cpu()[0]
    except (HailoRuntimeUnavailable, Exception) as e:
        if args.backend != "hailo_runtime":
            raise
        effective_backend = "torch_proxy_fallback"
        if fallback_reason is None:
            fallback_reason = f"Runtime execution fallback: {type(e).__name__}: {e}"
        masker_ex, dec_ex = _build_executors("torch_proxy_fallback")
        with torch.no_grad():
            mask_parts = reconstruct_masker_tiled(tf, masker_ex, tile_w=args.tile_w)
            mask = mask_parts["est_mask"]
            tf_exp, _, _ = reconstruct_with_executor(
                tf,
                dec_ex,
                n_src=args.n_src,
                n_filters=args.n_filters,
                block=args.block_chan,
                tile_w=args.tile_w,
            )
            masked = mask * tf_exp
            hybrid_out = _decode_masked_to_wave(
                masked,
                model.encoder,
                n_src=args.n_src,
                n_filters=args.n_filters,
                target_len=mix.shape[-1],
            ).cpu()[0]
    t3 = time.perf_counter()

    # Save listenable outputs.
    sf.write(str(out_dir / "mix.wav"), mix_np, args.sample_rate)
    for i in range(ref_out.shape[0]):
        sf.write(str(out_dir / f"pytorch_full_sep_src{i+1}.wav"), ref_out[i].numpy(), args.sample_rate)
    for i in range(pytorch_proxy_out.shape[0]):
        sf.write(str(out_dir / f"pytorch_proxy_sep_src{i+1}.wav"), pytorch_proxy_out[i].numpy(), args.sample_rate)
    for i in range(hybrid_out.shape[0]):
        sf.write(str(out_dir / f"hybrid_sep_src{i+1}.wav"), hybrid_out[i].numpy(), args.sample_rate)

    gt = _load_gt_sources_from_mix(mix_path, target_sr=args.sample_rate)
    if gt is not None:
        t = min(gt.shape[-1], hybrid_out.shape[-1])
        for i in range(min(3, gt.shape[0])):
            sf.write(str(out_dir / f"gt_voice_{i+1}.wav"), gt[i, :t].numpy(), args.sample_rate)

    metrics = {
        "ok": True,
        "backend": args.backend,
        "effective_backend": effective_backend,
        "strict_runtime_safe": strict_runtime_safe,
        "mix_wav": str(mix_path),
        "model_id": args.model_id,
        "sample_rate_hybrid": args.sample_rate,
        "sample_rate_reference": ref_sr,
        "n_src_hybrid": int(hybrid_out.shape[0]),
        "n_src_reference": int(ref_out.shape[0]),
        "n_src_pytorch_proxy": int(pytorch_proxy_out.shape[0]),
        "input_samples": int(mix.shape[-1]),
        "latent_w": int(tf.shape[-1]),
        "latency_ms_total": float((t3 - t0) * 1000.0),
        "latency_ms_encoder_cpu": float((t1 - t0) * 1000.0),
        "latency_ms_pytorch_path": float((t_cpu - t1) * 1000.0),
        "latency_ms_masker": float((t2 - t1) * 1000.0),
        "latency_ms_decoder": float((t3 - t2) * 1000.0),
        "rtf_total": float((t3 - t0) / max(1e-9, mix.shape[-1] / args.sample_rate)),
    }
    if fallback_reason is not None:
        metrics["backend_fallback_reason"] = fallback_reason

    # Parity between full PyTorch baseline and hybrid path.
    if ref_out.shape[0] > 0 and hybrid_out.shape[0] > 0:
        tt = min(ref_out.shape[-1], hybrid_out.shape[-1])
        mse, est_idx, ref_idx = _best_assignment_metrics(hybrid_out[:, :tt], ref_out[:, :tt])
        est_aligned = hybrid_out[est_idx, :tt]
        ref_aligned = ref_out[ref_idx, :tt]
        max_abs = float(torch.max(torch.abs(est_aligned - ref_aligned)).item())
        metrics["parity_available"] = True
        metrics["parity_hybrid_indices"] = est_idx
        metrics["parity_reference_indices"] = ref_idx
        metrics["parity_mse"] = mse
        metrics["parity_max_abs"] = max_abs
    else:
        metrics["parity_available"] = False
        metrics["parity_note"] = "empty source count in hybrid or reference"

    metrics_path = out_dir / "hybrid_validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    print(f"[OK] outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
