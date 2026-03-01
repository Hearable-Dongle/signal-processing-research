import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent
HAILO_ASTEROID_ROOT = REPO_ROOT / "hailo-asteroid"
if str(HAILO_ASTEROID_ROOT) not in sys.path:
    sys.path.insert(0, str(HAILO_ASTEROID_ROOT))

from asteroid.models import ConvTasNet
from general_utils.constants import LIBRIMIX_PATH


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_mix_wav(librimix_root: Path, split: str = "train-360") -> Path:
    candidates = sorted((librimix_root / "Libri2Mix").glob(f"**/wav8k/**/{split}/mix_clean/*.wav"))
    if not candidates:
        candidates = sorted((librimix_root / "Libri2Mix").glob(f"**/{split}/mix_clean/*.wav"))
    if not candidates:
        candidates = sorted((librimix_root / "Libri2Mix").glob("**/wav8k/**/mix_clean/*.wav"))
    if not candidates:
        candidates = sorted((librimix_root / "Libri2Mix").glob("**/mix_clean/*.wav"))
    if not candidates:
        raise FileNotFoundError(f"No Libri2Mix mix_clean wav found under {librimix_root}")
    return candidates[0]


def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    est = est.view(-1)
    ref = ref.view(-1)
    ref = ref - ref.mean()
    est = est - est.mean()
    proj = (torch.dot(est, ref) / (torch.dot(ref, ref) + eps)) * ref
    noise = est - proj
    ratio = (torch.sum(proj**2) + eps) / (torch.sum(noise**2) + eps)
    return float(10.0 * torch.log10(ratio).item())


def load_wav_mono(path: Path, target_sr: int) -> torch.Tensor:
    wav_np, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav_np = wav_np.mean(axis=1, keepdims=True).T  # [1, T]
    if sr != target_sr:
        wav_np = resample_poly(wav_np, up=target_sr, down=sr, axis=1).astype(np.float32)
    return torch.from_numpy(wav_np)


def get_refs_from_mix(mix_path: Path, target_sr: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    mix_str = str(mix_path)
    if "/mix_clean/" not in mix_str:
        return None, None
    s1 = Path(mix_str.replace("/mix_clean/", "/s1/"))
    s2 = Path(mix_str.replace("/mix_clean/", "/s2/"))
    if not s1.exists() or not s2.exists():
        return None, None
    return load_wav_mono(s1, target_sr), load_wav_mono(s2, target_sr)


def align_length(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    t = min(a.shape[-1], b.shape[-1])
    return a[..., :t], b[..., :t]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real LibriMix forward pass with Hailo-format ConvTasNet wrapper.")
    parser.add_argument("--output_dir", default="", help="Output directory. Default: hailo/module_runs/<ts>_librimix_forward")
    parser.add_argument("--mix_wav", default="", help="Optional explicit mix_clean wav path.")
    parser.add_argument("--librimix_root", default=str(LIBRIMIX_PATH))
    parser.add_argument("--split", default="train-360", help="Preferred split when auto-picking a wav.")
    parser.add_argument("--model_id", default="mpariente/ConvTasNet_WHAM!_sepclean")
    parser.add_argument("--norm_mode", choices=["channel", "global"], default="global")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / f"hailo/module_runs/{ts}_librimix_forward")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ConvTasNet.from_pretrained(args.model_id).eval()
    model_sr = int(getattr(model, "sample_rate", 8000) or 8000)

    mix_path = Path(args.mix_wav) if args.mix_wav else find_mix_wav(Path(args.librimix_root), split=args.split)
    mix = load_wav_mono(mix_path, target_sr=model_sr)
    mix_batched = mix.unsqueeze(0)  # [1, 1, T]

    with torch.no_grad():
        ref_sep = model(mix_batched).cpu()  # [1, 2, T]

    # Import after reference forward because this module monkey-patches TDConvNet.
    import hailo.convtasnet_to_onnx as c2o

    c2o.NORM_MODE = args.norm_mode

    c2o.convert_model_to_4d(model)
    export_model = c2o.HailoExportWrapper(
        model,
        mask_mul_mode="normal",
        force_n_src_1=False,
        bypass_concat=False,
        skip_topology_mode="concat",
        deconv_mode="grouped",
    ).eval()

    mix_4d = mix_batched.unsqueeze(2)  # [1, 1, 1, T]
    with torch.no_grad():
        hailo_sep = export_model(mix_4d).cpu()  # [1, 2, T]

    ref_sep, hailo_sep = align_length(ref_sep, hailo_sep)
    parity_max_abs = float((ref_sep - hailo_sep).abs().max().item())

    # Save outputs.
    sf.write(str(out_dir / "mix.wav"), mix.cpu().squeeze(0).numpy(), model_sr)
    sf.write(str(out_dir / "sep_ref_src1.wav"), ref_sep[0, 0].numpy(), model_sr)
    sf.write(str(out_dir / "sep_ref_src2.wav"), ref_sep[0, 1].numpy(), model_sr)
    sf.write(str(out_dir / "sep_hailo_src1.wav"), hailo_sep[0, 0].numpy(), model_sr)
    sf.write(str(out_dir / "sep_hailo_src2.wav"), hailo_sep[0, 1].numpy(), model_sr)

    metrics = {
        "mix_wav": str(mix_path),
        "model_id": args.model_id,
        "sample_rate": model_sr,
        "parity_max_abs_vs_reference": parity_max_abs,
    }

    gt_s1, gt_s2 = get_refs_from_mix(mix_path, target_sr=model_sr)
    if gt_s1 is not None and gt_s2 is not None:
        est1 = hailo_sep[0, 0].unsqueeze(0)
        est2 = hailo_sep[0, 1].unsqueeze(0)
        est1, gt_s1 = align_length(est1, gt_s1)
        est2, gt_s2 = align_length(est2, gt_s2)
        p1 = si_snr(est1, gt_s1) + si_snr(est2, gt_s2)
        p2 = si_snr(est1, gt_s2) + si_snr(est2, gt_s1)
        if p2 > p1:
            est1, est2 = est2, est1
            p1 = p2
        metrics["si_snr_hailo_src1_vs_gt_s1_db"] = si_snr(est1, gt_s1)
        metrics["si_snr_hailo_src2_vs_gt_s2_db"] = si_snr(est2, gt_s2)
        metrics["si_snr_hailo_pair_sum_db"] = p1

    metrics_path = out_dir / "librimix_hailo_forward_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(f"[OK] mix: {mix_path}")
    print(f"[OK] outputs: {out_dir}")
    print(f"[OK] metrics: {metrics_path}")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
