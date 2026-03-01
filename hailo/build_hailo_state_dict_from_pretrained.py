import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
HAILO_ASTEROID_ROOT = REPO_ROOT / "asteroid"
if str(HAILO_ASTEROID_ROOT) not in sys.path:
    sys.path.insert(0, str(HAILO_ASTEROID_ROOT))

from asteroid.models import ConvTasNet
from asteroid.models.hailo_conv_tasnet import HailoConvTasNet
from hailo.convtasnet_to_onnx import convert_model_to_4d, replace_globln_with_affine, replace_prelu_with_relu


def _copy_slice(dst: torch.Tensor, src: torch.Tensor) -> tuple[int, tuple[int, ...]]:
    shape = tuple(min(d, s) for d, s in zip(dst.shape, src.shape))
    if not shape:
        return 0, shape
    slices = tuple(slice(0, n) for n in shape)
    with torch.no_grad():
        dst.zero_()
        dst[slices] = src[slices].to(dtype=dst.dtype)
    copied = 1
    for n in shape:
        copied *= int(n)
    return copied, shape


def _set_decoder_head_identity(model: HailoConvTasNet) -> None:
    # Conv1x1 head has no direct counterpart in Asteroid ConvTasNet decoder.
    # Use deterministic channel averaging per source as a stable baseline.
    dec = model.decoder.conv if hasattr(model.decoder, "conv") else model.decoder
    if not isinstance(dec, torch.nn.Conv2d):
        return
    with torch.no_grad():
        dec.weight.zero_()
        if dec.bias is not None:
            dec.bias.zero_()
        out_ch, in_ch = dec.weight.shape[0], dec.weight.shape[1]
        if out_ch <= 0 or in_ch <= 0:
            return
        per_src = max(1, in_ch // max(1, out_ch))
        for src_idx in range(out_ch):
            start = src_idx * per_src
            end = min(start + per_src, in_ch)
            if end <= start:
                continue
            dec.weight[src_idx, start:end, 0, 0] = 1.0 / float(end - start)


def transfer_pretrained_to_hailo(src_model: ConvTasNet, dst_model: HailoConvTasNet) -> dict[str, object]:
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    copied_params = 0
    copied_elems = 0
    details: list[dict[str, object]] = []

    def assign(dst_key: str, src_key: str, reshape_norm_weight: bool = False) -> None:
        nonlocal copied_params, copied_elems
        if dst_key not in dst_sd or src_key not in src_sd:
            return
        src_t = src_sd[src_key]
        dst_t = dst_sd[dst_key]
        if reshape_norm_weight and src_t.ndim == 1 and dst_t.ndim == 4:
            src_t = src_t.view(src_t.shape[0], 1, 1, 1)
        n, shape = _copy_slice(dst_t, src_t)
        details.append({"dst": dst_key, "src": src_key, "copied_shape": list(shape)})
        copied_params += 1
        copied_elems += n

    assign("encoder.conv.weight", "encoder.weight")
    assign("masker.bottleneck_norm.affine.weight", "masker.bottleneck.0.gamma", reshape_norm_weight=True)
    assign("masker.bottleneck_norm.affine.bias", "masker.bottleneck.0.beta")
    assign("masker.bottleneck_conv.weight", "masker.bottleneck.1.weight")
    assign("masker.bottleneck_conv.bias", "masker.bottleneck.1.bias")

    dst_blocks = len(dst_model.masker.tcn)
    src_blocks = 0
    for k in src_sd.keys():
        if k.startswith("masker.TCN.") and ".shared_block.0.weight" in k:
            src_blocks += 1
    n_blocks = min(dst_blocks, src_blocks)

    for i in range(n_blocks):
        sp = f"masker.TCN.{i}"
        dp = f"masker.tcn.{i}"
        assign(f"{dp}.in_conv.weight", f"{sp}.shared_block.0.weight")
        assign(f"{dp}.in_conv.bias", f"{sp}.shared_block.0.bias")
        assign(f"{dp}.in_norm.affine.weight", f"{sp}.shared_block.2.gamma", reshape_norm_weight=True)
        assign(f"{dp}.in_norm.affine.bias", f"{sp}.shared_block.2.beta")
        assign(f"{dp}.depth_conv.weight", f"{sp}.shared_block.3.weight")
        assign(f"{dp}.depth_conv.bias", f"{sp}.shared_block.3.bias")
        assign(f"{dp}.depth_norm.affine.weight", f"{sp}.shared_block.5.gamma", reshape_norm_weight=True)
        assign(f"{dp}.depth_norm.affine.bias", f"{sp}.shared_block.5.beta")
        assign(f"{dp}.res_conv.weight", f"{sp}.res_conv.weight")
        assign(f"{dp}.res_conv.bias", f"{sp}.res_conv.bias")
        assign(f"{dp}.skip_conv.weight", f"{sp}.skip_conv.weight")
        assign(f"{dp}.skip_conv.bias", f"{sp}.skip_conv.bias")

    assign("masker.mask_head.weight", "masker.mask_net.1.weight")
    assign("masker.mask_head.bias", "masker.mask_net.1.bias")

    _set_decoder_head_identity(dst_model)
    dst_model.load_state_dict(dst_sd, strict=True)

    return {
        "copied_params": copied_params,
        "copied_elements": copied_elems,
        "src_tcn_blocks": src_blocks,
        "dst_tcn_blocks": dst_blocks,
        "mapped_tcn_blocks": n_blocks,
        "mapping_details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build HailoConvTasNet state dict from HF Asteroid ConvTasNet pretrained weights."
    )
    parser.add_argument("--model_name", default="JorisCos/ConvTasNet_Libri3Mix_sepclean_8k")
    parser.add_argument("--output", default="")
    parser.add_argument("--n_src", type=int, default=2)
    parser.add_argument("--n_filters", type=int, default=256)
    parser.add_argument("--bn_chan", type=int, default=128)
    parser.add_argument("--hid_chan", type=int, default=256)
    parser.add_argument("--skip_chan", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--conv_kernel_size", type=int, default=3)
    parser.add_argument("--encdec_kernel_size", type=int, default=16)
    parser.add_argument("--encdec_stride", type=int, default=8)
    parser.add_argument("--truncate_k_blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else (REPO_ROOT / f"hailo/module_runs/{ts}/hailo_state_dict_from_pretrained.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".json")

    src = ConvTasNet.from_pretrained(args.model_name).eval()
    replace_prelu_with_relu(src)
    replace_globln_with_affine(src)
    with contextlib.redirect_stdout(io.StringIO()):
        convert_model_to_4d(src)

    dst = HailoConvTasNet(
        n_src=args.n_src,
        n_filters=args.n_filters,
        kernel_size=args.encdec_kernel_size,
        stride=args.encdec_stride,
        bn_chan=args.bn_chan,
        hid_chan=args.hid_chan,
        skip_chan=args.skip_chan,
        n_blocks=args.n_blocks,
        n_repeats=args.n_repeats,
        conv_kernel_size=args.conv_kernel_size,
        mask_act="sigmoid",
        norm_mode="affine",
        mask_mul_mode="normal",
        force_n_src_1=False,
        skip_topology_mode="project",
        decoder_mode="conv1x1_head",
        truncate_k_blocks=args.truncate_k_blocks,
    ).eval()

    transfer_stats = transfer_pretrained_to_hailo(src, dst)
    payload = {
        "state_dict": dst.state_dict(),
        "meta": {
            "source_model_name": args.model_name,
            "target_config": {
                "n_src": args.n_src,
                "n_filters": args.n_filters,
                "bn_chan": args.bn_chan,
                "hid_chan": args.hid_chan,
                "skip_chan": args.skip_chan,
                "n_blocks": args.n_blocks,
                "n_repeats": args.n_repeats,
                "conv_kernel_size": args.conv_kernel_size,
                "encdec_kernel_size": args.encdec_kernel_size,
                "encdec_stride": args.encdec_stride,
                "truncate_k_blocks": args.truncate_k_blocks,
                "decoder_mode": "conv1x1_head",
                "skip_topology_mode": "project",
            },
            "transfer_stats": transfer_stats,
        },
    }
    torch.save(payload, out_path)
    meta_path.write_text(json.dumps(payload["meta"], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[OK] wrote state dict: {out_path}")
    print(f"[OK] wrote metadata: {meta_path}")
    print(
        json.dumps(
            {
                "source_model_name": args.model_name,
                "output": str(out_path),
                "copied_params": transfer_stats["copied_params"],
                "copied_elements": transfer_stats["copied_elements"],
                "mapped_tcn_blocks": transfer_stats["mapped_tcn_blocks"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
