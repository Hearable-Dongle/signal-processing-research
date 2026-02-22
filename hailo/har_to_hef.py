import argparse
import json
import re
from pathlib import Path

import numpy as np
from hailo_sdk_client import ClientRunner

SAMPLE_RATE = 16000


def _apply_apu_neg_mantissa_bugfix() -> None:
    """
    Workaround for Hailo DFC bug where ApuNegMantissaCorrection may try to
    process layers (e.g. HailoAvgPool) that do not implement neg_weights().
    """
    try:
        from hailo_model_optimization.algorithms.apu_neg_mantissa_correction import (
            ACTIVATION_CORRECTION_DICT,
            ApuNegMantissaCorrection,
        )
        from hailo_model_optimization.acceleras.hailo_layers.base_hailo_none_nn_core_layer import (
            BaseHailoNonNNCoreLayer,
        )
    except Exception:
        return

    def _safe_should_correct_layer(layer):
        return (
            not isinstance(layer, BaseHailoNonNNCoreLayer)
            and layer.get_activation_name() in ACTIVATION_CORRECTION_DICT
            and hasattr(layer, "neg_weights")
            and hasattr(layer, "activation_atomic_op")
        )

    ApuNegMantissaCorrection.should_correct_layer = staticmethod(_safe_should_correct_layer)


def _load_calibration_data(npz_path: str, num_samples: int) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        if not data.files:
            raise ValueError(f"No arrays found in calibration NPZ: {npz_path}")
        key = "calib_data" if "calib_data" in data.files else data.files[0]
        calib_data = data[key]
    if not isinstance(calib_data, np.ndarray):
        raise TypeError(f"Calibration payload '{key}' must be a numpy array")
    if calib_data.ndim < 2:
        raise ValueError(f"Calibration array must include sample dimension, got shape={calib_data.shape}")
    if num_samples > 0 and calib_data.shape[0] > num_samples:
        calib_data = calib_data[:num_samples]
    return calib_data.astype(np.float32, copy=False)


def _to_nhwc(calib_data: np.ndarray) -> np.ndarray:
    # Target layout is NHWC (N, 1, T, 1). T is inferred from the provided data.
    if calib_data.ndim == 4:
        # NCHW -> NHWC
        if calib_data.shape[1] == 1 and calib_data.shape[2] == 1:
            return np.transpose(calib_data, (0, 2, 3, 1))
        # Already NHWC
        if calib_data.shape[1] == 1 and calib_data.shape[3] == 1:
            return calib_data
    if calib_data.ndim == 3 and calib_data.shape[1] == 1:
        # NCT -> NHWC
        return calib_data[:, :, :, np.newaxis]
    if calib_data.ndim == 2:
        # NT -> NHWC
        return calib_data[:, np.newaxis, :, np.newaxis]
    raise ValueError(
        "Unsupported calibration shape. Expected one of "
        "(N,1,1,T), (N,1,T,1), (N,1,T), (N,T); "
        f"got {calib_data.shape}"
    )


def _get_expected_input_sample_shape(runner: ClientRunner, input_name: str):
    hn = runner.get_hn_dict()
    layers = hn.get("layers", {})
    if input_name not in layers:
        return None
    layer = layers[input_name]
    shapes = layer.get("output_shapes") or layer.get("input_shapes")
    if not shapes:
        return None
    shape = list(shapes[0])
    if len(shape) != 4:
        return None
    return tuple(shape[1:])  # (H, W, C)


def _fit_nhwc_to_expected(calib_data: np.ndarray, expected_hwc, policy: str) -> np.ndarray:
    exp_h, exp_w, exp_c = expected_hwc
    cur_h, cur_w, cur_c = calib_data.shape[1:]

    if exp_h not in (-1, cur_h) and exp_h != 1:
        raise ValueError(f"Unexpected model input H dimension: {expected_hwc}")
    if exp_c not in (-1, cur_c) and exp_c != 1:
        raise ValueError(f"Unexpected model input C dimension: {expected_hwc}")

    if exp_w in (-1, cur_w) or exp_w == cur_w:
        return calib_data

    if policy == "error":
        raise ValueError(
            f"Calibration temporal length mismatch. expected W={exp_w}, got W={cur_w}. "
            "Use --input_length_policy pad|repeat|crop or provide matching clips."
        )

    if cur_w < exp_w:
        if policy == "repeat":
            reps = int(np.ceil(exp_w / cur_w))
            tiled = np.tile(calib_data, (1, 1, reps, 1))
            return tiled[:, :, :exp_w, :]
        out = np.zeros((calib_data.shape[0], cur_h, exp_w, cur_c), dtype=calib_data.dtype)
        out[:, :, :cur_w, :] = calib_data
        return out

    return calib_data[:, :, :exp_w, :]


def _read_text_if_exists(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text()


def _extract_reshape_failure_layers(error_text: str):
    match = re.search(r"Reshape is needed for layers:\s*(.*?)(?:, but adding a reshape has failed\.|$)", error_text, re.S)
    if not match:
        return []
    return [layer.strip() for layer in match.group(1).split(",") if layer.strip()]


def _extract_negexp_failure(error_text: str):
    layer_match = re.search(r"Quantization failed in layer\s+([^\s]+)\s+due to unsupported required slope", error_text)
    shift_match = re.search(r"Desired shift is\s+([0-9.]+)", error_text)
    bits_match = re.search(r"op has only\s+([0-9]+)\s+data bits", error_text)
    if not layer_match:
        return None
    return {
        "layer": layer_match.group(1),
        "desired_shift": float(shift_match.group(1)) if shift_match else None,
        "data_bits": int(bits_match.group(1)) if bits_match else None,
    }


def _normalize_exception(exc: Exception):
    error_text = str(exc)
    etype = type(exc).__name__
    payload = {
        "exception_type": etype,
        "exception_message": error_text,
        "signature": etype,
        "negexp": None,
        "reshape_layers": [],
    }
    if "NegativeSlopeExponentNonFixable" in etype or "unsupported required slope" in error_text:
        neg = _extract_negexp_failure(error_text)
        payload["negexp"] = neg
        if neg and neg.get("layer"):
            payload["signature"] = f"{etype}:{neg['layer']}"
    elif "BadInputsShape" in etype:
        payload["signature"] = f"{etype}:input_shape"
    elif "Reshape is needed for layers" in error_text:
        layers = _extract_reshape_failure_layers(error_text)
        payload["reshape_layers"] = layers
        payload["signature"] = f"{etype}:reshape:{len(layers)}"
    return payload


def _write_failure_report(path: str, exc: Exception) -> None:
    normalized = _normalize_exception(exc)
    lines = [
        f"exception_type: {normalized['exception_type']}",
        f"exception_message: {normalized['exception_message']}",
    ]
    neg = normalized.get("negexp")
    if neg:
        lines.append(f"negative_exponent_layer: {neg.get('layer')}")
        if neg.get("desired_shift") is not None:
            lines.append(f"desired_shift: {neg.get('desired_shift')}")
        if neg.get("data_bits") is not None:
            lines.append(f"data_bits: {neg.get('data_bits')}")
    layers = normalized.get("reshape_layers", [])
    if layers:
        lines.append(f"reshape_layer_count: {len(layers)}")
        lines.append("reshape_layers_head:")
        lines.extend(layers[:50])
    Path(path).write_text("\n".join(lines) + "\n")


def _write_failure_json(path: str, exc: Exception) -> None:
    normalized = _normalize_exception(exc)
    Path(path).write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")


def _append_negative_exponent_override(args: argparse.Namespace, layer_name: str) -> bool:
    script_path = Path(args.debug_script_out)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    existing = script_path.read_text() if script_path.exists() else ""
    layer_key = f"layers=[{layer_name}]"
    if layer_key in existing:
        return False
    # Keep to the minimal syntax for broad SDK compatibility.
    # Some SDK builds reject extended kwargs for this command.
    line = f"model_optimization_config(negative_exponent, layers=[{layer_name}])"
    with script_path.open("a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(line + "\n")
    return True


def _get_hn_layer_names(har_path: str, hw_arch: str):
    runner = ClientRunner(har=har_path, hw_arch=hw_arch)
    hn = runner.get_hn_dict()
    layers = hn.get("layers", {})
    return set(layers.keys())


def _build_model_script(args: argparse.Namespace) -> str:
    script_lines = []
    if not args.enable_checker:
        script_lines.append("model_optimization_config(checker_cfg, policy=disabled)")
    script_lines.append(
        f"model_optimization_config(calibration, batch_size={args.batch_size}, calibset_size={args.num_samples})"
    )
    if args.compiler_optimization_level:
        script_lines.append(f"performance_param(compiler_optimization_level={args.compiler_optimization_level})")
    if args.model_script_file:
        script_lines.append(Path(args.model_script_file).read_text())
    if args.model_script:
        script_lines.append(args.model_script)
    if args.quick_opt:
        script_lines.append("post_quantization_optimization(bias_correction, policy=disabled)")
        script_lines.append("post_quantization_optimization(adaround, policy=disabled)")
        script_lines.append("post_quantization_optimization(finetune, policy=disabled)")
        script_lines.append("model_optimization_config(checker_cfg, policy=disabled)")
    return "\n".join(script_lines).strip() + "\n"


def _run_once(args: argparse.Namespace) -> None:
    print(f"Loading HAR from {args.har_path} (hw_arch={args.hw_arch})")
    runner = ClientRunner(har=args.har_path, hw_arch=args.hw_arch)
    if not args.skip_apu_neg_mantissa_bugfix:
        _apply_apu_neg_mantissa_bugfix()

    if not args.compile_only:
        if args.calib_npz:
            print(f"Loading calibration data from {args.calib_npz}")
            calib_data = _load_calibration_data(args.calib_npz, args.num_samples)
            calib_data = _to_nhwc(calib_data)
            print(f"Calibration data shape after conversion: {calib_data.shape}")
            args.num_samples = int(calib_data.shape[0])
        else:
            print(
                "No --calib_npz provided; generating synthetic calibration data "
                f"(NHWC: ({args.num_samples}, 1, {SAMPLE_RATE}, 1))"
            )
            calib_data = np.random.uniform(-1, 1, (args.num_samples, 1, SAMPLE_RATE, 1)).astype(np.float32)

        input_name = f"{args.model_name}/input_layer1"
        expected_hwc = _get_expected_input_sample_shape(runner, input_name)
        if expected_hwc is not None:
            exp_h, exp_w, exp_c = expected_hwc
            if exp_w not in (-1, calib_data.shape[2]) and args.input_length_policy != "error":
                print(
                    "Warning: calibration temporal length does not match model input. "
                    f"current W={calib_data.shape[2]}, expected W={exp_w}, "
                    f"policy={args.input_length_policy}. "
                    "Padding/repeat/crop can change calibration statistics and affect quantization."
                )
            fitted = _fit_nhwc_to_expected(calib_data, expected_hwc, policy=args.input_length_policy)
            if fitted.shape != calib_data.shape:
                print(
                    "Adjusted calibration shape to match model input: "
                    f"{calib_data.shape} -> {fitted.shape} (expected sample HWC={expected_hwc}, "
                    f"policy={args.input_length_policy})"
                )
            calib_data = fitted

        model_script = _build_model_script(args)
        print(f"Loading model script:\n{model_script}")
        runner.load_model_script(model_script)

        print(f"Starting Optimization (input layer: {input_name})...")
        try:
            runner.optimize({input_name: calib_data})
        except Exception as exc:
            log_path = args.log_failed_layers_path or args.debug_fail_log
            if log_path:
                _write_failure_report(log_path, exc)
                _write_failure_json(log_path + ".json", exc)
                print(f"Failure diagnostics written to {log_path}")
            raise
        print("Optimization complete.")

        if args.resume_optimized_har:
            runner.save_har(args.resume_optimized_har)
            print(f"Saved intermediate optimized HAR to {args.resume_optimized_har}")

        if args.save_optimized_har:
            runner.save_har(args.save_optimized_har)
            print(f"Optimized HAR saved to {args.save_optimized_har}")

    if args.optimize_only:
        print("Optimization-only mode complete.")
        return
    if args.stop_after_optimize:
        print("Stop-after-optimize mode complete.")
        return

    print("Starting Compilation...")
    try:
        hef = runner.compile()
    except Exception as exc:
        log_path = args.log_failed_layers_path or args.debug_fail_log
        if log_path:
            _write_failure_report(log_path, exc)
            _write_failure_json(log_path + ".json", exc)
            print(f"Failure diagnostics written to {log_path}")
        raise

    with open(args.hef_path, "wb") as f:
        f.write(hef)

    print(f"Success! HEF saved to {args.hef_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimize and compile HAR to HEF for Hailo 8")
    parser.add_argument("har_path", nargs="?", default="hailo/convtas.har", help="Input HAR file path")
    parser.add_argument("hef_path", nargs="?", default="hailo/convtas.hef", help="Output HEF file path")
    parser.add_argument("--model_name", default="convtas", help="Model name (used to derive input layer name)")
    parser.add_argument("--hw_arch", choices=["hailo8", "hailo8l", "hailo8r"], default="hailo8", help="Target hardware")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of calibration samples to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Calibration batch size (lower saves memory)")
    parser.add_argument("--enable_checker", action="store_true", help="Enable layer noise analysis (high memory usage)")
    parser.add_argument("--model_script", type=str, default=None, help="Additional ALLS model script commands")
    parser.add_argument("--model_script_file", type=str, default=None, help="Path to text file with ALLS commands")
    parser.add_argument(
        "--compiler_optimization_level",
        choices=["max"],
        default=None,
        help="Optional performance tuning (compiler-side).",
    )
    parser.add_argument("--calib_npz", type=str, default=None, help="Calibration dataset NPZ path")
    parser.add_argument("--save_optimized_har", type=str, default=None, help="Path to save optimized HAR for debugging")
    parser.add_argument("--compile_only", action="store_true", help="Skip optimize and compile HAR directly")
    parser.add_argument("--optimize_only", action="store_true", help="Run optimize only and skip compile")
    parser.add_argument(
        "--log_failed_layers_path",
        type=str,
        default=None,
        help="Path to write failure diagnostics when optimize/compile fails",
    )
    parser.add_argument(
        "--input_length_policy",
        choices=["pad", "repeat", "crop", "error"],
        default="pad",
        help="How to handle calibration temporal-length mismatch against model input.",
    )
    parser.add_argument(
        "--skip_apu_neg_mantissa_bugfix",
        action="store_true",
        help="Disable runtime workaround for known APU neg mantissa correction bug in some SDK versions.",
    )
    parser.add_argument(
        "--quick_opt",
        action="store_true",
        help="Disable long post-quantization algorithms for faster compile iteration.",
    )
    parser.add_argument("--debug_loop", action="store_true", help="Iteratively retry optimization/compile with auto overrides.")
    parser.add_argument("--max_debug_iters", type=int, default=8, help="Maximum debug loop iterations.")
    parser.add_argument(
        "--max_debug_stagnation_iters",
        type=int,
        default=2,
        help="Stop debug loop after this many consecutive iterations with no new auto-overrides.",
    )
    parser.add_argument(
        "--debug_script_out",
        type=str,
        default="hailo/debug_overrides.alls",
        help="Generated model-script override file for debug loop.",
    )
    parser.add_argument(
        "--debug_fail_log",
        type=str,
        default="hailo/compile_failure.txt",
        help="Failure text artifact path in debug loop.",
    )
    parser.add_argument(
        "--resume_optimized_har",
        type=str,
        default=None,
        help="Optional path to save optimized HAR after each successful optimize attempt.",
    )
    parser.add_argument("--stop_after_optimize", action="store_true", help="Stop after optimize phase in each iteration.")
    parser.add_argument("--negexp_rank", type=int, default=1, choices=[0, 1], help="Auto override rank for negative_exponent.")
    parser.add_argument("--negexp_split_threshold", type=int, default=1, help="Auto override split_threshold for negative_exponent.")
    parser.add_argument(
        "--negexp_auto_clip",
        choices=["enabled", "disabled", "allowed"],
        default="enabled",
        help="Auto override auto_clip for negative_exponent.",
    )
    parser.add_argument(
        "--negexp_auto_remove_offset",
        choices=["enabled", "disabled", "allowed"],
        default="enabled",
        help="Auto override auto_remove_offset for negative_exponent.",
    )

    args = parser.parse_args()
    if args.compile_only and args.optimize_only:
        raise ValueError("--compile_only and --optimize_only are mutually exclusive")
    if args.debug_loop and args.compile_only:
        raise ValueError("--debug_loop cannot be combined with --compile_only")
    if args.debug_loop and not args.calib_npz:
        raise ValueError("--debug_loop requires --calib_npz")

    if args.debug_loop:
        original_model_script_file = args.model_script_file
        args.model_script_file = args.debug_script_out
        hn_layers = _get_hn_layer_names(args.har_path, args.hw_arch)
        print(f"Loaded {len(hn_layers)} HN layer names for debug override validation.")
        out = Path(args.debug_script_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if original_model_script_file and original_model_script_file != args.debug_script_out:
            out.write_text(Path(original_model_script_file).read_text())
        elif not out.exists():
            out.write_text("")

        seen_signatures = []
        stagnant_iters = 0
        for it in range(1, args.max_debug_iters + 1):
            print(f"\n===== Debug Iteration {it}/{args.max_debug_iters} =====")
            try:
                _run_once(args)
                print("Debug loop success.")
                return
            except Exception as exc:
                _write_failure_report(args.debug_fail_log, exc)
                _write_failure_json(args.debug_fail_log + ".json", exc)
                normalized = _normalize_exception(exc)
                sig = normalized["signature"]
                seen_signatures.append(sig)
                neg = normalized.get("negexp")
                override_added = False
                if neg and neg.get("layer"):
                    layer_name = neg["layer"]
                    if layer_name in hn_layers:
                        override_added = _append_negative_exponent_override(args, layer_name)
                        if override_added:
                            print(f"Added negative_exponent override for {layer_name} to {args.debug_script_out}")
                    else:
                        print(
                            "Unresolvable negexp layer in current HN scope; "
                            f"skipping override append: {layer_name}"
                        )

                if override_added:
                    stagnant_iters = 0
                else:
                    stagnant_iters += 1

                print(
                    f"Iteration {it} result: signature={sig}, "
                    f"override_added={override_added}, stagnant_iters={stagnant_iters}/{args.max_debug_stagnation_iters}"
                )

                if _read_text_if_exists(args.debug_script_out).strip():
                    print(f"Active overrides ({args.debug_script_out}):")
                    print(_read_text_if_exists(args.debug_script_out).strip())

                if override_added:
                    continue
                if stagnant_iters >= args.max_debug_stagnation_iters:
                    print("No new overrides added for consecutive iterations. Stopping debug loop.")
                    raise
                # Retry once more in case stochastic behavior changes signature without a new override.
                continue
        raise RuntimeError(f"Debug loop exhausted after {args.max_debug_iters} iterations")

    _run_once(args)


if __name__ == "__main__":
    main()
