import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RampCase:
    stage: str
    run_tag: str
    mask_mul_mode: str
    truncate_k_blocks: int
    n_blocks: int
    n_repeats: int
    n_filters: int
    bn_chan: int
    hid_chan: int
    skip_chan: int
    input_len: int


def _descendant_pids(root_pid: int) -> list[int]:
    out = subprocess.run(
        ["pgrep", "-P", str(root_pid)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip()
    if not out:
        return []
    descendants = []
    for token in out.splitlines():
        token = token.strip()
        if not token:
            continue
        try:
            pid = int(token)
        except ValueError:
            continue
        descendants.append(pid)
        descendants.extend(_descendant_pids(pid))
    return descendants


def _kill_pids(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _kill_process_tree(root_pid: int) -> None:
    children = _descendant_pids(root_pid)
    _kill_pids(children, signal.SIGTERM)
    _kill_pids([root_pid], signal.SIGTERM)
    time.sleep(1)
    children = _descendant_pids(root_pid)
    _kill_pids(children, signal.SIGKILL)
    _kill_pids([root_pid], signal.SIGKILL)


def _kill_matching_processes(pattern: str) -> None:
    if not pattern:
        return
    proc = subprocess.run(
        ["ps", "-ef"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    pids = []
    for line in proc.stdout.splitlines():
        if pattern not in line or " rg " in line:
            continue
        cols = line.split()
        if len(cols) < 2:
            continue
        try:
            pid = int(cols[1])
        except ValueError:
            continue
        pids.append(pid)
    _kill_pids(pids, signal.SIGTERM)
    time.sleep(1)
    _kill_pids(pids, signal.SIGKILL)


def _run_command(cmd: list[str], env: dict[str, str], timeout_sec: int, run_tag: str = "") -> tuple[int, str, bool]:
    proc = subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout_sec)
        return proc.returncode, stdout or "", False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        _kill_process_tree(proc.pid)
        _kill_matching_processes(run_tag)
        try:
            stdout, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            _kill_process_tree(proc.pid)
            _kill_matching_processes(run_tag)
            stdout, _ = proc.communicate()
        return 124, stdout or "", True


def _ensure_summary(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "run_tag",
                "stage",
                "mask_mul_mode",
                "truncate_k_blocks",
                "n_blocks",
                "n_repeats",
                "n_filters",
                "bn_chan",
                "hid_chan",
                "skip_chan",
                "input_len",
                "har_success",
                "hef_success",
                "exception_type",
                "failure_head",
                "har_path",
                "hef_path",
                "har_log",
                "hef_log",
                "failure_log",
            ]
        )


def _append_summary(path: Path, row: list[str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerow(row)


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_failure(hef_log: Path, failure_json: Path) -> tuple[str, str]:
    if failure_json.exists():
        try:
            payload = json.loads(failure_json.read_text(encoding="utf-8"))
            etype = str(payload.get("exception_type") or "")
            emsg = str(payload.get("exception_message") or "")
            head = emsg.strip().splitlines()[0] if emsg.strip() else ""
            return etype, head[:500]
        except Exception:
            pass

    text = _safe_read_text(hef_log)
    etype_matches = re.findall(r"([A-Za-z]+Exception)", text)
    etype = etype_matches[-1] if etype_matches else "CompileFailed"
    lines = []
    for pattern in (
        r".*Agent infeasible.*",
        r".*Mapping Failed.*",
        r".*BackendAllocatorException.*",
        r".*Exception.*",
        r".*Failed.*",
    ):
        lines = re.findall(pattern, text)
        if lines:
            break
    head = lines[-1].strip() if lines else "compile failed"
    return etype, head[:500]


def _format_bool(v: bool) -> str:
    return "true" if v else "false"


def _extract_failed_families(text: str) -> list[str]:
    families = []
    for line in text.splitlines():
        m = re.search(r"\b([A-Za-z_]*?(?:conv|dw|reshape)[A-Za-z0-9_]*)\s+errors?:", line)
        if m:
            families.append(m.group(1))
    dedup = []
    for fam in families:
        if fam not in dedup:
            dedup.append(fam)
    return dedup


def _write_failure_signature(run_dir: Path, first_fail: dict[str, str] | None) -> None:
    sig_tsv = run_dir / "monolithic_failure_signature.tsv"
    with sig_tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "run_tag",
                "stage",
                "exception_type",
                "first_failed_family",
                "repeated_failed_families",
                "mapping_timeout_flag",
            ]
        )
        if first_fail is None:
            return
        text = "\n".join(
            [
                _safe_read_text(Path(first_fail["hef_log"])),
                _safe_read_text(Path(first_fail["failure_log"])),
            ]
        )
        families = _extract_failed_families(text)
        writer.writerow(
            [
                first_fail["run_tag"],
                first_fail["stage"],
                first_fail["exception_type"],
                families[0] if families else "unknown",
                ",".join(families),
                "true" if first_fail["exception_type"] == "Timeout" else "false",
            ]
        )


def _write_first_failure_context(run_dir: Path, first_fail: dict[str, str] | None, failure_boundary: str | None) -> None:
    payload = {
        "failure_boundary": failure_boundary,
        "first_failure": first_fail or {},
    }
    (run_dir / "first_failure_context.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_monolithic_case(
    root_dir: Path,
    run_dir: Path,
    stage_summary: Path,
    case: RampCase,
    args: argparse.Namespace,
    timeout_sec: int,
) -> dict[str, str]:
    har_path = run_dir / f"{case.run_tag}.har"
    hef_path = run_dir / f"{case.run_tag}.hef"
    har_log = run_dir / f"{case.run_tag}.log"
    hef_log = run_dir / f"{case.run_tag}_hef.log"
    failure_log = run_dir / f"{case.run_tag}_compile_failure.txt"
    failure_json = run_dir / f"{case.run_tag}_compile_failure.txt.json"

    env = os.environ.copy()
    env["HAILO_RUN_TS"] = args.run_ts
    env["HAILO_STAGE"] = f"prefix_{case.stage}"
    env["HAILO_SUMMARY_PATH"] = str(stage_summary)

    print(f"[CASE] {case.run_tag}")
    har_cmd = [
        "./hailo/scripts/hailo_module_to_har.sh",
        "hailo_convtasnet",
        case.run_tag,
        "--n_src",
        "2",
        "--n_filters",
        str(case.n_filters),
        "--bn_chan",
        str(case.bn_chan),
        "--hid_chan",
        str(case.hid_chan),
        "--skip_chan",
        str(case.skip_chan),
        "--n_blocks",
        str(case.n_blocks),
        "--n_repeats",
        str(case.n_repeats),
        "--truncate_k_blocks",
        str(case.truncate_k_blocks),
        "--mask_mul_mode",
        case.mask_mul_mode,
        "--skip_topology_mode",
        "project",
        "--decoder_mode",
        "conv1x1_head",
        "--time_len",
        str(case.input_len),
    ]
    har_rc, har_out, har_timeout = _run_command(har_cmd, env=env, timeout_sec=timeout_sec, run_tag=case.run_tag)
    if har_out:
        with har_log.open("a", encoding="utf-8") as f:
            f.write("\n[prefix_ramp wrapper output]\n")
            f.write(har_out)

    har_success = har_rc == 0 and har_path.exists()
    hef_success = False
    etype = ""
    failure_head = ""

    if har_timeout:
        etype = "Timeout"
        failure_head = "HAR export/translate timed out"
    elif not har_success:
        etype = "HARBuildFailed"
        failure_head = "har_to_hef skipped because HAR build failed"
    else:
        hef_cmd = [
            str(root_dir / "hailo" / "to-hailo-env" / "bin" / "python"),
            "-m",
            "hailo.har_to_hef",
            str(har_path),
            str(hef_path),
            "--model_name",
            case.run_tag,
            "--hw_arch",
            args.hw_arch,
            "--calib_npz",
            args.calib_npz,
            "--input_length_policy",
            "error",
            "--log_failed_layers_path",
            str(failure_log),
        ]
        if args.quick_opt == 1:
            hef_cmd.append("--quick_opt")

        hef_rc, hef_out, hef_timeout = _run_command(hef_cmd, env=env, timeout_sec=timeout_sec, run_tag=case.run_tag)
        hef_log.write_text(hef_out, encoding="utf-8")
        hef_success = hef_rc == 0 and hef_path.exists()
        if hef_timeout:
            etype = "Timeout"
            failure_head = "HEF compile timed out"
        elif not hef_success:
            etype, failure_head = _parse_failure(hef_log, failure_json)

    return {
        "run_tag": case.run_tag,
        "stage": case.stage,
        "mask_mul_mode": case.mask_mul_mode,
        "truncate_k_blocks": str(case.truncate_k_blocks),
        "n_blocks": str(case.n_blocks),
        "n_repeats": str(case.n_repeats),
        "n_filters": str(case.n_filters),
        "bn_chan": str(case.bn_chan),
        "hid_chan": str(case.hid_chan),
        "skip_chan": str(case.skip_chan),
        "input_len": str(case.input_len),
        "har_success": _format_bool(har_success),
        "hef_success": _format_bool(hef_success),
        "exception_type": etype,
        "failure_head": failure_head,
        "har_path": str(har_path),
        "hef_path": str(hef_path),
        "har_log": str(har_log),
        "hef_log": str(hef_log),
        "failure_log": str(failure_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Progressive exact-shape monolithic ramp to isolate first HEF failure")
    parser.add_argument("--run_ts", required=True)
    parser.add_argument("--time_limit_min", type=int, default=60)
    parser.add_argument("--case_timeout_min", type=int, default=10)
    parser.add_argument("--calib_npz", default="hailo/calibration_1000ms_16k_64.npz")
    parser.add_argument("--hw_arch", choices=["hailo8", "hailo8l", "hailo8r"], default="hailo8")
    parser.add_argument("--quick_opt", type=int, choices=[0, 1], default=1)
    parser.add_argument("--n_filters", type=int, default=256)
    parser.add_argument("--bn_chan", type=int, default=128)
    parser.add_argument("--hid_chan", type=int, default=256)
    parser.add_argument("--skip_chan", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--input_len", type=int, default=16000)
    parser.add_argument("--stage_mode", choices=["exact_shape_incremental"], default="exact_shape_incremental")
    parser.add_argument("--stop_on_first_false", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    run_dir = root_dir / "hailo" / "module_runs" / args.run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = run_dir / "monolithic_prefix_ramp_summary.tsv"
    _ensure_summary(summary)
    stage_summary = run_dir / "summary.tsv"

    total_budget = max(60, args.time_limit_min * 60)
    per_case = max(60, args.case_timeout_min * 60)
    start = time.monotonic()

    stage_a: list[RampCase] = [
        RampCase(
            stage="A_masker_depth",
            run_tag=f"prefixA_bypass_active1_b{args.n_blocks}_r{args.n_repeats}",
            mask_mul_mode="bypass",
            truncate_k_blocks=1,
            n_blocks=args.n_blocks,
            n_repeats=args.n_repeats,
            n_filters=args.n_filters,
            bn_chan=args.bn_chan,
            hid_chan=args.hid_chan,
            skip_chan=args.skip_chan,
            input_len=args.input_len,
        )
    ]
    for active in range(1, args.n_blocks + 1):
        stage_a.append(
            RampCase(
                stage="A_masker_depth",
                run_tag=f"prefixA_active{active}_b{args.n_blocks}_r{args.n_repeats}",
                mask_mul_mode="normal",
                truncate_k_blocks=active,
                n_blocks=args.n_blocks,
                n_repeats=args.n_repeats,
                n_filters=args.n_filters,
                bn_chan=args.bn_chan,
                hid_chan=args.hid_chan,
                skip_chan=args.skip_chan,
                input_len=args.input_len,
            )
        )

    stage_b = [
        RampCase("B_block_count", f"prefixB_blocks{nb}_r1", "normal", nb, nb, 1, args.n_filters, args.bn_chan, args.hid_chan, args.skip_chan, args.input_len)
        for nb in [1, 2, 3, 4]
    ]
    stage_c = [
        RampCase("C_repeat_count", f"prefixC_blocks4_r{nr}", "normal", 4, 4, nr, args.n_filters, args.bn_chan, args.hid_chan, args.skip_chan, args.input_len)
        for nr in [1, 2, 3]
    ]

    rows: list[dict[str, str]] = []
    failure_boundary: str | None = None

    def run_stage(cases: list[RampCase], stop_on_fail: bool) -> bool:
        nonlocal failure_boundary
        for case in cases:
            remaining = int(total_budget - (time.monotonic() - start))
            if remaining <= 0:
                failure_boundary = failure_boundary or "Time budget exhausted"
                return False
            row = _run_monolithic_case(
                root_dir=root_dir,
                run_dir=run_dir,
                stage_summary=stage_summary,
                case=case,
                args=args,
                timeout_sec=min(per_case, remaining),
            )
            rows.append(row)
            _append_summary(
                summary,
                [
                    row["run_tag"],
                    row["stage"],
                    row["mask_mul_mode"],
                    row["truncate_k_blocks"],
                    row["n_blocks"],
                    row["n_repeats"],
                    row["n_filters"],
                    row["bn_chan"],
                    row["hid_chan"],
                    row["skip_chan"],
                    row["input_len"],
                    row["har_success"],
                    row["hef_success"],
                    row["exception_type"],
                    row["failure_head"],
                    row["har_path"],
                    row["hef_path"],
                    row["har_log"],
                    row["hef_log"],
                    row["failure_log"],
                ],
            )
            if row["hef_success"] != "true":
                if failure_boundary is None:
                    failure_boundary = f"{row['stage']}:{row['run_tag']}"
                print(f"[HEF FAIL] {row['run_tag']}: {row['exception_type']} | {row['failure_head']}")
                if stop_on_fail:
                    return False
            else:
                print(f"[HEF PASS] {row['run_tag']}")
        return True

    stop_on_false = bool(args.stop_on_first_false)
    stage_a_ok = run_stage(stage_a, stop_on_fail=stop_on_false)
    if stage_a_ok:
        stage_b_ok = run_stage(stage_b, stop_on_fail=True)
        if stage_b_ok:
            run_stage(stage_c, stop_on_fail=True)

    total_rows = len(rows)
    har_pass = sum(1 for r in rows if r["har_success"] == "true")
    hef_pass = sum(1 for r in rows if r["hef_success"] == "true")
    first_fail = next((r for r in rows if r["hef_success"] != "true"), None)

    print(f"HAR status: {har_pass}/{total_rows}")
    print(f"HEF status: {hef_pass}/{total_rows} ; first_failure={first_fail['run_tag'] if first_fail else 'none'}")
    print(f"Failure boundary: {failure_boundary or 'frontier not reached'}")
    print(f"Summary: {summary}")

    _write_failure_signature(run_dir, first_fail)
    _write_first_failure_context(run_dir, first_fail, failure_boundary)

    if first_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
