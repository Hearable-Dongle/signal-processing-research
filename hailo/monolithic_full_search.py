import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CaseConfig:
    n_filters: int
    bn_chan: int
    hid_chan: int
    skip_chan: int
    n_blocks: int
    n_repeats: int
    input_len: int

    @property
    def run_tag(self) -> str:
        return (
            f"mono_f{self.n_filters}_bn{self.bn_chan}_hid{self.hid_chan}"
            f"_skip{self.skip_chan}_b{self.n_blocks}_r{self.n_repeats}_t{self.input_len}"
        )


LADDER_CASES = [
    CaseConfig(256, 128, 256, 128, 2, 1, 16000),
    CaseConfig(256, 96, 192, 96, 2, 1, 16000),
    CaseConfig(256, 64, 128, 64, 2, 1, 16000),
    CaseConfig(192, 64, 128, 64, 2, 1, 16000),
    CaseConfig(128, 64, 128, 64, 2, 1, 16000),
    CaseConfig(128, 48, 96, 48, 2, 1, 16000),
    CaseConfig(128, 32, 64, 32, 2, 1, 16000),
    CaseConfig(96, 32, 64, 32, 2, 1, 16000),
    CaseConfig(96, 24, 48, 24, 1, 1, 16000),
    CaseConfig(64, 24, 48, 24, 1, 1, 16000),
]


def _ensure_summary(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "run_tag",
                "n_filters",
                "bn_chan",
                "hid_chan",
                "skip_chan",
                "n_blocks",
                "n_repeats",
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
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)


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


def _run_command(cmd: list[str], env: dict[str, str], timeout_sec: int) -> tuple[int, str, bool]:
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
        os.killpg(proc.pid, 15)
        try:
            stdout, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
            stdout, _ = proc.communicate()
        return 124, stdout or "", True


def _allocator_failure(row: dict[str, str]) -> bool:
    etype = row.get("exception_type", "")
    fhead = row.get("failure_head", "")
    return (
        "BackendAllocatorException" in etype
        or "Agent infeasible" in fhead
        or "Mapping Failed" in fhead
    )


def _choose_next_knob(rows: list[dict[str, str]]) -> str:
    failed = [r for r in rows if r["hef_success"] != "true"]
    if not failed:
        return "none (winner found)"

    if all(_allocator_failure(r) for r in failed):
        # Single explicit next change from the smallest case in this ladder.
        last = failed[-1]
        hid = int(last["hid_chan"])
        if hid > 16:
            return f"reduce hid_chan: {hid} -> {max(16, hid - 16)}"
        bn = int(last["bn_chan"])
        if bn > 8:
            new_bn = max(8, bn - 8)
            return f"reduce bn_chan/skip_chan together: {bn} -> {new_bn}"
        nf = int(last["n_filters"])
        if nf > 32:
            return f"reduce n_filters: {nf} -> {max(32, nf - 32)}"
        nb = int(last["n_blocks"])
        if nb > 1:
            return f"reduce n_blocks: {nb} -> {nb - 1}"
        return "reduce input_len: 16000 -> 8000"

    # If mixed failures, recommend first explicit unresolved error family.
    top = failed[0]
    return f"resolve non-allocator failure first: {top['exception_type']}"


def _format_bool(v: bool) -> str:
    return "true" if v else "false"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run monolithic ConvTasNet ONNX->HAR->HEF search ladder")
    parser.add_argument("--run_ts", required=True)
    parser.add_argument("--profile", choices=["focused", "extended"], default="focused")
    parser.add_argument("--time_limit_min", type=int, default=120)
    parser.add_argument("--case_timeout_min", type=int, default=12)
    parser.add_argument("--calib_npz", default="hailo/calibration_1000ms_16k_64.npz")
    parser.add_argument("--hw_arch", choices=["hailo8", "hailo8l", "hailo8r"], default="hailo8")
    parser.add_argument("--quick_opt", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    run_dir = root_dir / "hailo" / "module_runs" / args.run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "monolithic_full_hef_summary.tsv"
    _ensure_summary(summary_path)

    # Keep using existing stage summary for HAR internals.
    stage_summary = run_dir / "summary.tsv"

    started = time.monotonic()
    time_limit_sec = max(1, args.time_limit_min * 60)
    case_timeout_sec = max(60, args.case_timeout_min * 60)

    rows: list[dict[str, str]] = []
    winner: Optional[str] = None

    for case in LADDER_CASES:
        elapsed = time.monotonic() - started
        remaining = int(time_limit_sec - elapsed)
        if remaining <= 0:
            break

        per_case_timeout = min(case_timeout_sec, remaining)
        run_tag = case.run_tag

        har_path = run_dir / f"{run_tag}.har"
        hef_path = run_dir / f"{run_tag}.hef"
        har_log = run_dir / f"{run_tag}.log"
        hef_log = run_dir / f"{run_tag}_hef.log"
        failure_log = run_dir / f"{run_tag}_compile_failure.txt"
        failure_json = run_dir / f"{run_tag}_compile_failure.txt.json"

        env = os.environ.copy()
        env["HAILO_RUN_TS"] = args.run_ts
        env["HAILO_STAGE"] = "stage_monolithic_full"
        env["HAILO_SUMMARY_PATH"] = str(stage_summary)

        print(f"[CASE] {run_tag}")

        har_cmd = [
            "./hailo/scripts/hailo_module_to_har.sh",
            "hailo_convtasnet",
            run_tag,
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
            "0",
            "--mask_mul_mode",
            "normal",
            "--skip_topology_mode",
            "project",
            "--decoder_mode",
            "conv1x1_head",
            "--time_len",
            str(case.input_len),
        ]

        har_rc, har_out, har_timeout = _run_command(har_cmd, env=env, timeout_sec=per_case_timeout)
        if har_out:
            with har_log.open("a", encoding="utf-8") as f:
                f.write("\n[monolithic_full_search wrapper output]\n")
                f.write(har_out)

        har_success = har_rc == 0 and har_path.exists()
        hef_success = False
        exception_type = ""
        failure_head = ""

        if har_timeout:
            exception_type = "Timeout"
            failure_head = "HAR export/translate timed out"
        elif not har_success:
            exception_type = "HARBuildFailed"
            failure_head = "har_to_hef skipped because HAR build failed"
        else:
            hef_cmd = [
                str(root_dir / "hailo" / "to-hailo-env" / "bin" / "python"),
                "-m",
                "hailo.har_to_hef",
                str(har_path),
                str(hef_path),
                "--model_name",
                run_tag,
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

            remaining = int(time_limit_sec - (time.monotonic() - started))
            if remaining <= 0:
                exception_type = "Timeout"
                failure_head = "time budget exhausted before HEF compile"
            else:
                hef_rc, hef_out, hef_timeout = _run_command(
                    hef_cmd,
                    env=env,
                    timeout_sec=min(per_case_timeout, remaining),
                )
                hef_log.write_text(hef_out, encoding="utf-8")
                hef_success = hef_rc == 0 and hef_path.exists()

                if hef_timeout:
                    exception_type = "Timeout"
                    failure_head = "HEF compile timed out"
                elif not hef_success:
                    exception_type, failure_head = _parse_failure(hef_log, failure_json)

        row = {
            "run_tag": run_tag,
            "n_filters": str(case.n_filters),
            "bn_chan": str(case.bn_chan),
            "hid_chan": str(case.hid_chan),
            "skip_chan": str(case.skip_chan),
            "n_blocks": str(case.n_blocks),
            "n_repeats": str(case.n_repeats),
            "input_len": str(case.input_len),
            "har_success": _format_bool(har_success),
            "hef_success": _format_bool(hef_success),
            "exception_type": exception_type,
            "failure_head": failure_head,
            "har_path": str(har_path),
            "hef_path": str(hef_path),
            "har_log": str(har_log),
            "hef_log": str(hef_log),
            "failure_log": str(failure_log),
        }
        rows.append(row)

        _append_summary(
            summary_path,
            [
                row["run_tag"],
                row["n_filters"],
                row["bn_chan"],
                row["hid_chan"],
                row["skip_chan"],
                row["n_blocks"],
                row["n_repeats"],
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

        if hef_success:
            winner = run_tag
            print(f"[HEF PASS] {run_tag}")
            if args.profile == "focused":
                break
        else:
            print(f"[HEF FAIL] {run_tag}: {exception_type} | {failure_head}")

    total = len(rows)
    har_pass = sum(1 for r in rows if r["har_success"] == "true")
    hef_pass = sum(1 for r in rows if r["hef_success"] == "true")
    runtime_status = "fail"
    runtime_output_dir = "none"

    next_knob = _choose_next_knob(rows)
    next_knob_path = run_dir / "monolithic_full_next_knob.json"
    next_knob_path.write_text(
        json.dumps(
            {
                "run_ts": args.run_ts,
                "winner": winner,
                "next_knob": next_knob,
                "profile": args.profile,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"HAR status: {har_pass}/{total}")
    print(f"HEF status: {hef_pass}/{total} ; winner={winner or 'none'}")
    print(f"Runtime status: {runtime_status} ; output_dir={runtime_output_dir} (skipped by request)")
    print(f"Next knob: {next_knob}")

    if hef_pass == 0:
        print("no monolithic hef success", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
