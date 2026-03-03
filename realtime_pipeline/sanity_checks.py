from __future__ import annotations

import argparse
import compileall
import json
from pathlib import Path

from .contracts import PipelineConfig
from .separation_backends import probe_backend_support


def run_sanity_checks(
    *,
    out_dir: str | Path,
    scene_config_path: str | Path,
) -> dict:
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    compile_ok = bool(compileall.compile_dir(str(Path(__file__).resolve().parent), quiet=1))

    import_report: dict[str, dict[str, object]] = {}
    for mod in ["realtime_pipeline", "speaker_identity_grouping", "direction_assignment", "localization", "simulation"]:
        try:
            __import__(mod)
            import_report[mod] = {"ok": True}
        except Exception as exc:
            import_report[mod] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    backend_report = probe_backend_support(PipelineConfig())

    e2e_report: dict[str, object]
    try:
        from .simulation_runner import run_simulation_pipeline

        smoke_dir = out_root / "mock_smoke"
        summary = run_simulation_pipeline(
            scene_config_path=scene_config_path,
            out_dir=smoke_dir,
            use_mock_separation=True,
        )
        e2e_report = {
            "ok": True,
            "summary": summary,
        }
    except Exception as exc:
        e2e_report = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    overall_ok = bool(
        compile_ok
        and all(bool(v.get("ok")) for v in import_report.values())
        and bool(e2e_report.get("ok"))
    )

    report = {
        "overall_ok": overall_ok,
        "compile_ok": compile_ok,
        "imports": import_report,
        "backend_probe": backend_report,
        "mock_e2e": e2e_report,
    }

    with (out_root / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run realtime_pipeline sanity checks")
    p.add_argument("--scene-config", required=True)
    p.add_argument("--out-dir", default="realtime_pipeline/output/validation")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = run_sanity_checks(out_dir=args.out_dir, scene_config_path=args.scene_config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
