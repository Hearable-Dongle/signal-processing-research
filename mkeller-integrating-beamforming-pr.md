## Summary
Prototype integration PR that combines:
- Realtime dual-path pipeline orchestration (fast path + slow path)
- Verification framework and subsystem/integrated metrics docs
- Localization-aware beamforming and benchmark expansion

This is intentionally broad for research velocity and simulation-first iteration.

## What Changed
- Added realtime pipeline docs and orchestration support around existing modules.
- Added verification framework and `VERIFICATION.md` to define metrics + sanity checks.
- Extended beamforming flow with localization bridge + fallback DOA estimation.
- Added steering modes (oracle/localized/both, fixed/dynamic), target-weight handling, and richer evaluation outputs.
- Added beamforming benchmark runner/config for scene sweeps and aggregate summaries.
- Added shared target policy in simulation config handling (`classification` + target selection helpers).
- Updated top-level and beamforming READMEs.

## Validation Run
- `PYTHONPATH=. beamforming/beamforming-sim-env/bin/python -m pytest -q tests/test_realtime_pipeline_shared_state.py`
- `PYTHONPATH=. beamforming/beamforming-sim-env/bin/python -m pytest -q tests/test_realtime_pipeline_backend_resolver.py`
- `PYTHONPATH=. beamforming/beamforming-sim-env/bin/python -m pytest -q tests/test_realtime_pipeline_integration_mock.py`

## Notes
- PR includes research artifacts/models/logs/audio outputs and is not minimized.
- Follow-up cleanup PR can split code vs artifacts and trim large binaries if needed.

## Next TODOs
- Add deterministic simulation regression suite with pinned scenes + thresholds.
- Add per-subsystem metric exporters (localization/ID/grouping/beamforming) into one report.
- Add SII/STOI-focused objective tracking in integrated benchmark outputs.
