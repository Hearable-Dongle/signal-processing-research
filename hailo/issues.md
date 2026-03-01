# Issues and Suggestions

## Current blocker (active)
1. `NegativeSlopeExponentNonFixable` during `runner.optimize`
- Symptom: fails on `convtas/ne_activation_avgpool6` with `Desired shift is 10.0, but op has only 8 data bits`.
- Evidence: `hailo/compile_failure_longrun_20260216_030111.txt`, `hailo/compile_failure_longrun_20260216_030111.txt.json`.
- Likely root cause: normalization-heavy subgraph creates activation ranges/slopes that are hard to quantize with current calibration statistics.

2. Debug-loop override layer name instability
- Symptom: after first failure, appended override layer (`convtas/ne_activation_avgpool6`) is not resolvable in subsequent iterations (`Layer ... not found in model`).
- Impact: loop can spend full optimization cycles without actionable progress.

## Historical blockers (mostly resolved)
1. Large reshape-insertion failures in compiler logs
- Symptom: long `Reshape is needed for layers ...` lists in SDK logs.
- Interpretation: graph/layout compatibility issues from earlier export variants.
- Status: not the latest direct blocker in current artifacts.

2. Earlier unsupported-dimension path (`262144`)
- Interpretation: prior export/layout issue and/or unsupported op lowering behavior.
- Status: stale compared with current `NegativeSlopeExponentNonFixable` failure.

## Contributing factors
1. Calibration length mismatch against model input length
- Current datasets include 200 ms clips (`W=3200`) while model path commonly expects `W=16000`.
- Padding/repeat/crop can materially change calibration distribution and quantization outcomes.

2. Layer-scoped overrides are not always stable across optimization passes
- Parsed failing layer names may not belong to current HN scope used by model script validation.

## Suggestions
1. Keep export in 4D horizontal form and simplify activation/norm ops earlier
- Replace `PReLU -> ReLU` at export time.
- Replace `GlobLN -> channel-wise LN` at export time.
- Keep capacity unchanged (same blocks/repeats/channels), only swap problematic ops.

2. Validate override targets against HN scope before append
- Append `negative_exponent` override only if parsed layer exists in `runner.get_hn_dict()["layers"]`.
- Log unresolved layer names explicitly and avoid writing unusable overrides.

3. Improve calibration observability
- Log explicit warning when `W` mismatch is handled by `pad|repeat|crop`.
- Prefer calibration clips matching target input length when possible (e.g., 1000 ms for 16 kHz input length 16000).

## Expected success signals
1. First optimization failure signature changes away from `NegativeSlopeExponentNonFixable:convtas/ne_activation_avgpool6` after export substitutions.
2. `compile_failure.txt(.json)` captures clear per-iteration signatures when debug loop is enabled.
3. Final target: HEF generated without manual trial-and-error looping.
