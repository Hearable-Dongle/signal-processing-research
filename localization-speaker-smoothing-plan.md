# Write `localization-speaker-smoothing-plan.md`, Then Run Before/After Comparisons and Update `temp-pr.txt`

## Summary
Create this plan file first, then execute the comparison suite and update `temp-pr.txt` from real outputs. The deliverable is a concrete before/after report for `localization`, `speaker grouping`, and `direction assignment`, plus showcase artifact bundles and a PR-ready summary grounded in measured results.

## Implementation Changes
### 1. Make the comparison runner artifact-complete
- Extend `realtime_pipeline/robustness_validate.py` so one run produces:
  - per-stage `baseline` vs `robust` summary CSVs by speaker-count bucket
  - per-scene delta CSVs
  - deterministic showcase selection for one `1`, one `2`, and one `3+` candidate per subsystem
  - a summary JSON containing metric aggregates and selected showcase paths
- Keep comparison semantics fixed:
  - `baseline`: priors/hold logic disabled through existing toggles/config construction
  - `robust`: current prior-aware defaults
- Add explicit output structure under a timestamped run root with stage folders for `localization`, `grouping`, `direction_assignment`, and `pipeline`.

### 2. Produce stage-specific before/after artifacts
- Localization:
  - keep visual-first artifacts only
  - generate per-frame metrics, bucket summaries, per-scene delta rows, and showcase plots for one scene per bucket
  - include DOA timeline, jitter/continuity comparison, and tracked-vs-raw comparison where available
- Speaker grouping:
  - generate switch-rate, stability, confidence, and count-ratio summaries by bucket
  - produce showcase identity trace plots for one scene per bucket
  - if existing reconstruction/listening helpers can be reused cheaply, apply them only to showcase candidates; otherwise keep grouping showcase bundles visual-only
- Direction assignment:
  - generate bucketed before/after summaries and per-scene deltas
  - export showcase plots plus audio bundles for one scene per bucket
  - keep `baseline` and `robust` artifact folders separate and parallel
- Pipeline / beamformed-output comparison:
  - use the realtime simulation path to generate raw mix and enhanced/beamformed outputs for showcase scenes
  - produce one showcase candidate per bucket with audio files, trace CSVs, and summary plots
  - treat this as the default “raw and beamformed outputs” path where audio comparison is most meaningful

### 3. Deterministic showcase selection
- For each subsystem and each bucket `1`, `2`, `3+`, select exactly one showcase candidate.
- Selection rule:
  - only consider scenes with complete expected artifacts
  - rank by closeness to the bucket’s median robust-minus-baseline improvement for the primary metric
  - break ties deterministically by scene id or synthetic scene name
- Save the selected candidates in the run summary JSON so downstream docs and `temp-pr.txt` can reference exact folders.

### 4. Run the balanced comparison suite
- Use an overnight-safe balanced run, not the smallest smoke run.
- Default run shape:
  - repo simulation scenes for localization and pipeline-backed comparisons
  - synthetic scenes for grouping and direction assignment
  - enough samples per bucket to stabilize summary metrics while still keeping runtime practical on one workstation
- After completion, verify all stage/bucket pairs have both `baseline` and `robust` rows and all showcase folders exist.

### 5. Update `temp-pr.txt` from measured results
- Rewrite `temp-pr.txt` so it begins with a concise PR message:
  - title line
  - short rationale
  - compact list of major changes
  - short validation summary
- Follow that with a deeper measured-results section:
  - before/after metrics for localization by `1`, `2`, `3+`
  - before/after metrics for speaker grouping by `1`, `2`, `3+`
  - before/after metrics for direction assignment by `1`, `2`, `3+`
  - brief notes on the selected showcase candidates and where their artifacts live
  - concise caveats for any stage where audio artifacts are intentionally omitted
- Use only concrete values from the completed run; no placeholders.

## Public Interfaces / Outputs
- `localization-speaker-smoothing-plan.md` is added at repo root as the written execution plan.
- `realtime_pipeline.robustness_validate` may gain small CLI additions only if needed for:
  - showcase count per bucket
  - output root control
  - optional audio export toggles for applicable stages
- Output contract for the comparison run:
  - timestamped root
  - per-stage summary CSVs
  - per-scene delta CSVs
  - showcase subfolders
  - summary JSON with showcase paths and bucket summaries

## Test and Acceptance Criteria
- Sanity:
  - comparison runner `--help` works after changes
  - one smoke run validates folder layout before launching the balanced run
- Full acceptance:
  - `baseline` and `robust` outputs exist for all requested stages and for buckets `1`, `2`, `3+`
  - one showcase candidate exists per bucket per subsystem
  - direction assignment and pipeline showcase bundles include audio outputs
  - `temp-pr.txt` contains real before/after metrics and real artifact references
  - no placeholder numbers remain in docs or summaries

## Assumptions and Defaults
- Audio artifacts are required by default only for `direction_assignment` and pipeline/end-to-end comparisons.
- Showcase depth is fixed at one candidate per bucket per subsystem.
- Runtime target is balanced overnight-safe, not fastest possible.
