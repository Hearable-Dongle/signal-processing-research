# Next Agent Instructions

You are analyzing completed overnight outputs for the prior-aware robustness pass across localization, speaker grouping, direction assignment, and the realtime pipeline.

Your job:
- inspect the generated outputs
- identify the main wins, regressions, and likely root causes
- decide what should be tuned next, especially in speaker grouping
- produce a concrete next-step plan for follow-up changes

Context
- The overnight runs are complete.
- The raw overnight output root is:
  - `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night`
- The main cross-stage comparison bundle is:
  - `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/robustness_validation`

Most important files to inspect first

1. Cross-stage before/after compare
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/robustness_validation/summary.json`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/robustness_validation/summary_by_bucket.csv`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/robustness_validation/delta_summary_by_bucket.csv`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/robustness_validation/showcases.json`

2. Speaker grouping specific outputs
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/speaker_identity_grouping/summary.json`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/speaker_identity_grouping/per_mixture_summary.csv`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/speaker_identity_grouping/sample_pair_rows.csv`

3. Direction assignment specific outputs
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/direction_assignment/summary.json`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/direction_assignment/per_scene_metrics.csv`

4. Localization benchmark
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/localization_benchmark/20260307_215110/README_summary.md`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/localization_benchmark/20260307_215110/summary_by_method.csv`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/localization_benchmark/20260307_215110/summary_by_k.csv`

5. End-to-end focus sanity
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/focus_sanity/summary.json`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/focus_sanity/per_scene_metrics.json`

Showcase folders for manual inspection
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/robustness_validation/showcases/localization`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/robustness_validation/showcases/grouping`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/robustness_validation/showcases/direction_assignment`
- `/home/mkeller/mkeller/signal-processing-research/overnight_runs/20260307_sat_night/results/robustness_validation/showcases/pipeline`

What to answer

Speaker grouping
- Why did `switch_rate` and `stability` regress in the `2` and `3+` buckets?
- Is the main problem the continuity bonus, the switch penalty, or the weak-evidence carry-forward logic?
- Which scenes in the grouping showcases best demonstrate the failure?
- What exact parameter or logic changes should be tried next?

Localization
- Robust mode reduced jitter and improved continuity in dense scenes, but why did MAE get worse in `3+`?
- Is the hold behavior too sticky, or is the SRP peak match tolerance too wide?
- Which localization showcase most clearly shows the tradeoff?

Direction assignment
- Why is `2-speaker` much better while `1-speaker` got worse?
- Is the transition penalty too conservative for low-speaker scenes?
- Which direction-assignment showcase best supports the next tuning pass?

Pipeline
- Map churn improved, but MAE is mixed. Is that acceptable or is the state hold too aggressive?
- Does the focus sanity output suggest the same problem as the robustness pipeline bundle?

Expected output from you
- A short findings-first review, ordered by severity
- A concrete follow-up plan with exact parameters or code areas to change next
- Identification of the best showcase scene per subsystem for human review
- A recommendation on whether robust defaults should stay enabled as-is, be partially rolled back, or be retuned before merge

Constraints
- Do not re-run large overnight jobs unless a missing detail blocks the analysis.
- Prefer reading the completed outputs first.
- If you need one small validation rerun, keep it narrowly scoped and explain why.
