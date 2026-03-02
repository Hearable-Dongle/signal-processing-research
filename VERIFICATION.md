# Verification Framework

This document defines how we verify each subsystem and the integrated pipeline, with **Speech Intelligibility Index (SII)** as the primary optimization target.

## Verification Goal

Primary objective:
- maximize intelligibility of target speech in processed output, measured by **SII**.

Secondary diagnostics:
- SI-SDR, SNR, RMSE
- angular localization/direction errors
- identity consistency and ID accuracy
- runtime / realtime factor

Current sign-off scope:
- **simulation only** (hardware checks are out of scope for this gate)

## Subsystems

- Localization
- Speaker Identification (enrolled-ID accuracy)
- Speaker Grouping (unsupervised ID stability)
- Beamforming
- Integrated system (all subsystems together)

## Verification Matrix

## 1) Localization

Implementation source:
- `localization/benchmark/*`

Metrics:
- `mae_deg_matched_mean` (target <= 12 deg)
- `recall_mean` (target >= 0.80)
- `precision_mean` (target >= 0.80)

Sanity checks/artifacts:
- `summary_by_method.csv`
- `README_summary.md`
- `overall_method_comparison.png`
- `scene_type_mae_comparison.png`
- `k_trends.png`

Verification command:
```bash
python -m localization.benchmark.run --preset quick --max-scenes 20
```

---

## 2) Speaker Identification (Enrolled-ID)

Definition:
- Identification is measured as mapping tracks to known enrolled IDs.
- This is distinct from grouping stability.

Implementation source:
- `direction_assignment.validate` output (`track_assignments.csv`)
- verifier: `verification/speaker_id_verify.py`

Metrics:
- `id_accuracy` (target >= 0.90)
- `macro_f1` (target >= 0.85)
- `id_switch_rate` (target <= 0.10)
- confidence separation: `mean_conf_correct` vs `mean_conf_wrong`

Sanity checks/artifacts:
- confusion matrix plot (`confusion_matrix.png`)
- per-scene/per-chunk CSVs from direction-assignment validation
- per-scene track assignment CSVs

Verification command:
```bash
python -m direction_assignment.validate --num-scenes 12 --identity-mode enroll_audio
```

---

## 3) Speaker Grouping

Implementation source:
- `speaker_identity_grouping.validate`
- `speaker_identity_grouping.reconstruct_examples`
- `speaker_identity_grouping.listening_report`

Metrics:
- `majority_vote_accuracy` (target >= 0.88)
- `switch_rate` (target <= 0.15)
- `speaker_count_ratio` (target in [0.8, 1.25])
- `realtime_factor_total` (target <= 1.0)

Sanity checks/artifacts:
- stitched speaker audio tracks
- `chunk_mappings.csv`
- instability-ranked listening snippets

Verification command:
```bash
python -m speaker_identity_grouping.validate --max-mixtures 25 --chunk-ms 200 --device cpu
```

---

## 4) Beamforming

Implementation source:
- `beamforming.main`
- `beamforming.benchmark.run`
- SII augmentation via `verification/sii_utils.py`

Metrics:
- primary: `delta_sii_median` (target > 0.03)
- secondary: `delta_si_sdr_db_mean` (target > 1.0 dB)
- secondary: `delta_snr_db_mean` (target > 1.0 dB)

Sanity checks/artifacts:
- `steering_comparison.csv`
- output audio bundles (`audio/*.wav`, `*_norm_to_ref.wav`)
- beam pattern and convergence plots

Verification command (single scene example):
```bash
python -m beamforming.main \
  --simulation-scene-file simulation/simulations/configs/library_scene/library_k2_scene00.json \
  --steering-source both \
  --steering-time both
```

---

## 5) Integrated End-to-End

Implementation source:
- `realtime_pipeline.simulation_runner`
- verifier: `verification/integrated_verify.py`

Metrics:
- primary: `delta_sii_full` (target > 0.02)
- liveness checks: `fast_frames > 0`, `slow_chunks > 0`, `speaker_map_updates > 0`

Sanity checks/artifacts:
- integrated output audio (`enhanced_fast_path.wav`)
- integrated `summary.json`
- ablation score JSON placeholder (`ablation_scores.json`)

Verification command:
```bash
python -m realtime_pipeline.simulation_runner \
  --scene-config simulation/simulations/configs/library_scene/library_k2_scene00.json \
  --out-dir realtime_pipeline/output/sim_run
```

## Unified Verification Runner

Use the verification package to run everything and aggregate results:

```bash
python -m verification.run_all --quick \
  --scene-config simulation/simulations/configs/library_scene/library_k2_scene00.json \
  --out-root verification/output
```

Outputs:
- `verification/output/<timestamp>/summary.json`
- `verification/output/<timestamp>/subsystem_scores.csv`
- `verification/output/<timestamp>/artifacts_manifest.json`
- `verification/output/<timestamp>/README_summary.md`

## Gate Policy

A subsystem is:
- `pass` if all thresholded metrics pass
- `warn` if run succeeds but one or more thresholded metrics fail
- `error` if run fails or required outputs are missing

Overall verification:
- `overall_pass = true` only when all subsystems are `pass`

## Sanity Checklist (Human Review)

For each run, review:
1. Localization plots: check obvious DOA drift/failure patterns.
2. Speaker ID confusion matrix: check dominant diagonal.
3. Grouping stitched audio: verify continuity per speaker track.
4. Beamforming audio and beam patterns: verify suppression/focus behavior.
5. Integrated audio (`enhanced_fast_path.wav`): confirm intelligibility gain vs raw mix.

## Known Limitations

- Integrated ablation ladder is currently minimal and uses mock realtime separation path in default verifier.
- Real separation backend validation depends on environment availability (`multispeaker_separation` or Asteroid model install).
- SII computation in verification currently assumes a synthetic-reference alignment strategy; for hardware this must be replaced with controlled reference captures.

## Next Expansion Tasks

1. Expand integrated ablation ladder to include oracle-localization and oracle-ID branches explicitly.
2. Add STOI as optional secondary intelligibility metric.
3. Add CI target for quick verification runner with deterministic simulation seed.
4. Add per-subsystem trend dashboards over time (regression tracking).
