# Look Into

This file is a parking lot for beamforming follow-ups that are useful but not currently prioritized.  
Use it to capture next-stage ideas, quick rationale, and what to validate when revisiting.

## Next Stage (Deferred)

1. GSC stabilization hardening
- Add adaptive-step and leakage controls in `gsc_fd`.
- Add per-bin stability guards (condition checks + fallback to last stable weights).
- Gate updates more aggressively when direction/activity confidence is low.
- Goal: reduce warble/static bursts and make `gsc_fd` competitive with `mvdr_fd`.

2. Add runtime profiles
- Add profile flag (example: `balanced`, `low_latency`) that maps to grouped defaults.
- `balanced`: keep current quality-oriented smoothing/postfilter.
- `low_latency`: reduce smoothing/postfilter aggressiveness for tighter delay.
- Goal: explicit quality/latency tradeoff without manual retuning each run.

3. Comparison/reporting improvements
- Include `profile` label in method comparison CSV/report for clean A/B tracking.
- Add simple pass/fail gates (artifact + intelligibility) in report output.
- Goal: quickly detect regressions and make method/profile decisions obvious.

4. Validation checklist when resumed
- Run `mvdr_fd`, `gsc_fd`, `delay_sum` on same scene set.
- Compare:
  - `delta_sii_mean`, `delta_stoi_mean`, `delta_si_sdr_db_mean`
  - `high_band_noise_ratio_mean`, `frame_gain_delta_p95_mean`, `spectral_flux_non_speech_mean`
  - `clip_rate_ge_0p99`
- Listen to worst-scoring scenes before finalizing defaults.

