# LCMV Target Band Debug Checklist

Use this when `lcmv_target_band` is not improving over `mvdr_fd`, sounds unstable, or collapses toward delay-sum/raw.

## Angle-space checks
- Confirm the selected target DOA, focus DOA, and localization peaks are all in the same angle space.
- For GT-override benchmarks, verify the initial focus direction is converted into backend angle space before entering the fast path.
- Check `srp_trace` for:
  - `peaks_deg`
  - `target_selection.selected_target_doa_deg`
  - `target_activity.debug`

## Constraint checks
- Confirm the 3 target constraints are built at:
  - `theta - width`
  - `theta`
  - `theta + width`
- Confirm the desired response is `[1, 1, 1]`.
- Confirm the center DOA is still the one used for target-activity and `Rnn` estimation in v1.

## Width checks
- If the target is still attenuated, the band may be too narrow.
- If suppression is weak and output sounds too close to delay-sum, the band may be too wide.
- First comparison points:
  - `0 deg` width should behave like a single-target constraint
  - `10 deg` width is the current default

## Covariance checks
- Compare `lcmv_target_band` vs `mvdr_fd` under the same:
  - target activity mode
  - covariance mode
  - diagonal load
- If adaptive and frozen modes are bit-identical, inspect whether `Rnn` is actually changing.
- If GT DOA + target-band still equals delay-sum, inspect the solved weights numerically against the delay-sum steering weights.

## Refresh / runtime checks
- Verify the new mode uses the covariance-beamforming path and not a fallback.
- Inspect:
  - `weights_reused`
  - `noise_model_update_trace`
  - `beamforming_mode` in packet/debug fields
- If periodic artifacts appear, compare:
  - `40/80`
  - `60/120`
  and listen for weight-refresh modulation.

## Localization checks
- For estimated single-active mode, inspect:
  - `dominant_direction_step_p95_deg`
  - GT trace MAE / acc@20
- If the DOA is wrong, target-band will preserve the wrong region more robustly, which is still a failure.

## Listening comparisons
- Always compare these side by side on the same recording:
  - `raw_mix_mean.wav`
  - `delay_sum`
  - `mvdr_fd`
  - `lcmv_target_band`
- If `lcmv_target_band` protects the voice better than `mvdr_fd` while still changing the output meaningfully versus delay-sum, the mode is doing the intended job.
