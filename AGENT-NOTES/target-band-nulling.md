# Target-Band + Null Follow-up

Current default:
- `lcmv_target_band`
- preserve `theta - width`, `theta`, `theta + width`
- no null constraint by default

## How to add nulling
Extend the target-band constraint matrix with one additional null steering vector:
- preserve:
  - `theta - width`
  - `theta`
  - `theta + width`
- null:
  - `theta_null`

Desired response becomes:
- `[1, 1, 1, 0]`

## Where to apply it
- Reuse the existing LCMV constraint solver in `realtime_pipeline/fast_path.py`.
- Keep the same center target DOA for:
  - target-activity estimation
  - `Rnn` target subtraction

## How to test it
- Start with GT target DOA and a known null DOA.
- Compare:
  - `lcmv_target_band`
  - `lcmv_target_band + null`
- Listen for:
  - target attenuation
  - null effectiveness
  - overconstraint artifacts

## Failure signs
- output collapses toward delay-sum
- target voice gets thinner than plain target-band mode
- nulling only works when the localization is perfect

If that happens:
- narrow the target band
- raise diagonal loading slightly
- only enable nulling when the null DOA is stable enough
