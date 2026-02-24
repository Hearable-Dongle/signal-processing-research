# Direction Assignment System Guide

## Goal
Implement a realtime direction-assignment module that binds speaker IDs (from `speaker_identity_grouping`) to DOAs and emits beamformer-ready steering arrays each chunk.

This is not a standalone utility. It is an interop module between:
- speaker separation + identity grouping
- localization SRP peak detection
- beamforming steering/weight update

## Where Code Lives
All code for this subsystem lives in `direction_assignment/`.

Implemented modules:
- `direction_assignment/config.py`
- `direction_assignment/types.py`
- `direction_assignment/geometry.py`
- `direction_assignment/mask_backprojection.py`
- `direction_assignment/doa_estimation.py`
- `direction_assignment/tracking.py`
- `direction_assignment/weight_policy.py`
- `direction_assignment/engine.py`
- `direction_assignment/payload_adapter.py`

## End-to-End Flow
1. Receive chunk payload:
- raw mic chunk `(samples, n_mics)`
- separated mono streams
- `stream_to_speaker` mapping
- SRP-PHAT peaks (and optional scores)

2. Backproject separated streams to multichannel:
- compute STFT of raw multichannel chunk
- compute STFT of each separated stream
- build ratio masks per stream
- apply masks to each mic STFT
- ISTFT to per-stream multichannel time signals

3. Per-stream DOA estimation:
- run GCC-PHAT on selected mic pairs
- estimate TDOA per pair with physical lag clamp
- solve azimuth by minimizing pairwise TDOA residual over angle grid
- compute confidence from coherence + fit residual

4. Speaker-level fusion + tracking:
- merge multiple streams mapping to same `speaker_id`
- optional snap to nearest SRP peak within tolerance
- circular EMA smoothing for stable trajectories
- stale/forget lifecycle for inactive speakers

5. Beamformer packet emission:
- output `{speaker_id: doa}` + confidence
- output ordered `target_speaker_ids`, `target_doas_deg`, `target_weights`
- this ordering is what beamforming should consume directly

## Data Contracts

### Input (`DirectionAssignmentInput`)
- `chunk_id: int`
- `timestamp_ms: float`
- `raw_mic_chunk: np.ndarray` shape `(samples, n_mics)`
- `separated_streams: list[np.ndarray]` (mono streams)
- `stream_to_speaker: dict[int, int | None]`
- `active_speakers: list[int]`
- `srp_doa_peaks_deg: list[float]`
- `srp_peak_scores: list[float] | None`

### Output (`DirectionAssignmentOutput`)
- `chunk_id: int`
- `timestamp_ms: float`
- `speaker_directions_deg: dict[int, float]`
- `speaker_confidence: dict[int, float]`
- `target_speaker_ids: list[int]`
- `target_doas_deg: list[float]`
- `target_weights: list[float]`
- `debug: dict`

## Balanced Payload Adapter
Use the adapter to construct `DirectionAssignmentInput` safely from realtime pipeline outputs:
- `build_direction_assignment_input(...) -> (DirectionAssignmentInput, BalancedPayloadBuildDebug)`
- `validate_balanced_payload(payload) -> None`

Adapter behavior:
- trims/pads separated stream lengths to match `raw_mic_chunk` sample length
- drops invalid `stream_to_speaker` keys not present in `separated_streams`
- recomputes `active_speakers` from `stream_to_speaker`
- drops `srp_peak_scores` if length mismatch with `srp_doa_peaks_deg`

## Beamforming Interop
Use output arrays directly with existing beamformer interfaces:
- `source_azimuths_deg = np.asarray(output.target_doas_deg, dtype=float)`
- `target_weights = np.asarray(output.target_weights, dtype=float)`

These are shape-compatible with `beamforming.compute_beamforming_weights(... source_azimuths_deg=..., target_weights=...)`.

## Focus/Enhancement Policy
`DirectionAssignmentEngine` supports runtime focus controls:
- `set_focus_speakers([speaker_ids])`
- `set_focus_direction(doa_deg)`

Default behavior:
- no focus -> equal weights
- focus speaker IDs -> focused=1.0, others=`non_focus_weight`
- focus direction only -> nearest speaker gets 1.0, others downweighted

## Important Config Knobs
In `DirectionAssignmentConfig`:
- `srp_snap_tolerance_deg`
- `doa_ema_alpha`
- `min_stream_rms`
- `min_pair_coherence`
- `speaker_stale_timeout_ms`
- `speaker_forget_timeout_ms`
- `non_focus_weight`

## Runtime Usage Example
```python
import numpy as np
from direction_assignment import (
    DirectionAssignmentEngine,
    DirectionAssignmentConfig,
    build_direction_assignment_input,
)

mic_geometry_xy = np.array([
    [0.00, 0.00],
    [0.04, 0.00],
    [0.08, 0.00],
    [0.12, 0.00],
])

engine = DirectionAssignmentEngine(
    mic_geometry=mic_geometry_xy,
    config=DirectionAssignmentConfig(sample_rate=16000),
)

payload, payload_debug = build_direction_assignment_input(
    chunk_id=42,
    timestamp_ms=8400.0,
    raw_mic_chunk=raw_chunk,              # (samples, n_mics)
    separated_streams=separated_streams,  # list[(samples,)] (normalized if needed)
    stream_to_speaker=stream_to_speaker,  # from speaker_identity_grouping
    active_speakers=active_speakers,      # recomputed if mismatch
    srp_doa_peaks_deg=srp_peaks,
    srp_peak_scores=srp_scores,           # dropped if mismatch length
)

out = engine.update(payload)

# Feed beamformer
source_azimuths_deg = np.asarray(out.target_doas_deg, dtype=float)
target_weights = np.asarray(out.target_weights, dtype=float)
```

## Integration Checklist for Next Agent
1. Add realtime call-site that builds `DirectionAssignmentInput` each chunk.
2. Confirm raw chunk and separated streams are time-aligned to the same chunk window.
3. Confirm mic geometry passed to engine matches the physical array frame used by beamforming/localization.
4. Route `target_doas_deg` and `target_weights` into beamformer steering update path.
5. Expose optional user controls for focus speaker or focus direction.
6. Add chunk-level logging/telemetry from `output.debug` for tuning.

## Validation Checklist
1. Shape checks pass for all chunks.
2. No NaN/inf in emitted directions or weights.
3. `target_doas_deg` and `target_weights` lengths always match.
4. Speaker directions remain stable across adjacent chunks when speakers are stationary.
5. Focus controls visibly alter `target_weights` as expected.

## Validation + Metrics + Artifacts
Use:

```bash
python -m direction_assignment.validate \
  --num-scenes 5 \
  --duration-sec 8 \
  --speaker-choices 2,3,4 \
  --export-audio \
  --export-plots
```

Outputs:
- `direction_assignment/output/validation_<timestamp>/summary.json`
- `direction_assignment/output/validation_<timestamp>/per_scene_metrics.csv`
- `direction_assignment/output/validation_<timestamp>/per_chunk_metrics.csv`
- `direction_assignment/output/validation_<timestamp>/per_speaker_metrics.csv`
- Per scene:
  - `scenario_metadata.json`
  - `track_assignments.csv`
  - `audio/*.wav` (mixture, oracle sources, estimated speaker tracks)
  - `plots/*.png` (DOA timeline, error histogram, room topdown, weight timeline)
