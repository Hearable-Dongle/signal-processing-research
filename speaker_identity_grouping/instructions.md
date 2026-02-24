# Speaker Identity Grouping: Implementation Reference

## Purpose
Provide persistent speaker IDs for realtime separated streams.

This module only performs identity grouping. It does not perform DOA estimation, direction assignment, beamforming, or separation.

## Scope and Non-Scope
- In scope: per-chunk stream-to-speaker ID mapping with stable IDs, inactivity retirement, and confidence values.
- Out of scope: speaker recognition, enrollment, diarization timelines, beam steering, localization.

## Runtime Constraints
- Designed for CPU-only realtime operation.
- Intended chunk size: ~200 ms at 16 kHz (configurable).
- Default behavior uses in-memory state only (no persistence across process restart).

## Module Layout
- `speaker_identity_grouping/contracts.py`
: Input/output/config/state dataclasses.
- `speaker_identity_grouping/grouper.py`
: `SpeakerIdentityGrouper` implementation.
- `speaker_identity_grouping/__init__.py`
: Public exports.

## Public Contract

### Input (`IdentityChunkInput`)
- `chunk_id: int`
- `timestamp_ms: float`
- `sample_rate_hz: int`
- `streams: list[np.ndarray]` where each stream is mono for one separated source.

### Output (`IdentityChunkOutput`)
- `chunk_id: int`
- `timestamp_ms: float`
- `stream_to_speaker: dict[int, int | None]`
- `active_speakers: list[int]`
- `new_speakers: list[int]`
- `retired_speakers: list[int]`
- `per_stream_confidence: dict[int, float]`
- `debug: dict[str, object]`

`None` means the stream was treated as inactive (typically below VAD RMS threshold).

## Configuration (`IdentityConfig`)
Defaults:
- `sample_rate_hz=16000`
- `chunk_duration_ms=200`
- `vad_rms_threshold=0.01`
- `match_threshold=0.82`
- `ema_alpha=0.1`
- `max_speakers=8`
- `retire_after_chunks=25`
- `new_speaker_confidence=0.5`

Embedding parameters:
- `n_mfcc=13`
- `n_mels=26`
- `n_fft=512`
- `frame_length_ms=25.0`
- `frame_hop_ms=10.0`
- `preemphasis=0.97`

## Algorithm
1. Retire stale speakers whose `current_chunk_id - last_seen_chunk > retire_after_chunks`.
2. For each stream:
- Compute RMS.
- If RMS below threshold: mark inactive.
- Else extract MFCC-based embedding and normalize.
3. For active streams, compute cosine similarity to active speaker centroids.
4. Apply one-to-one assignment with Hungarian (`linear_sum_assignment`) on cost `1 - similarity`.
5. Accept match only when similarity >= `match_threshold`.
6. Unmatched streams:
- If registry has capacity: create new speaker ID.
- Else: force-map to nearest existing speaker (continuity fallback).
7. Update speaker centroids using EMA and re-normalize.
8. Emit output contract.

## Usage
```python
from speaker_identity_grouping import (
    IdentityChunkInput,
    IdentityConfig,
    SpeakerIdentityGrouper,
)

cfg = IdentityConfig()
grouper = SpeakerIdentityGrouper(cfg)

chunk = IdentityChunkInput(
    chunk_id=12,
    timestamp_ms=2400.0,
    sample_rate_hz=16000,
    streams=separated_streams_mono,
)

out = grouper.update(chunk)
# out.stream_to_speaker: {0: 2, 1: None, 2: 0}
```

## Edge-Case Behavior
- Silent/noisy low-energy stream: returns `None` speaker assignment.
- Registry full and unmatched stream appears: reuses nearest existing speaker ID and records fallback in `debug["forced_reuse_streams"]`.
- Long inactivity: speaker ID retired and reported in `retired_speakers`.
- Sample-rate mismatch: raises `ValueError`.

## Integration Guidance
- Keep this module independent from `direction_assignment` and `beamforming`.
- Downstream modules should consume `IdentityChunkOutput` only.
- Integration point should be the separation orchestration path immediately after separated streams are produced.

## Tests and Acceptance Checklist
Current coverage is in `tests/test_speaker_identity_grouping.py`:
- stable ID across chunks for same speaker
- silent stream handling
- two-stream separation into distinct IDs
- TTL retirement after inactivity
- max speaker cap fallback behavior

Acceptance for implementation work:
- all tests pass
- contract keys remain stable
- no cross-module dependency added from this module into beamforming/localization

## Tradeoffs
- MFCC+cosine is lightweight and robust for small speaker counts, but weaker than learned speaker encoders in hard overlap conditions.
- One-to-one per-chunk assignment reduces duplicate speaker IDs but may still be affected by severe source bleeding.
- TTL retirement keeps state bounded but can break continuity after long silence gaps.

## Future TODOs
- Optional `never_expire` mode for long-session continuity experiments.
- Optional persistent registry backend (load/save state).
- Optional learned embedding backends (ECAPA/x-vector) behind same public contract.
- Future orchestration-level mapping from `speaker_id` to beamforming weight policies.

## Validation Script
Use `speaker_identity_grouping/validate.py` to measure grouping quality on LibriMix with ConvTasNet-separated chunks.

Validation dependencies (in your active env):
```bash
pip install numpy scipy soundfile torch asteroid requests
```

Run from repo root:
```bash
python -m speaker_identity_grouping.validate \
  --mix Libri2Mix \
  --sample-rate 16000 \
  --mode min \
  --subset test \
  --max-mixtures 25 \
  --chunk-ms 200 \
  --device cpu
```

### Outputs
- `summary.json`: run config + overall metrics.
- `per_mixture_metrics.csv`: per-utterance metrics.
- `pair_rows.csv`: chunk-level oracle/predicted identity pairs.

### Primary metrics
- `majority_vote_accuracy`: for each oracle source, fraction of chunks mapped to its dominant predicted speaker ID.
- `switch_rate`: frequency of predicted speaker-ID changes over time for the same oracle source.
- `speaker_count_ratio`: `unique_pred_speakers / unique_oracle_sources`.
- `avg_identity_ms_per_chunk`: identity module latency.
- `avg_total_ms_per_chunk`, `realtime_factor_total`: total separation + identity timing vs chunk budget.

## Reconstruction Demo Script
Use `speaker_identity_grouping/reconstruct_examples.py` to export listenable examples of chunked ConvTasNet output stitched into persistent `speaker_id` tracks.

Run from repo root:
```bash
python -m speaker_identity_grouping.reconstruct_examples \
  --mix Libri2Mix \
  --sample-rate 16000 \
  --mode min \
  --subset test \
  --num-mixtures 10 \
  --chunk-ms 200 \
  --hop-ratio 0.5 \
  --device cpu
```

### Per-mixture outputs
- `mixture.wav`
- `gt_source_1.wav`, `gt_source_2.wav` (or more for higher-speaker mixtures)
- `speaker_<id>_stitched.wav` (persistent reconstructed stream per speaker ID)
- `chunk_mappings.csv` (per-chunk/per-stream identity diagnostics)
- `speaker_activity.csv`
- `reconstruction_summary.json`

### Run-level outputs
- `index.csv`
- `run_summary.json`

### Stitching behavior
- Chunks are combined with Hann overlap-add (`hop_ratio=0.5` by default).
- Identity mapping (`stream_to_speaker`) controls which persistent speaker track each chunk stream contributes to.
- Unassigned (`None`) streams are skipped.

## Listening Report Utility
Use `speaker_identity_grouping/listening_report.py` to rank reconstructed examples by instability and generate short snippets around problematic chunks.

Run from repo root:
```bash
python -m speaker_identity_grouping.listening_report \
  --reconstruct-dir speaker_identity_grouping/output/reconstruct_smoke
```

### Outputs (inside `<reconstruct-dir>/listening_report/`)
- `ranked_examples.csv`
- `ranked_examples.json`
- `top_listen_first.txt`
- `snippets/<mixture_id>/...wav`

### Ranking score
Composite instability score:
- `3.0 * flip_rate`
- `2.0 * mapping_change_rate`
- `2.0 * unassigned_rate`
- `1.0 * (1 - mean_confidence_assigned)`

Higher score means higher priority for listening/debugging.
