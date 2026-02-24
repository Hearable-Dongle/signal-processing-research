from __future__ import annotations

import numpy as np

from .config import DirectionAssignmentConfig
from .doa_estimation import estimate_stream_doa
from .geometry import build_mic_pairs
from .mask_backprojection import backproject_streams_to_multichannel
from .tracking import SpeakerObservation, aggregate_speaker_observations, update_speaker_states
from .types import DirectionAssignmentInput, DirectionAssignmentOutput, SpeakerDirectionState
from .weight_policy import build_target_weights


class DirectionAssignmentEngine:
    def __init__(
        self,
        mic_geometry: np.ndarray,
        config: DirectionAssignmentConfig | None = None,
        mic_pairs: list[tuple[int, int]] | None = None,
    ):
        self.cfg = config or DirectionAssignmentConfig()
        self.mic_geometry = np.asarray(mic_geometry, dtype=float)
        if self.mic_geometry.ndim != 2:
            raise ValueError("mic_geometry must be 2D")
        if self.mic_geometry.shape[1] > 2:
            self.mic_geometry = self.mic_geometry[:, :2]

        self.mic_pairs = mic_pairs or build_mic_pairs(
            self.mic_geometry,
            min_baseline_m=self.cfg.min_pair_baseline_m,
        )
        if not self.mic_pairs:
            raise ValueError("No valid microphone pairs available for direction assignment")

        self._states: dict[int, SpeakerDirectionState] = {}
        self._focus_speaker_ids: set[int] | None = None
        self._focus_direction_deg: float | None = None

    def set_focus_speakers(self, speaker_ids: list[int] | None) -> None:
        self._focus_speaker_ids = set(speaker_ids) if speaker_ids else None

    def set_focus_direction(self, doa_deg: float | None) -> None:
        self._focus_direction_deg = None if doa_deg is None else float(doa_deg % 360.0)

    def reset(self) -> None:
        self._states.clear()

    def _validate_input(self, payload: DirectionAssignmentInput) -> None:
        if payload.raw_mic_chunk.ndim != 2:
            raise ValueError("raw_mic_chunk must be shape (samples, n_mics)")
        if payload.raw_mic_chunk.shape[1] != self.mic_geometry.shape[0]:
            raise ValueError("raw_mic_chunk channel count does not match mic_geometry")

    def update(self, payload: DirectionAssignmentInput) -> DirectionAssignmentOutput:
        self._validate_input(payload)

        debug: dict = {
            "chunk_id": payload.chunk_id,
            "stream_debug": {},
            "backprojection": {},
        }

        streams_mc, bp_debug = backproject_streams_to_multichannel(
            raw_mic_chunk=payload.raw_mic_chunk,
            separated_streams=payload.separated_streams,
            cfg=self.cfg,
        )
        debug["backprojection"] = bp_debug

        observations: list[SpeakerObservation] = []

        for stream_idx, speaker_id in payload.stream_to_speaker.items():
            if speaker_id is None:
                continue
            if stream_idx < 0 or stream_idx >= len(streams_mc):
                continue

            stream_mono = np.asarray(payload.separated_streams[stream_idx], dtype=float).reshape(-1)
            rms = float(np.sqrt(np.mean(stream_mono ** 2) + 1e-12))
            if rms < self.cfg.min_stream_rms:
                debug["stream_debug"][stream_idx] = {
                    "speaker_id": int(speaker_id),
                    "skipped": "low_rms",
                    "rms": rms,
                }
                continue

            doa, conf, doa_debug = estimate_stream_doa(
                stream_multichannel=streams_mc[stream_idx],
                mic_geometry=self.mic_geometry,
                mic_pairs=self.mic_pairs,
                cfg=self.cfg,
            )
            if doa is None:
                debug["stream_debug"][stream_idx] = {
                    "speaker_id": int(speaker_id),
                    "skipped": "doa_failed",
                    "rms": rms,
                    **doa_debug,
                }
                continue

            observations.append(
                SpeakerObservation(
                    speaker_id=int(speaker_id),
                    doa_deg=float(doa),
                    confidence=float(conf),
                )
            )

            debug["stream_debug"][stream_idx] = {
                "speaker_id": int(speaker_id),
                "rms": rms,
                "doa_deg": float(doa),
                "confidence": float(conf),
                **doa_debug,
            }

        aggregated, agg_debug = aggregate_speaker_observations(observations)
        debug["aggregation"] = agg_debug

        self._states, track_debug = update_speaker_states(
            states=self._states,
            aggregated_obs=aggregated,
            timestamp_ms=payload.timestamp_ms,
            cfg=self.cfg,
            srp_peaks_deg=payload.srp_doa_peaks_deg,
            srp_peak_scores=payload.srp_peak_scores,
        )
        debug["tracking"] = track_debug

        # Keep only non-stale states for active output.
        stale_ids = set(track_debug.get("stale_speakers", []))
        output_states = {
            sid: st
            for sid, st in self._states.items()
            if sid not in stale_ids
        }

        speaker_directions = {
            int(sid): float(st.direction_deg)
            for sid, st in output_states.items()
        }
        speaker_confidence = {
            int(sid): float(st.confidence)
            for sid, st in output_states.items()
        }

        active_candidates = sorted(output_states.keys())
        target_weights = build_target_weights(
            states=output_states,
            candidate_ids=active_candidates,
            focus_speaker_ids=self._focus_speaker_ids,
            focus_direction_deg=self._focus_direction_deg,
            cfg=self.cfg,
        )

        paired = list(zip(active_candidates, target_weights))
        paired.sort(key=lambda x: (-x[1], x[0]))

        target_speaker_ids = [int(sid) for sid, _ in paired]
        target_doas_deg = [float(output_states[sid].direction_deg) for sid, _ in paired]
        target_weights_sorted = [float(w) for _, w in paired]

        return DirectionAssignmentOutput(
            chunk_id=payload.chunk_id,
            timestamp_ms=payload.timestamp_ms,
            speaker_directions_deg=speaker_directions,
            speaker_confidence=speaker_confidence,
            target_speaker_ids=target_speaker_ids,
            target_doas_deg=target_doas_deg,
            target_weights=target_weights_sorted,
            debug=debug,
        )
