from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConversationScenarioConfig


@dataclass
class ScheduledUtterance:
    speaker_id: int
    start_sec: float
    end_sec: float
    kind: str
    interrupted: bool = False


@dataclass
class ConversationPlan:
    speaker_ids: list[int]
    utterances: list[ScheduledUtterance]


def build_conversation_plan(cfg: ConversationScenarioConfig, rng: np.random.Generator) -> ConversationPlan:
    tt = cfg.turn_taking
    max_duration_supported = max(tt.min_speakers, int(np.floor(cfg.render.duration_sec / max(1.2, tt.utterance_sec_range[0] * 0.85))))
    n_speakers = int(rng.integers(tt.min_speakers, min(tt.max_speakers, max_duration_supported) + 1))
    speaker_ids = list(range(n_speakers))
    utterances: list[ScheduledUtterance] = []
    remaining_intro_speakers = [int(v) for v in rng.permutation(speaker_ids)]

    current_t = float(rng.uniform(0.0, 0.35))
    current_speaker = remaining_intro_speakers.pop(0)

    while current_t < cfg.render.duration_sec - 0.15:
        utt_len = float(rng.uniform(tt.utterance_sec_range[0], tt.utterance_sec_range[1]))
        if remaining_intro_speakers:
            remaining_turns = len(remaining_intro_speakers) + 1
            remaining_time = max(0.4, cfg.render.duration_sec - current_t)
            intro_cap = max(tt.utterance_sec_range[0], remaining_time / remaining_turns - tt.pause_sec_range[0])
            utt_len = min(utt_len, intro_cap)
        end_t = min(cfg.render.duration_sec, current_t + utt_len)
        interrupted = False

        if (
            n_speakers > 1
            and rng.random() < tt.interruption_probability
            and end_t - current_t > tt.overlap_sec_range[1] + 0.25
        ):
            overlap = float(rng.uniform(tt.overlap_sec_range[0], tt.overlap_sec_range[1]))
            interrupter_choices = [sid for sid in speaker_ids if sid != current_speaker]
            interrupter = int(interrupter_choices[int(rng.integers(0, len(interrupter_choices)))])
            interrupt_start = max(current_t + 0.45, end_t - overlap)
            utterances.append(
                ScheduledUtterance(
                    speaker_id=current_speaker,
                    start_sec=current_t,
                    end_sec=interrupt_start + overlap * 0.45,
                    kind="utterance",
                    interrupted=True,
                )
            )
            utterances.append(
                ScheduledUtterance(
                    speaker_id=interrupter,
                    start_sec=interrupt_start,
                    end_sec=min(cfg.render.duration_sec, interrupt_start + float(rng.uniform(0.9, 2.4))),
                    kind="interruption",
                )
            )
            interrupted = True
            current_t = utterances[-1].end_sec + float(rng.uniform(tt.pause_sec_range[0], tt.pause_sec_range[1]))
            current_speaker = interrupter
        else:
            utterances.append(
                ScheduledUtterance(
                    speaker_id=current_speaker,
                    start_sec=current_t,
                    end_sec=end_t,
                    kind="utterance",
                )
            )
            if (
                n_speakers > 1
                and not remaining_intro_speakers
                and rng.random() < tt.overlap_probability
                and end_t - current_t > tt.overlap_sec_range[0] + 0.2
            ):
                overlap = float(rng.uniform(tt.overlap_sec_range[0], tt.overlap_sec_range[1]))
                other_choices = [sid for sid in speaker_ids if sid != current_speaker]
                other = int(other_choices[int(rng.integers(0, len(other_choices)))])
                utterances.append(
                    ScheduledUtterance(
                        speaker_id=other,
                        start_sec=max(current_t, end_t - overlap),
                        end_sec=min(cfg.render.duration_sec, end_t + float(rng.uniform(0.35, 1.2))),
                        kind="turn_overlap",
                    )
                )
                current_speaker = other
            elif n_speakers > 1 and not remaining_intro_speakers and rng.random() < tt.backchannel_probability:
                other_choices = [sid for sid in speaker_ids if sid != current_speaker]
                other = int(other_choices[int(rng.integers(0, len(other_choices)))])
                bc_len = float(rng.uniform(tt.backchannel_sec_range[0], tt.backchannel_sec_range[1]))
                bc_start = min(end_t - 0.08, current_t + float(rng.uniform(0.18, max(0.22, end_t - current_t - bc_len))))
                utterances.append(
                    ScheduledUtterance(
                        speaker_id=other,
                        start_sec=max(current_t, bc_start),
                        end_sec=min(cfg.render.duration_sec, max(current_t, bc_start) + bc_len),
                        kind="backchannel",
                    )
                )
                current_speaker = current_speaker if rng.random() < tt.persistence_probability else other
            else:
                if remaining_intro_speakers:
                    current_speaker = remaining_intro_speakers.pop(0)
                elif rng.random() >= tt.persistence_probability:
                    other_choices = [sid for sid in speaker_ids if sid != current_speaker]
                    if other_choices:
                        current_speaker = int(other_choices[int(rng.integers(0, len(other_choices)))])

            if not interrupted:
                current_t = end_t + float(rng.uniform(tt.pause_sec_range[0], tt.pause_sec_range[1]))

    utterances.sort(key=lambda item: (item.start_sec, item.end_sec, item.speaker_id))
    return ConversationPlan(speaker_ids=speaker_ids, utterances=utterances)
