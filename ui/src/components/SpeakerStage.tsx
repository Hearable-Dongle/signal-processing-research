import { speakerColor } from "../utils/color";
import type { Speaker } from "../types/contracts";

type Props = {
  speakers: Speaker[];
  selectedSpeakerId: number | null;
  onSpeakerTap: (speakerId: number) => void;
};

function polarToXY(directionDeg: number): { x: number; y: number } {
  const radians = (directionDeg * Math.PI) / 180;
  const radius = 42;
  const x = 50 + radius * Math.cos(radians);
  const y = 50 - radius * Math.sin(radians);
  return { x, y };
}

export function SpeakerStage({ speakers, selectedSpeakerId, onSpeakerTap }: Props) {
  return (
    <section className="panel speaker-stage">
      <h2>Speaker Stage</h2>
      <div className="room" data-testid="speaker-stage">
        <div className="mic-center" />
        {speakers.map((speaker) => {
          const { x, y } = polarToXY(speaker.direction_degrees);
          const selected = selectedSpeakerId === speaker.speaker_id;
          return (
            <button
              key={speaker.speaker_id}
              className={`speaker-dot ${selected ? "selected" : ""} ${speaker.active ? "" : "inactive"}`}
              data-testid={`speaker-${speaker.speaker_id}`}
              aria-label={`speaker-${speaker.speaker_id}`}
              onClick={() => onSpeakerTap(speaker.speaker_id)}
              onTouchStart={() => onSpeakerTap(speaker.speaker_id)}
              style={{ left: `${x}%`, top: `${y}%`, backgroundColor: speakerColor(speaker.speaker_id) }}
            >
              {speaker.speaker_id}
            </button>
          );
        })}
      </div>
    </section>
  );
}
