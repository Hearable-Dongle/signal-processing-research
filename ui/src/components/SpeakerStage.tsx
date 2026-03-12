import { speakerColor } from "../utils/color";
import type { GroundTruthSpeaker, ProcessingMode, Speaker } from "../types/contracts";

type Props = {
  speakers: Speaker[];
  groundTruth: GroundTruthSpeaker[];
  processingMode: ProcessingMode;
  selectedSpeakerId: number | null;
  onSpeakerTap: (speakerId: number) => void;
};

const MIC_MARKERS = [
  { id: "mic-1", label: "1", x: 50, y: 31 },
  { id: "mic-2", label: "2", x: 69, y: 50 },
  { id: "mic-3", label: "3", x: 50, y: 69 },
  { id: "mic-4", label: "4", x: 31, y: 50 },
];

const CABLE_MARKER = { id: "cable", label: "cable", x: 59.5, y: 40.5 };

function polarToXY(directionDeg: number): { x: number; y: number } {
  const radians = (directionDeg * Math.PI) / 180;
  const radius = 42;
  const x = 50 + radius * Math.cos(radians);
  const y = 50 - radius * Math.sin(radians);
  return { x, y };
}

function MicArrayMarkers({ prefix }: { prefix: string }) {
  return (
    <>
      {MIC_MARKERS.map((marker) => (
        <div
          key={`${prefix}-${marker.id}`}
          className="mic-array-marker"
          data-testid={`${prefix}-${marker.id}`}
          style={{ left: `${marker.x}%`, top: `${marker.y}%` }}
        >
          {marker.label}
        </div>
      ))}
      <div
        className="mic-array-cable"
        data-testid={`${prefix}-${CABLE_MARKER.id}`}
        style={{ left: `${CABLE_MARKER.x}%`, top: `${CABLE_MARKER.y}%` }}
      >
        {CABLE_MARKER.label}
      </div>
    </>
  );
}

export function SpeakerStage({ speakers, groundTruth, processingMode, selectedSpeakerId, onSpeakerTap }: Props) {
  const activeSpeakers = speakers.filter((speaker) => speaker.active);
  const showTrackedRoom = processingMode !== "beamform_from_ground_truth";
  const showGroundTruthBlock = true;

  return (
    <section className="panel speaker-stage">
      <h2>Speaker Stage</h2>
      {showTrackedRoom && (
        <>
          <div className="room" data-testid="speaker-stage">
            <div className="mic-center" />
            <MicArrayMarkers prefix="tracked" />
            {(processingMode === "localize_and_beamform" ? activeSpeakers : speakers).map((speaker) => {
              const { x, y } = polarToXY(speaker.direction_degrees);
              const selected = selectedSpeakerId === speaker.speaker_id;
              if (processingMode === "specific_speaker_enhancement") {
                return (
                  <button
                    key={speaker.speaker_id}
                    className={`speaker-dot ${selected ? "selected" : ""} ${speaker.active ? "active" : "inactive"}`}
                    data-testid={`speaker-${speaker.speaker_id}`}
                    aria-label={`speaker-${speaker.speaker_id}`}
                    onClick={() => onSpeakerTap(speaker.speaker_id)}
                    onTouchStart={() => onSpeakerTap(speaker.speaker_id)}
                    style={{ left: `${x}%`, top: `${y}%`, backgroundColor: speakerColor(speaker.speaker_id) }}
                    title={`Weight ${speaker.gain_weight.toFixed(2)}`}
                  >
                    {speaker.speaker_id}
                  </button>
                );
              }
              return (
                <div
                  key={speaker.speaker_id}
                  className={`speaker-dot speaker-dot-passive ${speaker.active ? "active" : "inactive"}`}
                  data-testid={`active-speaker-${speaker.speaker_id}`}
                  style={{ left: `${x}%`, top: `${y}%`, backgroundColor: speakerColor(speaker.speaker_id) }}
                  title={`${speaker.direction_degrees.toFixed(1)}° / weight ${speaker.gain_weight.toFixed(2)}`}
                />
              );
            })}
          </div>
        </>
      )}
      {showGroundTruthBlock && (
        <div className="ground-truth-block">
          <h3>Ground Truth</h3>
          {groundTruth.length === 0 ? (
            <p className="ground-truth-empty">No ground truth loaded.</p>
          ) : (
            <div className="ground-truth-viz-wrap">
              <div className="ground-truth-room" data-testid="ground-truth-stage">
                <div className="mic-center" />
                <MicArrayMarkers prefix="ground-truth" />
                {groundTruth.map((gt) => {
                  const { x, y } = polarToXY(gt.direction_degrees);
                  return (
                    <div
                      key={gt.source_id}
                      className="ground-truth-dot"
                      data-testid={`ground-truth-${gt.source_id}`}
                      style={{ left: `${x}%`, top: `${y}%` }}
                      aria-label={`ground-truth-${gt.source_id}`}
                      title={`src ${gt.source_id}: ${gt.direction_degrees.toFixed(1)}°`}
                    >
                      {processingMode === "specific_speaker_enhancement" ? gt.source_id : null}
                    </div>
                  );
                })}
              </div>
              {processingMode === "specific_speaker_enhancement" && (
                <div className="ground-truth-list">
                  {groundTruth.map((gt) => (
                    <span key={gt.source_id} className="ground-truth-item">
                      src {gt.source_id}: {gt.direction_degrees.toFixed(1)}°
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
