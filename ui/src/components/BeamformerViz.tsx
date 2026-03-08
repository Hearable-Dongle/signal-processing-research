import type { ProcessingMode, Speaker } from "../types/contracts";
import { speakerColor } from "../utils/color";

type Props = {
  speakers: Speaker[];
  processingMode: ProcessingMode;
  selectedSpeakerId: number | null;
};

function pointFor(directionDeg: number, magnitude: number): { x: number; y: number } {
  const radians = (directionDeg * Math.PI) / 180;
  const radius = 16 + magnitude * 26;
  return {
    x: 50 + radius * Math.cos(radians),
    y: 50 - radius * Math.sin(radians),
  };
}

export function BeamformerViz({ speakers, processingMode, selectedSpeakerId }: Props) {
  if (processingMode === "beamform_from_ground_truth") {
    return null;
  }

  const visibleSpeakers =
    processingMode === "localize_and_beamform" ? speakers.filter((speaker) => speaker.active) : speakers;

  if (!visibleSpeakers.length) {
    return (
      <div className="beamformer-panel">
        <div className="beamformer-header">
          <h3>Beamforming Weights</h3>
        </div>
        <p className="beamformer-empty">No tracked speakers available.</p>
      </div>
    );
  }

  const maxWeight = Math.max(...visibleSpeakers.map((speaker) => Math.max(0, speaker.gain_weight)), 1);
  const sorted = [...visibleSpeakers].sort((a, b) => a.direction_degrees - b.direction_degrees);
  const lobePath = sorted
    .map((speaker, index) => {
      const normalized = Math.max(0, speaker.gain_weight) / maxWeight;
      const { x, y } = pointFor(speaker.direction_degrees, normalized);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <div className="beamformer-panel">
      <div className="beamformer-header">
        <h3>Beamforming Weights</h3>
      </div>
      <div className="beamformer-viz-wrap">
        <svg
          viewBox="0 0 100 100"
          className="beamformer-viz"
          role="img"
          aria-label="Beamforming visualization"
          data-testid="beamformer-viz"
        >
          <circle cx="50" cy="50" r="34" className="beamformer-ring beamformer-ring-outer" />
          <circle cx="50" cy="50" r="22" className="beamformer-ring beamformer-ring-inner" />
          <path d={`${lobePath} Z`} className="beamformer-lobe" />
          <circle cx="50" cy="50" r="4" className="beamformer-center" />
          {sorted.map((speaker) => {
            const normalized = Math.max(0, speaker.gain_weight) / maxWeight;
            const { x, y } = pointFor(speaker.direction_degrees, normalized);
            const isSelected = selectedSpeakerId === speaker.speaker_id;
            return (
              <g key={speaker.speaker_id}>
                <line x1="50" y1="50" x2={x} y2={y} className="beamformer-spoke" />
                <circle
                  cx={x}
                  cy={y}
                  r={isSelected ? 4.5 : 3.2}
                  className={`beamformer-node ${isSelected ? "selected" : ""}`}
                  style={{ fill: speakerColor(speaker.speaker_id) }}
                  data-testid={`beam-node-${speaker.speaker_id}`}
                />
              </g>
            );
          })}
        </svg>
        <div className="beamformer-weight-list">
          {sorted.map((speaker) => (
            <div
              key={speaker.speaker_id}
              className={`beamformer-weight-item ${selectedSpeakerId === speaker.speaker_id ? "selected" : ""}`}
              data-testid={`beam-weight-${speaker.speaker_id}`}
            >
              <span className="beamformer-weight-name">Speaker {speaker.speaker_id}</span>
              <span className="beamformer-weight-meta">{speaker.direction_degrees.toFixed(1)} deg</span>
              <span className="beamformer-weight-value">Weight {speaker.gain_weight.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
