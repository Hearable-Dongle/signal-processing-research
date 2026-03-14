import type { Speaker } from "../types/contracts";

type DirectionSample = {
  directionDeg: number;
  intensity: number;
  atMs: number;
};

type Props = {
  speakers: Speaker[];
  trail: DirectionSample[];
};

function polarToXY(directionDeg: number, radius: number): { x: number; y: number } {
  const radians = (directionDeg * Math.PI) / 180;
  return {
    x: 50 + radius * Math.sin(radians),
    y: 50 - radius * Math.cos(radians),
  };
}

function wedgePath(directionDeg: number, widthDeg: number, radius: number): string {
  const start = polarToXY(directionDeg - widthDeg / 2, radius);
  const end = polarToXY(directionDeg + widthDeg / 2, radius);
  const largeArc = widthDeg > 180 ? 1 : 0;
  return `M 50 50 L ${start.x.toFixed(2)} ${start.y.toFixed(2)} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x.toFixed(2)} ${end.y.toFixed(2)} Z`;
}

export function DirectionalityViz({ speakers, trail }: Props) {
  const nowMs = Date.now();
  const activeSpeakers = speakers.filter((speaker) => speaker.active);
  const visibleTrail = trail.filter((sample) => nowMs - sample.atMs <= 4000);

  return (
    <section className="panel directionality-panel">
      <div className="directionality-header">
        <h2>Directionality</h2>
        <p className="directionality-copy">Frontend preview of the estimated speaker bearing around the array.</p>
      </div>

      <div className="directionality-wrap">
        <svg
          viewBox="0 0 100 100"
          className="directionality-viz"
          role="img"
          aria-label="Directionality visualization"
          data-testid="directionality-viz"
        >
          <defs>
            <radialGradient id="directionality-bg" cx="50%" cy="50%" r="55%">
              <stop offset="0%" stopColor="#fffaf4" />
              <stop offset="100%" stopColor="#f2ddcd" />
            </radialGradient>
          </defs>
          <circle cx="50" cy="50" r="44" className="directionality-ring-bg" />
          <circle cx="50" cy="50" r="36" className="directionality-ring" />
          <circle cx="50" cy="50" r="24" className="directionality-ring directionality-ring-inner" />
          <line x1="50" y1="50" x2="50" y2="10" className="directionality-zero-axis" />
          <text x="50" y="8" textAnchor="middle" className="directionality-zero-label">
            0° cable
          </text>

          {visibleTrail.map((sample, index) => {
            const ageRatio = Math.max(0, 1 - (nowMs - sample.atMs) / 4000);
            const { x, y } = polarToXY(sample.directionDeg, 18 + sample.intensity * 24);
            return (
              <circle
                key={`${sample.atMs}-${index}`}
                cx={x}
                cy={y}
                r={1.5 + sample.intensity * 1.4}
                className="directionality-trail-dot"
                style={{ opacity: 0.12 + ageRatio * 0.28 }}
              />
            );
          })}

          {activeSpeakers.map((speaker) => {
            const intensity = Math.max(speaker.confidence, speaker.activity_confidence, 0.25);
            const widthDeg = 18 + (1 - Math.min(intensity, 1)) * 20;
            const { x, y } = polarToXY(speaker.direction_degrees, 40);
            return (
              <g key={speaker.speaker_id}>
                <path
                  d={wedgePath(speaker.direction_degrees, widthDeg, 40)}
                  className="directionality-wedge"
                  style={{ opacity: 0.22 + intensity * 0.36 }}
                />
                <line x1="50" y1="50" x2={x} y2={y} className="directionality-spoke" />
                <circle cx={x} cy={y} r={3.2} className="directionality-node" />
              </g>
            );
          })}

          <circle cx="50" cy="50" r="4" className="directionality-center" />
        </svg>

        <div className="directionality-legend">
          <div className="directionality-legend-item">
            <span className="directionality-chip directionality-chip-live" />
            <span>Current dominant directions</span>
          </div>
          <div className="directionality-legend-item">
            <span className="directionality-chip directionality-chip-trace" />
            <span>Recent direction trail</span>
          </div>
          <p className="directionality-note">The cable edge is treated as the XVF3800 0° reference.</p>
        </div>
      </div>
    </section>
  );
}
