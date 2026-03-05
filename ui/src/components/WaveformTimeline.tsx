import { useMemo } from "react";

type Props = {
  bins: number[];
  totalDurationMs: number;
  playheadMs: number;
};

function formatMs(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function WaveformTimeline({ bins, totalDurationMs, playheadMs }: Props) {
  const points = useMemo(() => {
    if (!bins.length) {
      return "";
    }
    const n = bins.length;
    return bins
      .map((v, i) => {
        const x = (i / Math.max(1, n - 1)) * 100;
        const y = 50 - Math.max(0, Math.min(1, v)) * 45;
        return `${x},${y}`;
      })
      .join(" ");
  }, [bins]);

  const playheadPct = totalDurationMs > 0 ? Math.max(0, Math.min(100, (playheadMs / totalDurationMs) * 100)) : 0;

  return (
    <section className="waveform-panel" aria-label="Waveform timeline">
      <div className="waveform-svg-wrap">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="waveform-svg">
          <polyline points={points || "0,50 100,50"} className="waveform-line" />
          <line x1={playheadPct} y1={0} x2={playheadPct} y2={100} className="waveform-playhead" />
        </svg>
      </div>
      <div className="waveform-times">
        <span>0:00</span>
        <span>
          {formatMs(playheadMs)} / {formatMs(totalDurationMs)}
        </span>
      </div>
    </section>
  );
}
