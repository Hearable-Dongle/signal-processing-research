import { useMemo } from "react";

type Props = {
  beamformedBins: number[];
  beamformedDurationMs: number;
  rawMixedBins: number[];
  rawMixedDurationMs: number;
  playheadMs: number;
  canPlayBeamformed: boolean;
  canPauseBeamformed: boolean;
  isBeamformedPlaying: boolean;
  onPlayBeamformed: () => void;
  onPauseBeamformed: () => void;
  canPlayRawMixed: boolean;
  isRawMixedPlaying: boolean;
  onToggleRawMixedPlayback: () => void;
};

function formatMs(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function WaveformTrack({
  label,
  bins,
  durationMs,
  playheadMs,
  canPlay,
  isPlaying,
  onTogglePlayback,
  onPausePlayback,
  showPauseButton,
  primaryLabel,
}: {
  label: string;
  bins: number[];
  durationMs: number;
  playheadMs: number;
  canPlay: boolean;
  isPlaying: boolean;
  onTogglePlayback: () => void;
  onPausePlayback?: () => void;
  showPauseButton?: boolean;
  primaryLabel?: string;
}) {
  const smoothed = useMemo(() => {
    if (!bins.length) {
      return [];
    }
    const out = new Array<number>(bins.length);
    for (let i = 0; i < bins.length; i += 1) {
      const a = bins[Math.max(0, i - 1)];
      const b = bins[i];
      const c = bins[Math.min(bins.length - 1, i + 1)];
      out[i] = Math.max(0, Math.min(1, (a + 2 * b + c) / 4));
    }
    return out;
  }, [bins]);

  const envelopePath = useMemo(() => {
    if (!smoothed.length) {
      return "M 0 50 L 100 50";
    }
    return smoothed
      .map((v, i) => {
        const x = (i / Math.max(1, smoothed.length - 1)) * 100;
        const y = 50 - v * 40;
        return `${i === 0 ? "M" : "L"} ${x} ${y}`;
      })
      .join(" ");
  }, [smoothed]);

  const playheadPct = durationMs > 0 ? Math.max(0, Math.min(100, (playheadMs / durationMs) * 100)) : 0;

  return (
    <div className="waveform-track">
      <div className="waveform-track-title">{label}</div>
      <div className="waveform-svg-wrap">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="waveform-svg">
          <path d={envelopePath} className="waveform-trace" data-testid={`${label}-trace`} />
          <line x1={0} y1={50} x2={100} y2={50} className="waveform-midline" />
          <line x1={playheadPct} y1={0} x2={playheadPct} y2={100} className="waveform-playhead" />
        </svg>
      </div>
      <div className="waveform-times">
        <span>0:00</span>
        <span>
          {formatMs(playheadMs)} / {formatMs(durationMs)}
        </span>
      </div>
      <div className="waveform-track-actions">
        <button onClick={onTogglePlayback} disabled={!canPlay}>
          {primaryLabel ?? (isPlaying ? `Stop ${label}` : `Play ${label}`)}
        </button>
        {showPauseButton && (
          <button onClick={onPausePlayback} disabled={!isPlaying}>
            Pause {label}
          </button>
        )}
      </div>
    </div>
  );
}

export function WaveformTimeline({
  beamformedBins,
  beamformedDurationMs,
  rawMixedBins,
  rawMixedDurationMs,
  playheadMs,
  canPlayBeamformed,
  canPauseBeamformed,
  isBeamformedPlaying,
  onPlayBeamformed,
  onPauseBeamformed,
  canPlayRawMixed,
  isRawMixedPlaying,
  onToggleRawMixedPlayback,
}: Props) {
  return (
    <section className="waveform-panel" aria-label="Waveform timeline">
      <WaveformTrack
        label="Beamformed output"
        bins={beamformedBins}
        durationMs={beamformedDurationMs}
        playheadMs={playheadMs}
        canPlay={canPlayBeamformed}
        isPlaying={isBeamformedPlaying}
        onTogglePlayback={onPlayBeamformed}
        onPausePlayback={onPauseBeamformed}
        showPauseButton={canPauseBeamformed || isBeamformedPlaying}
        primaryLabel="Play Beamformed output"
      />
      <WaveformTrack
        label="Raw mixed input"
        bins={rawMixedBins}
        durationMs={rawMixedDurationMs}
        playheadMs={playheadMs}
        canPlay={canPlayRawMixed}
        isPlaying={isRawMixedPlaying}
        onTogglePlayback={onToggleRawMixedPlayback}
      />
    </section>
  );
}
