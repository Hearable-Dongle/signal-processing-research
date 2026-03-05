import { useMemo } from "react";

type PlaybackSource = "beamformed_output" | "raw_mixed_input";

type Props = {
  beamformedBins: number[];
  beamformedDurationMs: number;
  rawMixedBins: number[];
  rawMixedDurationMs: number;
  playheadMs: number;
  playbackSource: PlaybackSource;
  onPlaybackSourceChange: (value: PlaybackSource) => void;
  onTogglePlayback: () => void;
  canPlay: boolean;
  isPlaybackActive: boolean;
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
}: {
  label: string;
  bins: number[];
  durationMs: number;
  playheadMs: number;
}) {
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

  const playheadPct = durationMs > 0 ? Math.max(0, Math.min(100, (playheadMs / durationMs) * 100)) : 0;

  return (
    <div className="waveform-track">
      <div className="waveform-track-title">{label}</div>
      <div className="waveform-svg-wrap">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="waveform-svg">
          <polyline points={points || "0,50 100,50"} className="waveform-line" />
          <line x1={playheadPct} y1={0} x2={playheadPct} y2={100} className="waveform-playhead" />
        </svg>
      </div>
      <div className="waveform-times">
        <span>0:00</span>
        <span>
          {formatMs(playheadMs)} / {formatMs(durationMs)}
        </span>
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
  playbackSource,
  onPlaybackSourceChange,
  onTogglePlayback,
  canPlay,
  isPlaybackActive,
}: Props) {

  return (
    <section className="waveform-panel" aria-label="Waveform timeline">
      <WaveformTrack
        label="Beamformed output"
        bins={beamformedBins}
        durationMs={beamformedDurationMs}
        playheadMs={playheadMs}
      />
      <WaveformTrack label="Raw mixed input" bins={rawMixedBins} durationMs={rawMixedDurationMs} playheadMs={playheadMs} />
      <div className="waveform-controls">
        <label htmlFor="playback-source">Playback source</label>
        <select
          id="playback-source"
          aria-label="Playback source"
          value={playbackSource}
          onChange={(e) => onPlaybackSourceChange(e.target.value as PlaybackSource)}
        >
          <option value="beamformed_output">Beamformed output</option>
          <option value="raw_mixed_input">Raw mixed input</option>
        </select>
        <button onClick={onTogglePlayback} disabled={!canPlay}>
          {isPlaybackActive ? "Stop Playback" : "Play Output"}
        </button>
      </div>
    </section>
  );
}
