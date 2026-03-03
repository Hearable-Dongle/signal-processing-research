import type { MetricsMessage } from "../types/contracts";
import type { PlaybackStats } from "../audio/player";

type Props = {
  metrics: MetricsMessage | null;
  playback: PlaybackStats;
};

export function MetricsPanel({ metrics, playback }: Props) {
  return (
    <section className="panel">
      <h2>Metrics</h2>
      {!metrics ? (
        <p>No metrics yet.</p>
      ) : (
        <div className="metric-grid">
          <p>Fast RTF: {metrics.fast_rtf.toFixed(3)}</p>
          <p>Slow RTF: {metrics.slow_rtf.toFixed(3)}</p>
          <p>Startup lock: {metrics.startup_lock_ms.toFixed(1)} ms</p>
          <p>Reacquire median: {metrics.reacquire_catchup_ms_median.toFixed(1)} ms</p>
          <p>Nearest-change median: {metrics.nearest_change_catchup_ms_median.toFixed(1)} ms</p>
        </div>
      )}
      <h3>Playback</h3>
      <div className="metric-grid">
        <p>Buffered: {playback.buffered_ms.toFixed(1)} ms</p>
        <p>Drift: {playback.drift_ms.toFixed(1)} ms</p>
        <p>Underruns: {playback.underrun_count}</p>
        <p>Re-anchors: {playback.reanchor_count}</p>
        <p>Parse errors: {playback.parse_error_count}</p>
      </div>
    </section>
  );
}
