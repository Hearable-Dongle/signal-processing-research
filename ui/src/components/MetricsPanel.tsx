import type { PlaybackStats } from "../audio/player";
import type { MetricsMessage } from "../types/contracts";

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

      <h3>Playback Queue</h3>
      <div className="metric-grid">
        <p>State: {playback.play_state}</p>
        <p>Buffered: {playback.buffered_ms.toFixed(1)} ms</p>
        <p>Queued packets: {playback.queued_packet_count}</p>
        <p>Dropped packets: {playback.dropped_packet_count}</p>
        <p>Late packets: {playback.late_packet_count}</p>
        <p>Rebuffers: {playback.rebuffer_count}</p>
        <p>Reanchors: {playback.reanchor_count}</p>
        <p>Startup wait: {playback.startup_gate_wait_ms.toFixed(1)} ms</p>
        <p>Parse errors: {playback.parse_error_count}</p>
      </div>
    </section>
  );
}
