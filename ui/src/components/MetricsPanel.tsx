import type { MetricsMessage } from "../types/contracts";

type Props = {
  metrics: MetricsMessage | null;
};

export function MetricsPanel({ metrics }: Props) {
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
    </section>
  );
}
