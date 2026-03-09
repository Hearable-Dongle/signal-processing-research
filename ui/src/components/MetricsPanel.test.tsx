import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";

import { MetricsPanel } from "./MetricsPanel";

function Harness() {
  const [expanded, setExpanded] = useState(true);

  return (
    <MetricsPanel
      metrics={{
        schema_version: "v1",
        type: "metrics",
        timestamp_ms: 0,
        fast_rtf: 0.22,
        slow_rtf: 0.41,
        fast_stage_avg_ms: {},
        slow_stage_avg_ms: {},
        startup_lock_ms: 120,
        reacquire_catchup_ms_median: 34,
        nearest_change_catchup_ms_median: 18,
      }}
      playback={{
        play_state: "playing",
        buffered_ms: 120,
        queued_packet_count: 4,
        dropped_packet_count: 1,
        late_packet_count: 0,
        reanchor_count: 0,
        rebuffer_count: 0,
        startup_gate_wait_ms: 12,
        parse_error_count: 0,
      }}
      expanded={expanded}
      onToggleExpanded={() => setExpanded((prev) => !prev)}
    />
  );
}

test("metrics panel is expanded by default and can collapse", async () => {
  const user = userEvent.setup();

  render(<Harness />);

  const panel = screen.getByRole("button", { name: "Metrics Hide" }).closest("section");
  expect(panel).toHaveClass("expanded");
  expect(screen.getByTestId("metrics-body")).toHaveClass("expanded");
  expect(screen.getByText("Fast RTF: 0.220")).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Metrics Hide" }));
  expect(screen.getByRole("button", { name: "Metrics Show" })).toHaveAttribute("aria-expanded", "false");
  expect(panel).toHaveClass("collapsed");
  expect(screen.getByTestId("metrics-body")).toHaveClass("collapsed");

  await user.click(screen.getByRole("button", { name: "Metrics Show" }));
  expect(screen.getByRole("button", { name: "Metrics Hide" })).toHaveAttribute("aria-expanded", "true");
  expect(panel).toHaveClass("expanded");
  expect(screen.getByTestId("metrics-body")).toHaveClass("expanded");
  expect(screen.getByText("Fast RTF: 0.220")).toBeInTheDocument();
});
