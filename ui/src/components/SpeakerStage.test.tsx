import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";

import { SpeakerControlPopover } from "./SpeakerControlPopover";
import { SpeakerStage } from "./SpeakerStage";

function Harness() {
  const [selected, setSelected] = useState<number | null>(null);
  return (
    <>
      <SpeakerStage
        speakers={[
          {
            speaker_id: 4,
            direction_degrees: 30,
            confidence: 0.8,
            active: true,
            activity_confidence: 0.7,
            gain_weight: 1.0,
          },
        ]}
        beamforming={{
          timestamp_ms: 0,
          mode: "mvdr_fd",
          steering_direction_deg: 30,
          microphone_weights: [{ mic_index: 0, magnitude: 0.25, phase_degrees: 10, delay_samples: null }],
        }}
        groundTruth={[{ source_id: 0, direction_degrees: 30 }]}
        processingMode="specific_speaker_enhancement"
        selectedSpeakerId={selected}
        onSpeakerTap={(id) => setSelected(id)}
      />
      {selected !== null && <SpeakerControlPopover speakerId={selected} deltaDb={0} onAdjust={() => undefined} />}
    </>
  );
}

test("speaker tap opens popover", async () => {
  const user = userEvent.setup();
  render(<Harness />);

  expect(screen.getByTestId("ground-truth-stage")).toBeInTheDocument();
  expect(screen.getByTestId("ground-truth-0")).toBeInTheDocument();
  expect(screen.getByTestId("beamformer-viz")).toBeInTheDocument();
  expect(screen.getByTestId("beam-weight-4")).toHaveTextContent("Weight 1.00");
  expect(screen.getByTestId("beam-mic-0")).toHaveTextContent("|w| 0.250");

  await user.click(screen.getByTestId("speaker-4"));

  expect(screen.getByRole("dialog", { name: "speaker-4-control" })).toBeInTheDocument();
  expect(screen.getByTestId("beam-node-4")).toBeInTheDocument();
});

test("localize and beamform shows active speakers only without numbered speaker buttons", () => {
  render(
    <SpeakerStage
      speakers={[
        {
          speaker_id: 1,
          direction_degrees: 10,
          confidence: 0.9,
          active: true,
          activity_confidence: 0.9,
          gain_weight: 1.0,
        },
        {
          speaker_id: 2,
          direction_degrees: 200,
          confidence: 0.9,
          active: false,
          activity_confidence: 0.2,
          gain_weight: 0.1,
        },
      ]}
      beamforming={null}
      groundTruth={[{ source_id: 0, direction_degrees: 30 }]}
      processingMode="localize_and_beamform"
      selectedSpeakerId={null}
      onSpeakerTap={() => undefined}
    />
  );

  expect(screen.getByTestId("speaker-stage")).toBeInTheDocument();
  expect(screen.getByTestId("active-speaker-1")).toBeInTheDocument();
  expect(screen.queryByTestId("active-speaker-2")).not.toBeInTheDocument();
  expect(screen.queryByTestId("speaker-1")).not.toBeInTheDocument();
  expect(screen.getByTestId("beam-weight-1")).toBeInTheDocument();
  expect(screen.queryByTestId("beam-weight-2")).not.toBeInTheDocument();
  expect(screen.getByTestId("ground-truth-stage")).toBeInTheDocument();
  expect(screen.getByTestId("ground-truth-0")).toBeInTheDocument();
});

test("specific speaker enhancement renders active speaker fully and inactive speaker translucent", () => {
  render(
    <SpeakerStage
      speakers={[
        {
          speaker_id: 1,
          direction_degrees: 10,
          confidence: 0.9,
          active: true,
          activity_confidence: 0.9,
          gain_weight: 1.0,
        },
        {
          speaker_id: 2,
          direction_degrees: 200,
          confidence: 0.6,
          active: false,
          activity_confidence: 0.2,
          gain_weight: 0.4,
        },
      ]}
      beamforming={null}
      groundTruth={[]}
      processingMode="specific_speaker_enhancement"
      selectedSpeakerId={null}
      onSpeakerTap={() => undefined}
    />
  );

  expect(screen.getByTestId("speaker-1")).toHaveClass("active");
  expect(screen.getByTestId("speaker-2")).toHaveClass("inactive");
});

test("beamform from ground truth shows ground truth only", () => {
  render(
    <SpeakerStage
      speakers={[
        {
          speaker_id: 1,
          direction_degrees: 10,
          confidence: 0.9,
          active: true,
          activity_confidence: 0.9,
          gain_weight: 1.0,
        },
      ]}
      beamforming={null}
      groundTruth={[{ source_id: 0, direction_degrees: 30 }]}
      processingMode="beamform_from_ground_truth"
      selectedSpeakerId={null}
      onSpeakerTap={() => undefined}
    />
  );

  expect(screen.queryByTestId("speaker-stage")).not.toBeInTheDocument();
  expect(screen.queryByTestId("beamformer-viz")).not.toBeInTheDocument();
  expect(screen.getByTestId("ground-truth-stage")).toBeInTheDocument();
  expect(screen.getByTestId("ground-truth-0")).toBeInTheDocument();
});
