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

  await user.click(screen.getByTestId("speaker-4"));

  expect(screen.getByRole("dialog", { name: "speaker-4-control" })).toBeInTheDocument();
});
