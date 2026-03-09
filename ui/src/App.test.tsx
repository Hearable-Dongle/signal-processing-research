import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import App from "./App";

type MockMessageEvent = { data: string | ArrayBuffer };

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static OPEN = 1;
  readyState = MockWebSocket.OPEN;
  onmessage: ((event: MockMessageEvent) => void) | null = null;
  onclose: (() => void) | null = null;
  sent: string[] = [];
  binaryType = "blob";

  constructor(_url: string) {
    MockWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.onclose?.();
  }
}

class MockAudioContext {
  currentTime = 0;
  destination = {};

  async resume() {
    return;
  }

  createBuffer(_channels: number, length: number, _sampleRate: number) {
    return {
      duration: length / 16000,
      copyToChannel: () => undefined,
    };
  }

  createBufferSource() {
    return {
      buffer: null,
      connect: () => undefined,
      start: () => undefined,
    };
  }

  async close() {
    return;
  }
}

beforeEach(() => {
  MockWebSocket.instances = [];
});

test("speaker interaction emits select and adjust messages", async () => {
  const user = userEvent.setup();
  (globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket;
  const closeSpy = vi.spyOn(MockAudioContext.prototype, "close");
  (globalThis as unknown as { AudioContext: typeof AudioContext }).AudioContext = MockAudioContext as unknown as typeof AudioContext;
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ session_id: "abc123" }),
  } as Response);

  render(<App />);

  await user.click(screen.getByRole("button", { name: "Simulation Scene file plus optional background noise." }));
  await user.click(screen.getByRole("button", { name: "Start" }));
  await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));

  const ws = MockWebSocket.instances[0];
  ws.onmessage?.({
    data: JSON.stringify({
      schema_version: "v1",
      type: "speaker_state",
      timestamp_ms: 0,
      speakers: [
        {
          speaker_id: 9,
          direction_degrees: 20,
          confidence: 0.8,
          active: true,
          activity_confidence: 0.7,
          gain_weight: 1.1,
        },
      ],
    }),
  });

  await user.click(await screen.findByTestId("speaker-9"));
  await user.click(screen.getByRole("button", { name: "+" }));

  const messages = ws.sent.map((s) => JSON.parse(s));
  expect(messages.some((m) => m.type === "select_speaker" && m.speaker_id === 9)).toBe(true);
  expect(messages.some((m) => m.type === "adjust_speaker_gain" && m.speaker_id === 9 && m.delta_db_step === 1)).toBe(true);

  await user.click(screen.getByRole("button", { name: "Kill Current Run" }));
  expect(closeSpy).toHaveBeenCalled();
  expect((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.some((c) => String(c[0]).includes("/stop"))).toBe(true);
});

test("live mode start sends the ReSpeaker session config", async () => {
  const user = userEvent.setup();
  (globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket;
  (globalThis as unknown as { AudioContext: typeof AudioContext }).AudioContext = MockAudioContext as unknown as typeof AudioContext;
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ session_id: "live123" }),
  } as Response);

  render(<App />);

  await user.click(screen.getByRole("button", { name: "ReSpeaker Live Direct capture from the local USB microphone array." }));
  await user.clear(screen.getByLabelText("Audio device query"));
  await user.type(screen.getByLabelText("Audio device query"), "USB Mic");
  await user.click(screen.getByRole("button", { name: "Start" }));

  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
  const startCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.find((c) =>
    String(c[0]).includes("/api/session/start")
  );
  expect(startCall).toBeTruthy();
  const body = JSON.parse(String((startCall?.[1] as RequestInit | undefined)?.body ?? "{}"));
  expect(body.input_source).toBe("respeaker_live");
  expect(body.audio_device_query).toBe("USB Mic");
  expect(body.sample_rate_hz).toBe(48000);
  expect(body.channel_map).toEqual([0, 1, 2, 3]);
});

test("data collection exports raw channels for a captured set", async () => {
  const user = userEvent.setup();
  Object.defineProperty(URL, "createObjectURL", {
    value: URL.createObjectURL ?? vi.fn(),
    writable: true,
    configurable: true,
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    value: URL.revokeObjectURL ?? vi.fn(),
    writable: true,
    configurable: true,
  });
  const createObjectUrl = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:test");
  const revokeObjectUrl = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
  const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/api/session/start")) {
      return {
        ok: true,
        json: async () => ({ session_id: "record123" }),
      } as Response;
    }
    if (url.includes("/api/session/record123/stop")) {
      return {
        ok: true,
        json: async () => ({ status: "stopped" }),
      } as Response;
    }
    if (url.includes("/api/session/record123/raw-channels")) {
      return {
        ok: true,
        json: async () => ({
          session_id: "record123",
          sample_rate_hz: 16000,
          channel_count: 2,
          channels: [
            { channel_index: 0, filename: "channel_000.wav" },
            { channel_index: 1, filename: "channel_001.wav" },
          ],
        }),
      } as Response;
    }
    if (url.includes("/api/session/record123/raw-channel/0.wav")) {
      return {
        ok: true,
        arrayBuffer: async () => new Uint8Array([82, 73, 70, 70]).buffer,
      } as Response;
    }
    if (url.includes("/api/session/record123/raw-channel/1.wav")) {
      return {
        ok: true,
        arrayBuffer: async () => new Uint8Array([87, 65, 86, 69]).buffer,
      } as Response;
    }
    throw new Error(`Unexpected fetch ${url} ${String(init?.method ?? "GET")}`);
  }) as ReturnType<typeof vi.fn>;

  render(<App />);

  await user.click(screen.getByRole("button", { name: "Data Collection" }));
  expect(screen.getByLabelText("Collection id")).toBeInTheDocument();
  expect(screen.getByLabelText("Title")).toBeInTheDocument();
  expect(screen.getByLabelText("Notes")).toBeInTheDocument();
  expect(screen.getByLabelText("Device")).toHaveValue("ReSpeaker");
  await user.click(screen.getByRole("button", { name: "Record" }));
  await user.click(await screen.findByRole("button", { name: "Stop" }));

  expect(await screen.findByText(/Saved recording with 2 raw channels/i)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Export Set" }));

  expect(createObjectUrl).toHaveBeenCalled();
  expect(anchorClick).toHaveBeenCalled();
  expect(revokeObjectUrl).toHaveBeenCalled();
});
