import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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
  expect(screen.getByLabelText("Localization backend")).toHaveValue("capon_1src");
  await user.click(screen.getByRole("button", { name: "Start" }));
  await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));

  const startCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.find((c) =>
    String(c[0]).includes("/api/session/start")
  );
  expect(startCall).toBeTruthy();
  const startBody = JSON.parse(String((startCall?.[1] as RequestInit | undefined)?.body ?? "{}"));
  expect(startBody.processing_mode).toBe("specific_speaker_enhancement");
  expect(startBody.fast_path.localization_hop_ms).toBe(95);
  expect(startBody.fast_path.localization_window_ms).toBe(300);
  expect(startBody.fast_path.overlap).toBe(0.2);
  expect(startBody.fast_path.freq_low_hz).toBe(1200);
  expect(startBody.fast_path.freq_high_hz).toBe(5400);
  expect(startBody.slow_path.speaker_history_size).toBe(8);
  expect(startBody.slow_path.speaker_activation_min_predictions).toBe(3);
  expect(startBody.slow_path.speaker_match_window_deg).toBe(30);

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
  await user.click(screen.getByRole("switch", { name: "Enable slow path" }));
  await user.click(screen.getByRole("switch", { name: "Single active speaker" }));
  await user.selectOptions(screen.getByLabelText("Mic array profile"), "respeaker_xvf3800_0650");
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
  expect(body.mic_array_profile).toBe("respeaker_xvf3800_0650");
  expect(body.audio_device_query).toBe("USB Mic");
  expect(body.sample_rate_hz).toBe(48000);
  expect(body.channel_map).toBeUndefined();
  expect(body.fast_path.localization_hop_ms).toBe(95);
  expect(body.fast_path.localization_window_ms).toBe(300);
  expect(body.fast_path.overlap).toBe(0.2);
  expect(body.fast_path.freq_low_hz).toBe(1200);
  expect(body.fast_path.freq_high_hz).toBe(5400);
  expect(body.slow_path.enabled).toBe(true);
  expect(body.slow_path.single_active).toBe(false);
  expect(body.slow_path.speaker_history_size).toBe(8);
  expect(body.slow_path.speaker_activation_min_predictions).toBe(3);
  expect(body.slow_path.speaker_match_window_deg).toBe(30);
});

test("simulation start sends algorithm mode plus ground-truth toggles", async () => {
  const user = userEvent.setup();
  (globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket;
  (globalThis as unknown as { AudioContext: typeof AudioContext }).AudioContext = MockAudioContext as unknown as typeof AudioContext;
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ session_id: "beam123" }),
  } as Response);

  render(<App />);

  await user.click(screen.getByRole("button", { name: "Simulation Scene file plus optional background noise." }));
  fireEvent.change(screen.getByLabelText("Localization hop (ms)"), { target: { value: "50" } });
  fireEvent.change(screen.getByLabelText("Localization window (ms)"), { target: { value: "200" } });
  await user.click(screen.getByRole("switch", { name: "Enable slow path" }));
  await user.click(screen.getByRole("switch", { name: "Long memory" }));
  await user.click(screen.getByLabelText("Use ground truth location"));
  await user.click(screen.getByLabelText("Use ground truth speaker sources"));
  await user.click(screen.getByRole("button", { name: "Start" }));

  await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
  const startCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.find((c) =>
    String(c[0]).includes("/api/session/start")
  );
  const body = JSON.parse(String((startCall?.[1] as RequestInit | undefined)?.body ?? "{}"));
  expect(body.fast_path.localization_hop_ms).toBe(50);
  expect(body.fast_path.localization_window_ms).toBe(200);
  expect(body.slow_path.enabled).toBe(true);
  expect(body.slow_path.long_memory_enabled).toBe(true);
  expect(body.use_ground_truth_location).toBe(true);
  expect(body.use_ground_truth_speaker_sources).toBe(true);
  expect(body.processing_mode).toBe("specific_speaker_enhancement");
});

test("data collection exports raw channels for a captured set", async () => {
  const user = userEvent.setup();
  (globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket;
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
  expect(screen.getByLabelText("Device")).toHaveValue("XVF3800");
  expect(screen.getByLabelText("Mic array profile")).toHaveValue("respeaker_xvf3800_0650");
  await user.type(screen.getByLabelText("Recording notes"), "speaker near whiteboard");
  await user.clear(screen.getByLabelText("Pending speaker 1 name"));
  await user.type(screen.getByLabelText("Pending speaker 1 name"), "amber-otter");
  await user.clear(screen.getByLabelText("Pending speaker 1 period 1 DOA"));
  await user.type(screen.getByLabelText("Pending speaker 1 period 1 DOA"), "45");
  await user.click(screen.getByRole("button", { name: "Record" }));
  await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));

  MockWebSocket.instances[0]?.onmessage?.({
    data: JSON.stringify({
      schema_version: "v1",
      type: "speaker_state",
      timestamp_ms: 0,
      speakers: [
        {
          speaker_id: 7,
          direction_degrees: 0,
          confidence: 0.8,
          active: true,
          activity_confidence: 0.7,
          gain_weight: 1.0,
        },
      ],
      ground_truth: [],
    }),
  });

  expect(await screen.findByTestId("speaker-stage")).toBeInTheDocument();
  expect(screen.getByTestId("directionality-viz")).toBeInTheDocument();
  expect(screen.getByTestId("tracked-cable")).toHaveTextContent("cable");
  await user.click(await screen.findByRole("button", { name: "Stop" }));

  const startCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.find((c) =>
    String(c[0]).includes("/api/session/start")
  );
  const body = JSON.parse(String((startCall?.[1] as RequestInit | undefined)?.body ?? "{}"));
  expect(body.mic_array_profile).toBe("respeaker_xvf3800_0650");
  expect(body.channel_map).toBeUndefined();
  expect(body.monitor_source).toBe("raw_mixed");

  expect(await screen.findByText(/Saved recording with 2 raw channels/i)).toBeInTheDocument();
  expect(screen.getByLabelText(/Recording notes recording-/i)).toHaveValue("speaker near whiteboard");
  expect(screen.getByLabelText(/Speaker 1 name for recording-/i)).toHaveValue("amber-otter");
  expect(screen.getByLabelText(/Speaker 1 period 1 DOA for recording-/i)).toHaveValue(45);
  expect(screen.getByLabelText(/Speaker 1 period 1 start for recording-/i)).toHaveValue(0);
  expect(screen.getByText("raw ch0 · mic 1")).toBeInTheDocument();
  expect(screen.getByText("raw ch1 · mic 2")).toBeInTheDocument();
  expect(document.querySelectorAll("audio").length).toBe(2);
  await user.click(screen.getByRole("button", { name: "Export Set" }));

  expect(createObjectUrl).toHaveBeenCalled();
  expect(anchorClick).toHaveBeenCalled();
  expect(revokeObjectUrl).toHaveBeenCalled();
});
