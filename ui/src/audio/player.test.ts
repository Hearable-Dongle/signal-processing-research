import { RealtimeAudioPlayer } from "./player";

let mockNow = 0;
let scheduledStarts: number[] = [];

class MockAudioContext {
  destination = {};

  get currentTime(): number {
    return mockNow;
  }

  async resume(): Promise<void> {
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
      start: (t: number) => {
        scheduledStarts.push(t);
      },
    };
  }

  async close(): Promise<void> {
    return;
  }
}

function makePacket(timestampMs: number, samples = 160): ArrayBuffer {
  const bytes = new ArrayBuffer(16 + samples * 4);
  const view = new DataView(bytes);
  view.setUint8(0, "R".charCodeAt(0));
  view.setUint8(1, "T".charCodeAt(0));
  view.setUint8(2, "A".charCodeAt(0));
  view.setUint8(3, "1".charCodeAt(0));
  view.setUint8(4, 1);
  view.setUint32(8, timestampMs >>> 0, true);
  view.setUint32(12, Math.floor(timestampMs / 2 ** 32), true);
  return bytes;
}

describe("RealtimeAudioPlayer", () => {
  beforeEach(() => {
    mockNow = 0;
    scheduledStarts = [];
    vi.stubGlobal("AudioContext", MockAudioContext as unknown as typeof AudioContext);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("schedules packets by timestamp spacing", async () => {
    const player = new RealtimeAudioPlayer();
    await player.start();

    player.pushPacket(makePacket(0));
    player.pushPacket(makePacket(10));

    expect(scheduledStarts.length).toBe(2);
    expect(scheduledStarts[0]).toBeGreaterThanOrEqual(0.149);
    expect(Math.abs((scheduledStarts[1] - scheduledStarts[0]) - 0.01)).toBeLessThan(0.003);

    const stats = player.getStats();
    expect(stats.reanchor_count).toBe(0);
    expect(stats.parse_error_count).toBe(0);
  });

  test("soft re-anchors when packet arrives too late", async () => {
    const player = new RealtimeAudioPlayer();
    await player.start();

    player.pushPacket(makePacket(0));
    mockNow = 0.35;
    player.pushPacket(makePacket(10));

    expect(scheduledStarts.length).toBe(2);
    expect(scheduledStarts[1]).toBeGreaterThanOrEqual(0.359);

    const stats = player.getStats();
    expect(stats.underrun_count).toBeGreaterThan(0);
    expect(stats.reanchor_count).toBeGreaterThan(0);
  });

  test("counts malformed packets as parse errors", async () => {
    const player = new RealtimeAudioPlayer();
    await player.start();

    const bad = new ArrayBuffer(20);
    player.pushPacket(bad);

    expect(player.getStats().parse_error_count).toBe(1);
  });
});
