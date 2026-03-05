import { PlaybackQueueConsumer, type ConsumerPacket } from "./consumer";

type Dummy = { token: string };

function pkt(id: number, timestampMs: number, durationMs = 10): ConsumerPacket<Dummy> {
  return {
    id,
    timestampMs,
    durationMs,
    payload: { token: `p-${id}` },
  };
}

describe("PlaybackQueueConsumer", () => {
  test("startup gate blocks until threshold is filled", () => {
    const c = new PlaybackQueueConsumer<Dummy>({ targetLatencyMs: 120, startThresholdMs: 120 });
    c.enqueue(pkt(1, 0));
    c.enqueue(pkt(2, 10));
    c.enqueue(pkt(3, 20));

    const before = c.drain(0);
    expect(before).toHaveLength(0);
    expect(c.getStats().play_state).toBe("buffering");

    for (let i = 3; i < 12; i += 1) {
      c.enqueue(pkt(i + 1, i * 10));
    }

    const after = c.drain(5);
    expect(after.length).toBeGreaterThan(0);
    expect(c.getStats().play_state).toBe("playing");
  });

  test("out-of-order arrival is drained in timestamp order", () => {
    const c = new PlaybackQueueConsumer<Dummy>({ targetLatencyMs: 60, startThresholdMs: 30 });
    c.enqueue(pkt(1, 20));
    c.enqueue(pkt(2, 0));
    c.enqueue(pkt(3, 10));

    const out = c.drain(0);
    const orderedTs = out.map((x) => x.packet.timestampMs);
    expect(orderedTs).toEqual([0, 10, 20]);
  });

  test("late packets are dropped when outside lateness budget", () => {
    const c = new PlaybackQueueConsumer<Dummy>({
      targetLatencyMs: 100,
      startThresholdMs: 20,
      lateDropMs: 50,
      rebufferThresholdMs: 0,
    });

    c.enqueue(pkt(1, 0));
    c.enqueue(pkt(2, 10));
    c.drain(0);

    c.enqueue(pkt(3, 20));
    c.enqueue(pkt(4, 30));
    c.drain(500);

    const stats = c.getStats();
    expect(stats.dropped_packet_count).toBeGreaterThan(0);
    expect(stats.late_packet_count).toBeGreaterThan(0);
  });

  test("rebuffer triggers on starvation and resumes after refill", () => {
    const c = new PlaybackQueueConsumer<Dummy>({
      targetLatencyMs: 120,
      startThresholdMs: 120,
      rebufferThresholdMs: 25,
    });

    for (let i = 0; i < 14; i += 1) {
      c.enqueue(pkt(i + 1, i * 10));
    }
    const started = c.drain(0);
    expect(started.length).toBeGreaterThan(0);

    c.drain(2000);
    c.drain(2010);
    const stalled = c.drain(2020);
    expect(stalled).toHaveLength(0);
    expect(c.getStats().play_state).toBe("buffering");
    expect(c.getStats().rebuffer_count).toBeGreaterThan(0);

    for (let i = 0; i < 16; i += 1) {
      c.enqueue(pkt(100 + i, 3000 + i * 10));
    }
    const resumed = c.drain(2010);
    expect(resumed.length).toBeGreaterThan(0);
    expect(c.getStats().play_state).toBe("playing");
  });

  test("latency config update increases startup gate threshold", () => {
    const c = new PlaybackQueueConsumer<Dummy>({ targetLatencyMs: 100 });
    c.setConfig({ targetLatencyMs: 240 });

    for (let i = 0; i < 20; i += 1) {
      c.enqueue(pkt(i + 1, i * 10));
    }
    const out = c.drain(0);
    // 20 packets * 10ms = 200ms, below updated default startThreshold(240)
    expect(out).toHaveLength(0);
    expect(c.getStats().play_state).toBe("buffering");

    for (let i = 20; i < 26; i += 1) {
      c.enqueue(pkt(i + 1, i * 10));
    }
    const out2 = c.drain(5);
    expect(out2.length).toBeGreaterThan(0);
    expect(c.getStats().play_state).toBe("playing");
  });

  test("wan-like jitter profile stays controlled", () => {
    const c = new PlaybackQueueConsumer<Dummy>({
      targetLatencyMs: 220,
      startThresholdMs: 220,
      rebufferThresholdMs: 30,
    });

    let wallMs = 0;
    for (let i = 0; i < 120; i += 1) {
      const mediaTs = i * 10;
      const jitter = i % 17 === 0 ? 180 : i % 9 === 0 ? 40 : 5;
      wallMs += jitter;
      c.enqueue(pkt(i + 1, mediaTs));
      c.drain(wallMs);
    }

    const stats = c.getStats();
    expect(stats.rebuffer_count).toBeLessThan(8);
    expect(stats.dropped_packet_count).toBeLessThan(40);
  });
});
