const HEADER_BYTES = 16;
const MAGIC = "RTA1";
const VERSION = 1;
const SAMPLE_RATE = 16000;

export type PlaybackStats = {
  buffered_ms: number;
  drift_ms: number;
  underrun_count: number;
  reanchor_count: number;
  parse_error_count: number;
};

type ParsedPacket = {
  timestampMs: number;
  samples: Float32Array;
};

function parsePacket(payload: ArrayBuffer): ParsedPacket | null {
  if (payload.byteLength <= HEADER_BYTES) {
    return null;
  }
  const view = new DataView(payload);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3)
  );
  const version = view.getUint8(4);
  if (magic !== MAGIC || version !== VERSION) {
    return null;
  }
  const tsLow = view.getUint32(8, true);
  const tsHigh = view.getUint32(12, true);
  const timestampMs = tsLow + tsHigh * 2 ** 32;
  const sampleBytes = payload.slice(HEADER_BYTES);
  const samples = new Float32Array(sampleBytes);
  if (!samples.length) {
    return null;
  }
  return { timestampMs, samples };
}

export class RealtimeAudioPlayer {
  private ctx: AudioContext | null = null;
  private basePacketTsMs: number | null = null;
  private baseAudioTime = 0;
  private lastScheduledEnd = 0;
  private readonly initialDelaySeconds = 0.15;
  private readonly lateThresholdSeconds = 0.04;
  private readonly minLeadSeconds = 0.01;
  private readonly monotonicEpsilonSeconds = 0.001;
  private stats: PlaybackStats = {
    buffered_ms: 0,
    drift_ms: 0,
    underrun_count: 0,
    reanchor_count: 0,
    parse_error_count: 0,
  };

  async start(): Promise<void> {
    if (this.ctx) {
      return;
    }
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    await this.ctx.resume();
    this.basePacketTsMs = null;
    this.baseAudioTime = this.ctx.currentTime + this.initialDelaySeconds;
    this.lastScheduledEnd = this.baseAudioTime;
    this.stats = {
      buffered_ms: this.initialDelaySeconds * 1000,
      drift_ms: 0,
      underrun_count: 0,
      reanchor_count: 0,
      parse_error_count: 0,
    };
  }

  stop(): void {
    if (!this.ctx) {
      return;
    }
    void this.ctx.close();
    this.ctx = null;
    this.basePacketTsMs = null;
    this.baseAudioTime = 0;
    this.lastScheduledEnd = 0;
  }

  getStats(): PlaybackStats {
    return { ...this.stats };
  }

  pushPacket(payload: ArrayBuffer): void {
    if (!this.ctx) {
      return;
    }

    const parsed = parsePacket(payload);
    if (!parsed) {
      this.stats.parse_error_count += 1;
      return;
    }
    const { timestampMs, samples } = parsed;
    if (this.basePacketTsMs === null) {
      this.basePacketTsMs = timestampMs;
      this.baseAudioTime = this.ctx.currentTime + this.initialDelaySeconds;
      this.lastScheduledEnd = this.baseAudioTime;
    }

    const relativeSeconds = (timestampMs - this.basePacketTsMs) / 1000;
    const idealStart = this.baseAudioTime + relativeSeconds;
    const now = this.ctx.currentTime;
    const lateBy = now - idealStart;
    this.stats.drift_ms = lateBy * 1000;

    let scheduledStart = idealStart;
    if (lateBy > 0) {
      this.stats.underrun_count += 1;
    }
    if (lateBy > this.lateThresholdSeconds) {
      this.baseAudioTime += lateBy;
      scheduledStart = this.baseAudioTime + relativeSeconds;
      this.stats.reanchor_count += 1;
    }

    scheduledStart = Math.max(scheduledStart, now + this.minLeadSeconds);
    if (scheduledStart < this.lastScheduledEnd + this.monotonicEpsilonSeconds) {
      scheduledStart = this.lastScheduledEnd + this.monotonicEpsilonSeconds;
    }

    const mono = new Float32Array(samples.length);
    mono.set(samples);
    const buffer = this.ctx.createBuffer(1, mono.length, SAMPLE_RATE);
    buffer.copyToChannel(mono, 0);

    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.ctx.destination);
    src.start(scheduledStart);

    this.lastScheduledEnd = scheduledStart + buffer.duration;
    this.stats.buffered_ms = Math.max(0, (this.lastScheduledEnd - now) * 1000);
  }
}
