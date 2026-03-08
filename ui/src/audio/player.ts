import { PlaybackQueueConsumer, type ConsumerStats } from "./consumer";

const HEADER_BYTES = 16;
const DEFAULT_SAMPLE_RATE = 16000;
const MAGIC = "RTA1";
const VERSION = 1;

export type PlaybackStats = ConsumerStats & {
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
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  const version = view.getUint8(4);
  if (magic !== MAGIC || version !== VERSION) {
    return null;
  }

  const tsLow = view.getUint32(8, true);
  const tsHigh = view.getUint32(12, true);
  const timestampMs = tsLow + tsHigh * 2 ** 32;

  const sampleBytes = payload.slice(HEADER_BYTES);
  const samples = new Float32Array(sampleBytes);
  if (samples.length === 0) {
    return null;
  }
  return { timestampMs, samples };
}

function clampLatencyMs(ms: number): number {
  return Math.max(80, Math.min(2000, Math.round(ms)));
}

export class RealtimeAudioPlayer {
  private ctx: AudioContext | null = null;
  private consumer = new PlaybackQueueConsumer<AudioBuffer>({ targetLatencyMs: 220 });
  private packetId = 0;
  private parseErrors = 0;
  private drainTimerId: number | null = null;
  private sampleRateHz = DEFAULT_SAMPLE_RATE;

  async start(sampleRateHz: number = DEFAULT_SAMPLE_RATE): Promise<void> {
    if (this.ctx && this.sampleRateHz === sampleRateHz) {
      return;
    }
    if (this.ctx && this.sampleRateHz !== sampleRateHz) {
      this.stop();
    }
    this.sampleRateHz = sampleRateHz;
    this.ctx = new AudioContext({ sampleRate: this.sampleRateHz });
    await this.ctx.resume();
    this.packetId = 0;
    this.parseErrors = 0;
    this.consumer.reset();
    this.startDrainTimer();
  }

  stop(): void {
    this.stopDrainTimer();
    if (!this.ctx) {
      return;
    }
    void this.ctx.close();
    this.ctx = null;
    this.packetId = 0;
    this.parseErrors = 0;
    this.consumer.reset();
  }

  setTargetLatencyMs(targetLatencyMs: number): void {
    this.consumer.setConfig({ targetLatencyMs: clampLatencyMs(targetLatencyMs) });
  }

  pushPacket(payload: ArrayBuffer): void {
    if (!this.ctx) {
      return;
    }

    const parsed = parsePacket(payload);
    if (!parsed) {
      this.parseErrors += 1;
      return;
    }
    const { samples, timestampMs } = parsed;

    const mono = new Float32Array(samples.length);
    mono.set(samples);

    const buffer = this.ctx.createBuffer(1, mono.length, this.sampleRateHz);
    buffer.copyToChannel(mono, 0);

    this.packetId += 1;
    this.consumer.enqueue({
      id: this.packetId,
      timestampMs,
      durationMs: buffer.duration * 1000,
      payload: buffer,
    });
    this.drain();
  }

  getStats(): PlaybackStats {
    return {
      ...this.consumer.getStats(),
      parse_error_count: this.parseErrors,
    };
  }

  getPlaybackPositionMs(): number {
    if (!this.ctx) {
      return 0;
    }
    return this.consumer.getPlaybackPositionMs(this.ctx.currentTime * 1000);
  }

  private startDrainTimer(): void {
    if (this.drainTimerId !== null) {
      return;
    }
    this.drainTimerId = window.setInterval(() => {
      this.drain();
    }, 10);
  }

  private stopDrainTimer(): void {
    if (this.drainTimerId === null) {
      return;
    }
    window.clearInterval(this.drainTimerId);
    this.drainTimerId = null;
  }

  private drain(): void {
    if (!this.ctx) {
      return;
    }
    const scheduled = this.consumer.drain(this.ctx.currentTime * 1000);
    for (const item of scheduled) {
      const src = this.ctx.createBufferSource();
      src.buffer = item.packet.payload;
      src.connect(this.ctx.destination);
      src.start(item.startSeconds);
    }
  }
}
