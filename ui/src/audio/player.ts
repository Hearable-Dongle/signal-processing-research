const HEADER_BYTES = 16;

export class RealtimeAudioPlayer {
  private ctx: AudioContext | null = null;
  private nextTime = 0;
  private targetJitterSeconds = 0.12;

  async start(): Promise<void> {
    if (this.ctx) {
      return;
    }
    this.ctx = new AudioContext({ sampleRate: 16000 });
    await this.ctx.resume();
    this.nextTime = this.ctx.currentTime + this.targetJitterSeconds;
  }

  stop(): void {
    if (!this.ctx) {
      return;
    }
    void this.ctx.close();
    this.ctx = null;
    this.nextTime = 0;
  }

  pushPacket(payload: ArrayBuffer): void {
    if (!this.ctx || payload.byteLength <= HEADER_BYTES) {
      return;
    }
    const sampleBytes = payload.slice(HEADER_BYTES);
    const samples = new Float32Array(sampleBytes);
    if (!samples.length) {
      return;
    }

    const buffer = this.ctx.createBuffer(1, samples.length, 16000);
    buffer.copyToChannel(samples, 0);

    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.ctx.destination);

    const now = this.ctx.currentTime;
    if (this.nextTime < now + 0.1) {
      this.nextTime = now + 0.1;
    }
    src.start(this.nextTime);
    this.nextTime += buffer.duration;

    const maxAhead = 0.2;
    if (this.nextTime > now + maxAhead) {
      this.nextTime = now + maxAhead;
    }
  }
}
