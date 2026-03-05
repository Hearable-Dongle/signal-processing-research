export type PlaybackState = "buffering" | "playing";

export type ConsumerConfig = {
  targetLatencyMs: number;
  startThresholdMs: number;
  minBufferMs: number;
  maxBufferMs: number;
  minLeadMs: number;
  rebufferThresholdMs: number;
  lateDropMs: number;
  monotonicEpsilonMs: number;
  rebufferConsecutiveLowCount: number;
};

export type ConsumerPacket<T> = {
  id: number;
  timestampMs: number;
  durationMs: number;
  payload: T;
};

export type ScheduledPacket<T> = {
  packet: ConsumerPacket<T>;
  startSeconds: number;
  endSeconds: number;
};

export type ConsumerStats = {
  play_state: PlaybackState;
  buffered_ms: number;
  queued_packet_count: number;
  dropped_packet_count: number;
  late_packet_count: number;
  reanchor_count: number;
  rebuffer_count: number;
  startup_gate_wait_ms: number;
};

const DEFAULT_TARGET_MS = 220;

function defaultConfig(targetLatencyMs: number): ConsumerConfig {
  const target = Math.max(1, targetLatencyMs);
  return {
    targetLatencyMs: target,
    startThresholdMs: target,
    minBufferMs: Math.max(40, Math.round(target * 0.6)),
    maxBufferMs: Math.max(300, Math.round(target * 1.8)),
    minLeadMs: 10,
    rebufferThresholdMs: 20,
    lateDropMs: Math.max(50, Math.round(target * 1.2)),
    monotonicEpsilonMs: 1,
    rebufferConsecutiveLowCount: 3,
  };
}

export class PlaybackQueueConsumer<T> {
  private cfg: ConsumerConfig;
  private queue: ConsumerPacket<T>[] = [];
  private state: PlaybackState = "buffering";

  private anchorPacketTsMs: number | null = null;
  private anchorPlayoutTimeSec = 0;
  private lastScheduledEndSec = 0;
  private startupBufferingStartMs: number | null = null;
  private lowBufferStreak = 0;

  private stats: ConsumerStats = {
    play_state: "buffering",
    buffered_ms: 0,
    queued_packet_count: 0,
    dropped_packet_count: 0,
    late_packet_count: 0,
    reanchor_count: 0,
    rebuffer_count: 0,
    startup_gate_wait_ms: 0,
  };

  constructor(config?: Partial<ConsumerConfig>) {
    const defaults = defaultConfig(config?.targetLatencyMs ?? DEFAULT_TARGET_MS);
    this.cfg = { ...defaults, ...config };
  }

  reset(): void {
    this.queue = [];
    this.state = "buffering";
    this.anchorPacketTsMs = null;
    this.anchorPlayoutTimeSec = 0;
    this.lastScheduledEndSec = 0;
    this.startupBufferingStartMs = null;
    this.lowBufferStreak = 0;
    this.stats = {
      play_state: "buffering",
      buffered_ms: 0,
      queued_packet_count: 0,
      dropped_packet_count: 0,
      late_packet_count: 0,
      reanchor_count: 0,
      rebuffer_count: 0,
      startup_gate_wait_ms: 0,
    };
  }

  setConfig(partial: Partial<ConsumerConfig>): void {
    const prevTarget = this.cfg.targetLatencyMs;
    const nextTarget = partial.targetLatencyMs ?? prevTarget;

    if (partial.targetLatencyMs !== undefined && partial.startThresholdMs === undefined) {
      partial.startThresholdMs = nextTarget;
    }
    if (partial.targetLatencyMs !== undefined && partial.minBufferMs === undefined) {
      partial.minBufferMs = Math.max(40, Math.round(nextTarget * 0.6));
    }
    if (partial.targetLatencyMs !== undefined && partial.maxBufferMs === undefined) {
      partial.maxBufferMs = Math.max(300, Math.round(nextTarget * 1.8));
    }
    if (partial.targetLatencyMs !== undefined && partial.lateDropMs === undefined) {
      partial.lateDropMs = Math.max(50, Math.round(nextTarget * 1.2));
    }

    this.cfg = {
      ...this.cfg,
      ...partial,
    };
  }

  enqueue(packet: ConsumerPacket<T>): void {
    const item: ConsumerPacket<T> = {
      ...packet,
      durationMs: Math.max(0.1, packet.durationMs),
    };

    let idx = this.queue.findIndex(
      (p) => p.timestampMs > item.timestampMs || (p.timestampMs === item.timestampMs && p.id > item.id)
    );
    if (idx < 0) {
      idx = this.queue.length;
    }
    this.queue.splice(idx, 0, item);
    this.stats.queued_packet_count = this.queue.length;
    if (this.state === "buffering") {
      this.stats.buffered_ms = this.queueSpanMs();
    }
  }

  drain(nowMs: number): ScheduledPacket<T>[] {
    const nowSec = nowMs / 1000;
    this.updateBufferedStats(nowMs);

    if (this.state === "buffering") {
      const span = this.queueSpanMs();
      if (span < this.cfg.startThresholdMs) {
        if (this.startupBufferingStartMs === null) {
          this.startupBufferingStartMs = nowMs;
        }
        this.stats.play_state = "buffering";
        return [];
      }
      this.enterPlaying(nowMs);
    }

    const bufferedAhead = this.computeBufferedAheadMs(nowMs);
    if (bufferedAhead < this.cfg.rebufferThresholdMs) {
      this.lowBufferStreak += 1;
      if (this.lowBufferStreak >= this.cfg.rebufferConsecutiveLowCount) {
        this.state = "buffering";
        this.stats.rebuffer_count += 1;
        this.stats.play_state = "buffering";
        this.lowBufferStreak = 0;
        this.updateBufferedStats(nowMs);
        return [];
      }
    } else {
      this.lowBufferStreak = 0;
    }

    const out: ScheduledPacket<T>[] = [];
    const scheduleCutoffSec = nowSec + this.cfg.maxBufferMs / 1000;

    while (this.queue.length > 0) {
      const head = this.queue[0];
      if (this.anchorPacketTsMs === null) {
        this.enterPlaying(nowMs);
      }
      const anchorTs = this.anchorPacketTsMs ?? head.timestampMs;
      const idealStartSec = this.anchorPlayoutTimeSec + (head.timestampMs - anchorTs) / 1000;
      const lateByMs = (nowSec - idealStartSec) * 1000;

      if (lateByMs > this.cfg.lateDropMs) {
        this.queue.shift();
        this.stats.late_packet_count += 1;
        this.stats.dropped_packet_count += 1;
        continue;
      }

      let startSec = idealStartSec;
      const minStartSec = nowSec + this.cfg.minLeadMs / 1000;
      const monotonicMinSec = this.lastScheduledEndSec + this.cfg.monotonicEpsilonMs / 1000;

      if (startSec < minStartSec) {
        startSec = minStartSec;
        if (lateByMs > 0) {
          this.stats.late_packet_count += 1;
        }
      }
      if (startSec < monotonicMinSec) {
        startSec = monotonicMinSec;
      }

      if (startSec > scheduleCutoffSec) {
        break;
      }

      const endSec = startSec + head.durationMs / 1000;
      this.lastScheduledEndSec = endSec;
      out.push({ packet: head, startSeconds: startSec, endSeconds: endSec });
      this.queue.shift();
    }

    this.updateBufferedStats(nowMs);
    return out;
  }

  getStats(): ConsumerStats {
    return { ...this.stats };
  }

  getPlaybackPositionMs(nowMs: number): number {
    if (this.anchorPacketTsMs === null) {
      return 0;
    }
    const mediaNowMs = this.anchorPacketTsMs + (nowMs / 1000 - this.anchorPlayoutTimeSec) * 1000;
    return Math.max(0, mediaNowMs);
  }

  private enterPlaying(nowMs: number): void {
    if (!this.queue.length) {
      this.state = "buffering";
      this.stats.play_state = "buffering";
      return;
    }
    const head = this.queue[0];
    this.anchorPacketTsMs = head.timestampMs;
    this.anchorPlayoutTimeSec = nowMs / 1000 + this.cfg.targetLatencyMs / 1000;
    this.lastScheduledEndSec = this.anchorPlayoutTimeSec;
    this.state = "playing";
    this.stats.play_state = "playing";
    this.lowBufferStreak = 0;
    if (this.startupBufferingStartMs !== null) {
      this.stats.startup_gate_wait_ms += Math.max(0, nowMs - this.startupBufferingStartMs);
    }
    this.startupBufferingStartMs = null;
    this.stats.reanchor_count += 1;
  }

  private queueSpanMs(): number {
    if (!this.queue.length) {
      return 0;
    }
    const firstTs = this.queue[0].timestampMs;
    const last = this.queue[this.queue.length - 1];
    const lastEnd = last.timestampMs + last.durationMs;
    return Math.max(0, lastEnd - firstTs);
  }

  private computeBufferedAheadMs(nowMs: number): number {
    if (!this.queue.length || this.anchorPacketTsMs === null) {
      return this.queueSpanMs();
    }
    const mediaNowMs = this.anchorPacketTsMs + (nowMs / 1000 - this.anchorPlayoutTimeSec) * 1000;
    const last = this.queue[this.queue.length - 1];
    const lastEnd = last.timestampMs + last.durationMs;
    return Math.max(0, lastEnd - mediaNowMs);
  }

  private updateBufferedStats(nowMs: number): void {
    this.stats.queued_packet_count = this.queue.length;
    this.stats.play_state = this.state;
    this.stats.buffered_ms = this.computeBufferedAheadMs(nowMs);
  }
}
