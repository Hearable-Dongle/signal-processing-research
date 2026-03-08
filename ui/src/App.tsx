import { useMemo, useRef, useState } from "react";

import { DemoWsClient } from "./api/ws";
import { RealtimeAudioPlayer, type PlaybackStats } from "./audio/player";
import { createWavBlobFromFloat32Chunks } from "./audio/wav";
import { MetricsPanel } from "./components/MetricsPanel";
import { SceneLauncher, type SessionLaunchConfig } from "./components/SceneLauncher";
import { SpeakerControlPopover } from "./components/SpeakerControlPopover";
import { SpeakerStage } from "./components/SpeakerStage";
import { WaveformTimeline } from "./components/WaveformTimeline";
import {
  SCHEMA_VERSION,
  type GroundTruthSpeaker,
  type MetricsMessage,
  type MonitorSource,
  type ProcessingMode,
  type ServerMessage,
  type Speaker,
} from "./types/contracts";

const DEFAULT_SCENE = "simulation/simulations/configs/library_scene/library_k1_scene00.json";
const DEFAULT_BACKGROUND_NOISE = "wham_noise/tr/01dc0215_0.22439_01fc0207_-0.22439sp12.wav";
const DEFAULT_BACKGROUND_NOISE_GAIN = 0.15;
const AUDIO_HEADER_BYTES = 16;
const AUDIO_SAMPLE_RATE = 16000;
const DEFAULT_LATENCY_MS = 220;
const WAVEFORM_BINS = 800;
const DEFAULT_PROCESSING_MODE: ProcessingMode = "specific_speaker_enhancement";
type PlaybackSource = "beamformed_output" | "raw_mixed_input";

const DEFAULT_PLAYBACK_STATS: PlaybackStats = {
  play_state: "buffering",
  buffered_ms: 0,
  queued_packet_count: 0,
  dropped_packet_count: 0,
  late_packet_count: 0,
  reanchor_count: 0,
  rebuffer_count: 0,
  startup_gate_wait_ms: 0,
  parse_error_count: 0,
};

function accumulateWaveformBin(samples: Float32Array): number {
  if (!samples.length) {
    return 0;
  }
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const v = Math.abs(samples[i]);
    if (v > peak) {
      peak = v;
    }
  }
  return Math.max(0, Math.min(1, peak));
}

function computeWaveformBinsFromPcm16(samples: Int16Array, targetBins: number): number[] {
  if (!samples.length || targetBins <= 0) {
    return [];
  }
  const bins: number[] = [];
  const binSize = Math.max(1, Math.floor(samples.length / targetBins));
  for (let start = 0; start < samples.length; start += binSize) {
    let peak = 0;
    const end = Math.min(samples.length, start + binSize);
    for (let i = start; i < end; i += 1) {
      const v = Math.abs(samples[i] / 32768);
      if (v > peak) {
        peak = v;
      }
    }
    bins.push(Math.max(0, Math.min(1, peak)));
    if (bins.length >= targetBins) {
      break;
    }
  }
  return bins;
}

function parsePcm16MonoWav(data: ArrayBuffer): { sampleRateHz: number; samples: Int16Array } | null {
  if (data.byteLength < 44) {
    return null;
  }
  const dv = new DataView(data);
  if (dv.getUint32(0, false) !== 0x52494646 || dv.getUint32(8, false) !== 0x57415645) {
    return null;
  }
  const audioFormat = dv.getUint16(20, true);
  const channels = dv.getUint16(22, true);
  const sampleRateHz = dv.getUint32(24, true);
  const bitsPerSample = dv.getUint16(34, true);
  const dataSize = dv.getUint32(40, true);
  if (audioFormat !== 1 || channels !== 1 || bitsPerSample !== 16) {
    return null;
  }
  const payloadOffset = 44;
  const payloadBytes = Math.min(dataSize, Math.max(0, data.byteLength - payloadOffset));
  const samples = new Int16Array(data.slice(payloadOffset, payloadOffset + payloadBytes));
  return { sampleRateHz, samples };
}

export default function App() {
  const [status, setStatus] = useState("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [groundTruth, setGroundTruth] = useState<GroundTruthSpeaker[]>([]);
  const [selectedSpeakerId, setSelectedSpeakerId] = useState<number | null>(null);
  const [gainBySpeaker, setGainBySpeaker] = useState<Record<number, number>>({});
  const [metrics, setMetrics] = useState<MetricsMessage | null>(null);
  const [latencyMs, setLatencyMs] = useState(DEFAULT_LATENCY_MS);
  const [playbackStats, setPlaybackStats] = useState<PlaybackStats>(DEFAULT_PLAYBACK_STATS);
  const [waveformBins, setWaveformBins] = useState<number[]>([]);
  const [rawWaveformBins, setRawWaveformBins] = useState<number[]>([]);
  const [playheadMs, setPlayheadMs] = useState(0);
  const [isOutputPlaybackActive, setIsOutputPlaybackActive] = useState(false);
  const [isOutputPlaybackPaused, setIsOutputPlaybackPaused] = useState(false);
  const [processingMode, setProcessingMode] = useState<ProcessingMode>(DEFAULT_PROCESSING_MODE);
  const [monitorSource, setMonitorSource] = useState<MonitorSource>("processed");
  const [activePlaybackSource, setActivePlaybackSource] = useState<PlaybackSource | null>(null);

  const audioRef = useRef(new RealtimeAudioPlayer());
  const capturedAudioRef = useRef<Float32Array[]>([]);
  const totalSamplesRef = useRef(0);
  const rawMixedTotalSamplesRef = useRef(0);
  const outputPlaybackRef = useRef<HTMLAudioElement | null>(null);
  const outputPlaybackUrlRef = useRef<string | null>(null);

  function stopOutputPlayback(): void {
    const player = outputPlaybackRef.current;
    if (player) {
      player.pause();
      outputPlaybackRef.current = null;
    }
    if (outputPlaybackUrlRef.current) {
      URL.revokeObjectURL(outputPlaybackUrlRef.current);
      outputPlaybackUrlRef.current = null;
    }
    setIsOutputPlaybackActive(false);
    setIsOutputPlaybackPaused(false);
    setActivePlaybackSource(null);
  }

  function resetLocalSessionState(nextStatus: string): void {
    ws.close();
    audioRef.current.stop();
    stopOutputPlayback();
    setPlaybackStats(DEFAULT_PLAYBACK_STATS);
    setSelectedSpeakerId(null);
    setGroundTruth([]);
    setPlayheadMs(0);
    setRawWaveformBins([]);
    rawMixedTotalSamplesRef.current = 0;
    setStatus(nextStatus);
  }

  const ws = useMemo(
    () =>
      new DemoWsClient({
        onServerMessage: (msg: ServerMessage) => {
          if (msg.type === "speaker_state") {
            setSpeakers(msg.speakers);
            setGroundTruth(msg.ground_truth ?? []);
          }
          if (msg.type === "metrics") {
            setMetrics(msg);
          }
          if (msg.type === "session_event" && msg.event === "stopped") {
            setStatus("stopped");
          }
        },
        onAudioChunk: (chunk: ArrayBuffer) => {
          if (chunk.byteLength > AUDIO_HEADER_BYTES) {
            const payload = chunk.slice(AUDIO_HEADER_BYTES);
            const copy = new Float32Array(payload.slice(0));
            if (copy.length > 0) {
              capturedAudioRef.current.push(copy);
              totalSamplesRef.current += copy.length;
              const amp = accumulateWaveformBin(copy);
              setWaveformBins((prev) => {
                const next = prev.length >= WAVEFORM_BINS ? prev.slice(prev.length - WAVEFORM_BINS + 1) : [...prev];
                next.push(amp);
                return next;
              });
            }
          }
          audioRef.current.pushPacket(chunk);
          setPlaybackStats(audioRef.current.getStats());
          setPlayheadMs(audioRef.current.getPlaybackPositionMs());
        },
        onClose: () => setStatus((s) => (s === "running" ? "disconnected" : s)),
      }),
    []
  );

  async function startSession(config: SessionLaunchConfig): Promise<void> {
    const {
      inputSource,
      scenePath,
      backgroundNoisePath,
      backgroundNoiseGain,
      audioDeviceQuery,
      monitorSource: nextMonitorSource,
    } = config;
    setStatus("starting");
    setMonitorSource(nextMonitorSource);
    capturedAudioRef.current = [];
    totalSamplesRef.current = 0;
    rawMixedTotalSamplesRef.current = 0;
    setWaveformBins([]);
    setRawWaveformBins([]);
    setPlayheadMs(0);
    const resp = await fetch("http://localhost:8000/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input_source: inputSource,
        scene_config_path: scenePath,
        separation_mode: "mock",
        processing_mode: processingMode,
        monitor_source: nextMonitorSource,
        background_noise_audio_path: backgroundNoisePath,
        background_noise_gain: backgroundNoiseGain,
        audio_device_query: inputSource === "respeaker_live" ? audioDeviceQuery : undefined,
      }),
    });
    if (!resp.ok) {
      setStatus(`error (${resp.status})`);
      return;
    }
    const payload = (await resp.json()) as { session_id: string };
    setSessionId(payload.session_id);
    void (async () => {
      try {
        for (let i = 0; i < 30; i += 1) {
          const rawResp = await fetch(`http://localhost:8000/api/session/${payload.session_id}/raw-mix-wav`);
          if (rawResp.ok && typeof rawResp.arrayBuffer === "function") {
            const parsed = parsePcm16MonoWav(await rawResp.arrayBuffer());
            if (parsed) {
              rawMixedTotalSamplesRef.current = parsed.samples.length;
              setRawWaveformBins(computeWaveformBinsFromPcm16(parsed.samples, WAVEFORM_BINS));
            }
            break;
          }
          if (rawResp.ok) {
            break;
          }
          await new Promise((r) => setTimeout(r, 100));
        }
      } catch {
        // Raw mixed waveform visualization is best-effort.
      }
    })();
    audioRef.current.setTargetLatencyMs(latencyMs);
    await audioRef.current.start();
    setPlaybackStats(audioRef.current.getStats());
    ws.connect(payload.session_id);
    setStatus("running");
  }

  async function stopSession(): Promise<void> {
    if (sessionId) {
      await fetch(`http://localhost:8000/api/session/${sessionId}/stop`, { method: "POST" });
    }
    resetLocalSessionState("stopped");
  }

  async function killCurrentRun(): Promise<void> {
    setStatus("stopping");
    const sid = sessionId;
    resetLocalSessionState("stopped");
    if (sid) {
      try {
        await fetch(`http://localhost:8000/api/session/${sid}/stop`, { method: "POST" });
      } catch {
        // Local teardown is authoritative for kill; backend stop is best effort.
      }
    }
  }

  function onLatencyMsChange(nextLatencyMs: number): void {
    setLatencyMs(nextLatencyMs);
    audioRef.current.setTargetLatencyMs(nextLatencyMs);
    setPlaybackStats(audioRef.current.getStats());
  }

  function selectSpeaker(speakerId: number): void {
    if (processingMode !== "specific_speaker_enhancement") {
      return;
    }
    setSelectedSpeakerId(speakerId);
    ws.send({ schema_version: SCHEMA_VERSION, type: "select_speaker", speaker_id: speakerId });
  }

  function adjustSpeakerGain(speakerId: number, step: 1 | -1): void {
    if (processingMode !== "specific_speaker_enhancement") {
      return;
    }
    setGainBySpeaker((prev) => {
      const current = prev[speakerId] ?? 0;
      const next = Math.max(-12, Math.min(12, current + step));
      return { ...prev, [speakerId]: next };
    });
    ws.send({
      schema_version: SCHEMA_VERSION,
      type: "adjust_speaker_gain",
      speaker_id: speakerId,
      delta_db_step: step,
    });
  }

  function downloadWav(): void {
    if (!capturedAudioRef.current.length) {
      return;
    }
    const blob = createWavBlobFromFloat32Chunks(capturedAudioRef.current, AUDIO_SAMPLE_RATE);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const suffix = sessionId ? sessionId : "session";
    a.href = url;
    a.download = `realtime-output-${suffix}.wav`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function onProcessingModeChange(nextMode: ProcessingMode): void {
    setProcessingMode(nextMode);
    if (nextMode !== "specific_speaker_enhancement") {
      setSelectedSpeakerId(null);
    }
  }

  function onMonitorSourceChange(nextSource: MonitorSource): void {
    setMonitorSource(nextSource);
    if (status === "running") {
      ws.send({
        schema_version: SCHEMA_VERSION,
        type: "set_monitor_source",
        monitor_source: nextSource,
      });
    }
  }

  async function toggleOutputPlayback(source: PlaybackSource): Promise<void> {
    const player = outputPlaybackRef.current;
    if (activePlaybackSource === source && player) {
      try {
        await player.play();
        setIsOutputPlaybackActive(true);
        setIsOutputPlaybackPaused(false);
      } catch {
        stopOutputPlayback();
      }
      return;
    }
    stopOutputPlayback();
    let blob: Blob;
    if (source === "beamformed_output") {
      if (!capturedAudioRef.current.length) {
        return;
      }
      blob = createWavBlobFromFloat32Chunks(capturedAudioRef.current, AUDIO_SAMPLE_RATE);
    } else {
      if (!sessionId) {
        return;
      }
      const resp = await fetch(`http://localhost:8000/api/session/${sessionId}/raw-mix-wav`);
      if (!resp.ok) {
        return;
      }
      blob = await resp.blob();
    }
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    outputPlaybackUrlRef.current = url;
    outputPlaybackRef.current = audio;
    audio.onended = () => {
      stopOutputPlayback();
    };
    try {
      await audio.play();
      setIsOutputPlaybackActive(true);
      setIsOutputPlaybackPaused(false);
      setActivePlaybackSource(source);
    } catch {
      stopOutputPlayback();
    }
  }

  function pauseBeamformedPlayback(): void {
    if (activePlaybackSource !== "beamformed_output") {
      return;
    }
    const player = outputPlaybackRef.current;
    if (!player) {
      return;
    }
    player.pause();
    setIsOutputPlaybackActive(false);
    setIsOutputPlaybackPaused(true);
  }

  const totalDurationMs = (totalSamplesRef.current / AUDIO_SAMPLE_RATE) * 1000;
  const rawMixedDurationMs = (rawMixedTotalSamplesRef.current / AUDIO_SAMPLE_RATE) * 1000;
  const canPlayBeamformed = capturedAudioRef.current.length > 0;
  const canPlayRawMixed = Boolean(sessionId);

  return (
    <main className="app-shell">
      <h1>Realtime Speaker UI Demo</h1>
      <div className="layout">
        <SceneLauncher
          status={status}
          defaultScenePath={DEFAULT_SCENE}
          defaultBackgroundNoisePath={DEFAULT_BACKGROUND_NOISE}
          defaultBackgroundNoiseGain={DEFAULT_BACKGROUND_NOISE_GAIN}
          onStart={startSession}
          onStop={stopSession}
          onKillRun={killCurrentRun}
          canKillRun={status === "running" || status === "starting" || status === "stopping"}
          onDownloadWav={downloadWav}
          canDownloadWav={capturedAudioRef.current.length > 0}
          latencyMs={latencyMs}
          onLatencyMsChange={onLatencyMsChange}
          processingMode={processingMode}
          onProcessingModeChange={onProcessingModeChange}
          monitorSource={monitorSource}
          onMonitorSourceChange={onMonitorSourceChange}
        />
        <SpeakerStage
          speakers={speakers}
          groundTruth={groundTruth}
          processingMode={processingMode}
          selectedSpeakerId={processingMode === "specific_speaker_enhancement" ? selectedSpeakerId : null}
          onSpeakerTap={selectSpeaker}
        />
        <MetricsPanel metrics={metrics} playback={playbackStats} />
      </div>

      <WaveformTimeline
        beamformedBins={waveformBins}
        beamformedDurationMs={totalDurationMs}
        rawMixedBins={rawWaveformBins}
        rawMixedDurationMs={rawMixedDurationMs}
        playheadMs={playheadMs}
        canPlayBeamformed={canPlayBeamformed}
        canPauseBeamformed={activePlaybackSource === "beamformed_output" && isOutputPlaybackActive}
        isBeamformedPlaying={isOutputPlaybackActive && activePlaybackSource === "beamformed_output"}
        onPlayBeamformed={() => {
          void toggleOutputPlayback("beamformed_output");
        }}
        onPauseBeamformed={pauseBeamformedPlayback}
        canPlayRawMixed={canPlayRawMixed}
        isRawMixedPlaying={isOutputPlaybackActive && activePlaybackSource === "raw_mixed_input"}
        onToggleRawMixedPlayback={() => {
          void toggleOutputPlayback("raw_mixed_input");
        }}
      />

      {processingMode === "specific_speaker_enhancement" && selectedSpeakerId !== null && (
        <SpeakerControlPopover
          speakerId={selectedSpeakerId}
          deltaDb={gainBySpeaker[selectedSpeakerId] ?? 0}
          onAdjust={adjustSpeakerGain}
        />
      )}
    </main>
  );
}
