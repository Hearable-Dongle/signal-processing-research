import { useMemo, useRef, useState } from "react";

import { DemoWsClient } from "../api/ws";
import { RealtimeAudioPlayer, type PlaybackStats } from "../audio/player";
import { createWavBlobFromFloat32Chunks } from "../audio/wav";
import { MetricsPanel } from "./MetricsPanel";
import { SceneLauncher, type SessionLaunchConfig } from "./SceneLauncher";
import { SpeakerControlPopover } from "./SpeakerControlPopover";
import { SpeakerStage } from "./SpeakerStage";
import { WaveformTimeline } from "./WaveformTimeline";
import {
  SCHEMA_VERSION,
  type AlgorithmMode,
  type GroundTruthSpeaker,
  type MetricsMessage,
  type MonitorSource,
  type ServerMessage,
  type Speaker,
} from "../types/contracts";

const AUDIO_HEADER_BYTES = 16;
const DEFAULT_SAMPLE_RATE = 16000;
const DEFAULT_LATENCY_MS = 220;
const WAVEFORM_BINS = 800;
type PlaybackSource = "beamformed_output" | "raw_mixed_input";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

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

function apiUrl(path: string): string {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

async function fetchSessionWav(path: string): Promise<Blob | null> {
  try {
    const resp = await fetch(apiUrl(path));
    if (!resp.ok) {
      return null;
    }
    return await resp.blob();
  } catch {
    return null;
  }
}

type Props = {
  defaultScenePath: string;
  defaultBackgroundNoisePath: string;
  defaultBackgroundNoiseGain: number;
  defaultAlgorithmMode: AlgorithmMode;
};

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

export function RealtimeDemoPage({
  defaultScenePath,
  defaultBackgroundNoisePath,
  defaultBackgroundNoiseGain,
  defaultAlgorithmMode,
}: Props) {
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
  const [metricsExpanded, setMetricsExpanded] = useState(true);
  const [isOutputPlaybackActive, setIsOutputPlaybackActive] = useState(false);
  const [isOutputPlaybackPaused, setIsOutputPlaybackPaused] = useState(false);
  const [algorithmMode, setAlgorithmMode] = useState<AlgorithmMode>(defaultAlgorithmMode);
  const [monitorSource, setMonitorSource] = useState<MonitorSource>("processed");
  const [activePlaybackSource, setActivePlaybackSource] = useState<PlaybackSource | null>(null);
  const [activeInputSource, setActiveInputSource] = useState<SessionLaunchConfig["inputSource"]>("simulation");
  const [audioSampleRateHz, setAudioSampleRateHz] = useState(DEFAULT_SAMPLE_RATE);

  const audioRef = useRef(new RealtimeAudioPlayer());
  const capturedAudioRef = useRef<Float32Array[]>([]);
  const totalSamplesRef = useRef(0);
  const rawMixedTotalSamplesRef = useRef(0);
  const outputPlaybackRef = useRef<HTMLAudioElement | null>(null);
  const outputPlaybackUrlRef = useRef<string | null>(null);

  async function refreshStoredWaveforms(nextSessionId: string, inputSource: SessionLaunchConfig["inputSource"]): Promise<void> {
    const processedBlob = await fetchSessionWav(`/api/session/${nextSessionId}/processed-wav`);
    if (processedBlob) {
      const parsed = parsePcm16MonoWav(await processedBlob.arrayBuffer());
      if (parsed) {
        totalSamplesRef.current = parsed.samples.length;
        setWaveformBins(computeWaveformBinsFromPcm16(parsed.samples, WAVEFORM_BINS));
        setAudioSampleRateHz(parsed.sampleRateHz);
      }
    }

    const rawBlob = await fetchSessionWav(`/api/session/${nextSessionId}/raw-mix-wav`);
    if (rawBlob) {
      const parsed = parsePcm16MonoWav(await rawBlob.arrayBuffer());
      if (parsed) {
        rawMixedTotalSamplesRef.current = parsed.samples.length;
        setRawWaveformBins(computeWaveformBinsFromPcm16(parsed.samples, WAVEFORM_BINS));
        if (inputSource !== "respeaker_live") {
          setAudioSampleRateHz(parsed.sampleRateHz);
        }
      }
    }
  }

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
      algorithmMode: nextAlgorithmMode,
      localizationHopMs,
      localizationWindowMs,
      scenePath,
      backgroundNoisePath,
      backgroundNoiseGain,
      useGroundTruthLocation,
      useGroundTruthSpeakerSources,
      audioDeviceQuery,
      monitorSource: nextMonitorSource,
      sampleRateHz,
      channelMap,
    } = config;
    const playbackSampleRateHz = inputSource === "respeaker_live" ? sampleRateHz : DEFAULT_SAMPLE_RATE;
    setStatus("starting");
    setActiveInputSource(inputSource);
    setAlgorithmMode(nextAlgorithmMode);
    setMonitorSource(nextMonitorSource);
    setAudioSampleRateHz(playbackSampleRateHz);
    capturedAudioRef.current = [];
    totalSamplesRef.current = 0;
    rawMixedTotalSamplesRef.current = 0;
    setWaveformBins([]);
    setRawWaveformBins([]);
    setPlayheadMs(0);
    let resp: Response;
    try {
      resp = await fetch(apiUrl("/api/session/start"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          algorithm_mode: nextAlgorithmMode,
          localization_hop_ms: localizationHopMs,
          localization_window_ms: localizationWindowMs,
          input_source: inputSource,
          scene_config_path: scenePath,
          processing_mode: "specific_speaker_enhancement",
          monitor_source: nextMonitorSource,
          sample_rate_hz: inputSource === "respeaker_live" ? sampleRateHz : undefined,
          background_noise_audio_path: backgroundNoisePath,
          background_noise_gain: backgroundNoiseGain,
          use_ground_truth_location: inputSource === "simulation" ? useGroundTruthLocation : undefined,
          use_ground_truth_speaker_sources: inputSource === "simulation" ? useGroundTruthSpeakerSources : undefined,
          audio_device_query: inputSource === "respeaker_live" ? audioDeviceQuery : undefined,
          channel_map:
            inputSource === "respeaker_live" && channelMap.trim()
              ? channelMap.split(",").map((v) => Number(v.trim())).filter((v) => Number.isFinite(v))
              : undefined,
        }),
      });
    } catch {
      setStatus("error (network)");
      return;
    }
    if (!resp.ok) {
      setStatus(`error (${resp.status})`);
      return;
    }
    const payload = (await resp.json()) as { session_id: string };
    setSessionId(payload.session_id);
    void (async () => {
      for (let i = 0; i < 30; i += 1) {
        await refreshStoredWaveforms(payload.session_id, inputSource);
        if (totalSamplesRef.current > 0 || rawMixedTotalSamplesRef.current > 0) {
          break;
        }
        await new Promise((r) => setTimeout(r, 100));
      }
    })();
    audioRef.current.setTargetLatencyMs(latencyMs);
    await audioRef.current.start(playbackSampleRateHz);
    setPlaybackStats(audioRef.current.getStats());
    ws.connect(payload.session_id);
    setStatus("running");
  }

  async function stopSession(): Promise<void> {
    if (sessionId) {
      await fetch(apiUrl(`/api/session/${sessionId}/stop`), { method: "POST" });
    }
    resetLocalSessionState("stopped");
  }

  async function killCurrentRun(): Promise<void> {
    setStatus("stopping");
    const sid = sessionId;
    resetLocalSessionState("stopped");
    if (sid) {
      try {
        await fetch(apiUrl(`/api/session/${sid}/stop`), { method: "POST" });
      } catch {
        // Local teardown is authoritative for kill; backend stop is best effort.
      }
    }
  }

  async function stopActiveSession(): Promise<void> {
    try {
      await fetch(apiUrl("/api/session/active/stop"), { method: "POST" });
    } catch {
      // Best-effort stop for any externally started session.
    }
  }

  function onLatencyMsChange(nextLatencyMs: number): void {
    setLatencyMs(nextLatencyMs);
    audioRef.current.setTargetLatencyMs(nextLatencyMs);
    setPlaybackStats(audioRef.current.getStats());
  }

  function selectSpeaker(speakerId: number): void {
    setSelectedSpeakerId(speakerId);
    ws.send({ schema_version: SCHEMA_VERSION, type: "select_speaker", speaker_id: speakerId });
  }

  function adjustSpeakerGain(speakerId: number, step: 1 | -1): void {
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
    void (async () => {
      let blob: Blob | null = null;
      if (sessionId) {
        blob = await fetchSessionWav(`/api/session/${sessionId}/processed-wav`);
      }
      if (!blob && capturedAudioRef.current.length) {
        blob = createWavBlobFromFloat32Chunks(capturedAudioRef.current, audioSampleRateHz);
      }
      if (!blob) {
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const suffix = sessionId ? sessionId : "session";
      a.href = url;
      a.download = `realtime-output-${suffix}.wav`;
      a.click();
      URL.revokeObjectURL(url);
    })();
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
      if (sessionId) {
        blob = await fetchSessionWav(`/api/session/${sessionId}/processed-wav`);
      }
      if (!blob && capturedAudioRef.current.length) {
        blob = createWavBlobFromFloat32Chunks(capturedAudioRef.current, audioSampleRateHz);
      }
    } else {
      if (!sessionId) {
        return;
      }
      blob = await fetchSessionWav(`/api/session/${sessionId}/raw-mix-wav`);
      if (!blob) {
        return;
      }
    }
    if (!blob) {
      return;
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

  const totalDurationMs = (totalSamplesRef.current / audioSampleRateHz) * 1000;
  const rawMixedDurationMs = (rawMixedTotalSamplesRef.current / audioSampleRateHz) * 1000;
  const canPlayBeamformed = Boolean(sessionId) || capturedAudioRef.current.length > 0;
  const canPlayRawMixed = Boolean(sessionId);

  return (
    <main className="app-shell">
      <h1>Realtime Speaker UI Demo</h1>
      <div className={`layout ${metricsExpanded ? "" : "metrics-collapsed"}`.trim()}>
        <SceneLauncher
          status={status}
          defaultScenePath={defaultScenePath}
          defaultBackgroundNoisePath={defaultBackgroundNoisePath}
          defaultBackgroundNoiseGain={defaultBackgroundNoiseGain}
          onStart={startSession}
          onStop={stopSession}
          onKillRun={killCurrentRun}
          canKillRun={status === "running" || status === "starting" || status === "stopping"}
          onStopActiveSession={stopActiveSession}
          onDownloadWav={downloadWav}
          canDownloadWav={capturedAudioRef.current.length > 0}
          latencyMs={latencyMs}
          onLatencyMsChange={onLatencyMsChange}
          monitorSource={monitorSource}
          onMonitorSourceChange={onMonitorSourceChange}
        />
        <SpeakerStage
          speakers={speakers}
          groundTruth={groundTruth}
          processingMode="specific_speaker_enhancement"
          selectedSpeakerId={selectedSpeakerId}
          onSpeakerTap={selectSpeaker}
        />
        <MetricsPanel
          metrics={metrics}
          playback={playbackStats}
          expanded={metricsExpanded}
          onToggleExpanded={() => setMetricsExpanded((prev) => !prev)}
        />
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

      {selectedSpeakerId !== null && (
        <SpeakerControlPopover
          speakerId={selectedSpeakerId}
          deltaDb={gainBySpeaker[selectedSpeakerId] ?? 0}
          onAdjust={adjustSpeakerGain}
        />
      )}
    </main>
  );
}
