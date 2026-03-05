import { useMemo, useRef, useState } from "react";

import { DemoWsClient } from "./api/ws";
import { RealtimeAudioPlayer, type PlaybackStats } from "./audio/player";
import { createWavBlobFromFloat32Chunks } from "./audio/wav";
import { MetricsPanel } from "./components/MetricsPanel";
import { SceneLauncher } from "./components/SceneLauncher";
import { SpeakerControlPopover } from "./components/SpeakerControlPopover";
import { SpeakerStage } from "./components/SpeakerStage";
import { SCHEMA_VERSION, type MetricsMessage, type ServerMessage, type Speaker } from "./types/contracts";

const DEFAULT_SCENE = "simulation/simulations/configs/library_scene/library_k1_scene00.json";
const AUDIO_HEADER_BYTES = 16;
const AUDIO_SAMPLE_RATE = 16000;
const DEFAULT_LATENCY_MS = 180;
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

export default function App() {
  const [status, setStatus] = useState("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [selectedSpeakerId, setSelectedSpeakerId] = useState<number | null>(null);
  const [gainBySpeaker, setGainBySpeaker] = useState<Record<number, number>>({});
  const [metrics, setMetrics] = useState<MetricsMessage | null>(null);
  const [latencyMs, setLatencyMs] = useState(DEFAULT_LATENCY_MS);
  const [playbackStats, setPlaybackStats] = useState<PlaybackStats>(DEFAULT_PLAYBACK_STATS);

  const audioRef = useRef(new RealtimeAudioPlayer());
  const capturedAudioRef = useRef<Float32Array[]>([]);

  const ws = useMemo(
    () =>
      new DemoWsClient({
        onServerMessage: (msg: ServerMessage) => {
          if (msg.type === "speaker_state") {
            setSpeakers(msg.speakers);
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
            }
          }
          audioRef.current.pushPacket(chunk);
          setPlaybackStats(audioRef.current.getStats());
        },
        onClose: () => setStatus((s) => (s === "running" ? "disconnected" : s)),
      }),
    []
  );

  async function startSession(scenePath: string): Promise<void> {
    setStatus("starting");
    capturedAudioRef.current = [];
    const resp = await fetch("http://localhost:8000/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_config_path: scenePath, separation_mode: "mock" }),
    });
    if (!resp.ok) {
      setStatus(`error (${resp.status})`);
      return;
    }
    const payload = (await resp.json()) as { session_id: string };
    setSessionId(payload.session_id);
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
    ws.close();
    audioRef.current.stop();
    setPlaybackStats(DEFAULT_PLAYBACK_STATS);
    setStatus("stopped");
    setSelectedSpeakerId(null);
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

  return (
    <main className="app-shell">
      <h1>Realtime Speaker UI Demo</h1>
      <div className="layout">
        <SceneLauncher
          status={status}
          defaultScenePath={DEFAULT_SCENE}
          onStart={startSession}
          onStop={stopSession}
          onDownloadWav={downloadWav}
          canDownloadWav={capturedAudioRef.current.length > 0}
          latencyMs={latencyMs}
          onLatencyMsChange={onLatencyMsChange}
        />
        <SpeakerStage speakers={speakers} selectedSpeakerId={selectedSpeakerId} onSpeakerTap={selectSpeaker} />
        <MetricsPanel metrics={metrics} playback={playbackStats} />
      </div>
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
