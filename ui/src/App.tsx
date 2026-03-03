import { useMemo, useRef, useState } from "react";

import { DemoWsClient } from "./api/ws";
import { RealtimeAudioPlayer } from "./audio/player";
import { createWavBlobFromFloat32Chunks } from "./audio/wav";
import { MetricsPanel } from "./components/MetricsPanel";
import { SceneLauncher } from "./components/SceneLauncher";
import { SpeakerControlPopover } from "./components/SpeakerControlPopover";
import { SpeakerStage } from "./components/SpeakerStage";
import { SCHEMA_VERSION, type MetricsMessage, type ServerMessage, type Speaker } from "./types/contracts";

const DEFAULT_SCENE = "simulation/simulations/configs/library_scene/library_k1_scene00.json";
const AUDIO_HEADER_BYTES = 16;
const AUDIO_SAMPLE_RATE = 16000;

export default function App() {
  const [status, setStatus] = useState("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [selectedSpeakerId, setSelectedSpeakerId] = useState<number | null>(null);
  const [gainBySpeaker, setGainBySpeaker] = useState<Record<number, number>>({});
  const [metrics, setMetrics] = useState<MetricsMessage | null>(null);

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
    await audioRef.current.start();
    ws.connect(payload.session_id);
    setStatus("running");
  }

  async function stopSession(): Promise<void> {
    if (sessionId) {
      await fetch(`http://localhost:8000/api/session/${sessionId}/stop`, { method: "POST" });
    }
    ws.close();
    audioRef.current.stop();
    setStatus("stopped");
    setSelectedSpeakerId(null);
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
        />
        <SpeakerStage speakers={speakers} selectedSpeakerId={selectedSpeakerId} onSpeakerTap={selectSpeaker} />
        <MetricsPanel metrics={metrics} />
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
