import { useEffect, useMemo, useState } from "react";

import { DemoWsClient } from "../api/ws";
import { DirectionalityViz } from "./DirectionalityViz";
import { SpeakerStage } from "./SpeakerStage";
import type { DataCollectionSet, RawChannelFile, RawChannelsResponse, RecordingArtifactManifest, RecordingEntry } from "../types/dataCollection";
import type { ServerMessage, Speaker } from "../types/contracts";
import { createZipBlob, textFile } from "../utils/zip";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const DEFAULT_SAMPLE_RATE_HZ = 48000;
const DEFAULT_CHANNEL_MAP = "0,1,2,3";
const DEFAULT_DEVICE = "ReSpeaker";
const DEFAULT_MIC_ARRAY_PROFILE = "respeaker_xvf3800_0650";
const EMPTY_SPEAKERS: Speaker[] = [];

type DirectionSample = {
  directionDeg: number;
  intensity: number;
  atMs: number;
};

function apiUrl(path: string): string {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

function nowIso(): string {
  return new Date().toISOString();
}

function makeId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

async function fetchRawArtifacts(sessionId: string): Promise<RecordingArtifactManifest> {
  let manifestResp: Response | null = null;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const candidate = await fetch(apiUrl(`/api/session/${sessionId}/raw-channels`));
    if (candidate.ok) {
      manifestResp = candidate;
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  if (!manifestResp) {
    throw new Error("raw channel manifest failed");
  }
  const manifest = (await manifestResp.json()) as RawChannelsResponse;
  const channels: RawChannelFile[] = [];
  for (const channel of manifest.channels) {
    let wavResp: Response | null = null;
    for (let attempt = 0; attempt < 20; attempt += 1) {
      const candidate = await fetch(apiUrl(`/api/session/${sessionId}/raw-channel/${channel.channel_index}.wav`));
      if (candidate.ok) {
        wavResp = candidate;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    if (!wavResp) {
      throw new Error(`raw channel ${channel.channel_index} failed`);
    }
    channels.push({
      channelIndex: channel.channel_index,
      filename: channel.filename,
      bytes: new Uint8Array(await wavResp.arrayBuffer()),
    });
  }
  return { sampleRateHz: manifest.sample_rate_hz, channels };
}

export function DataCollectionPage() {
  const [collectionId, setCollectionId] = useState("dataset-set-001");
  const [collectionTitle, setCollectionTitle] = useState("Capstone data collection");
  const [collectionNotes, setCollectionNotes] = useState("");
  const [deviceName, setDeviceName] = useState(DEFAULT_DEVICE);
  const [micArrayProfile, setMicArrayProfile] = useState(DEFAULT_MIC_ARRAY_PROFILE);
  const [createdAtIso] = useState(nowIso);
  const [sessionStatus, setSessionStatus] = useState("idle");
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentRecordingStartIso, setCurrentRecordingStartIso] = useState<string | null>(null);
  const [recordings, setRecordings] = useState<RecordingEntry[]>([]);
  const [statusMessage, setStatusMessage] = useState("");
  const [speakers, setSpeakers] = useState<Speaker[]>(EMPTY_SPEAKERS);
  const [directionTrail, setDirectionTrail] = useState<DirectionSample[]>([]);

  const ws = useMemo(
    () =>
      new DemoWsClient({
        onServerMessage: (msg: ServerMessage) => {
          if (msg.type !== "speaker_state") {
            return;
          }
          setSpeakers(msg.speakers);
          const visibleSpeakers = msg.speakers.filter((speaker) => speaker.active);
          if (!visibleSpeakers.length) {
            return;
          }
          const atMs = Date.now();
          setDirectionTrail((prev) => {
            const next = [
              ...prev,
              ...visibleSpeakers.map((speaker) => ({
                directionDeg: speaker.direction_degrees,
                intensity: Math.max(speaker.activity_confidence, speaker.confidence, 0.2),
                atMs,
              })),
            ];
            return next.slice(-120);
          });
        },
        onAudioChunk: () => undefined,
        onClose: () => {
          setSpeakers([]);
        },
      }),
    []
  );

  useEffect(() => () => ws.close(), [ws]);

  const canStartRecording = !currentSessionId && sessionStatus !== "starting" && sessionStatus !== "stopping";
  const canStopRecording = Boolean(currentSessionId);

  async function startRecording(): Promise<void> {
    setSessionStatus("starting");
    setStatusMessage("");
    try {
      const resp = await fetch(apiUrl("/api/session/start"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_source: "respeaker_live",
          separation_mode: "mock",
          processing_mode: "specific_speaker_enhancement",
          monitor_source: "processed",
          sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
          audio_device_query: deviceName,
          mic_array_profile: micArrayProfile,
          channel_map: DEFAULT_CHANNEL_MAP.split(",").map((value) => Number(value)),
        }),
      });
      if (!resp.ok) {
        throw new Error(`start failed (${resp.status})`);
      }
      const payload = (await resp.json()) as { session_id: string };
      setCurrentSessionId(payload.session_id);
      setCurrentRecordingStartIso(nowIso());
      setSpeakers([]);
      setDirectionTrail([]);
      ws.connect(payload.session_id);
      setSessionStatus("capturing");
      setStatusMessage(`Recording collection ${collectionId} into session ${payload.session_id}.`);
    } catch (err) {
      setSessionStatus("error");
      setStatusMessage(err instanceof Error ? err.message : "Failed to start recording.");
    }
  }

  async function stopRecording(): Promise<void> {
    if (!currentSessionId) {
      return;
    }
    const sessionId = currentSessionId;
    const startedAtIso = currentRecordingStartIso ?? nowIso();
    setSessionStatus("stopping");
    setStatusMessage(`Stopping session ${sessionId} and downloading raw channels.`);
    try {
      ws.close();
      await fetch(apiUrl(`/api/session/${sessionId}/stop`), { method: "POST" });
      const artifacts = await fetchRawArtifacts(sessionId);
      setRecordings((prev) => [
        ...prev,
        {
          recordingId: makeId("recording"),
          sessionId,
          startedAtIso,
          stoppedAtIso: nowIso(),
          status: "ready",
          deviceName,
          artifacts,
        },
      ]);
      setSessionStatus("idle");
      setStatusMessage(`Saved recording with ${artifacts.channels.length} raw channels.`);
    } catch (err) {
      setRecordings((prev) => [
        ...prev,
        {
          recordingId: makeId("recording"),
          sessionId,
          startedAtIso,
          stoppedAtIso: nowIso(),
          status: "failed",
          deviceName,
          error: err instanceof Error ? err.message : "Artifact collection failed.",
        },
      ]);
      setSessionStatus("error");
      setStatusMessage(err instanceof Error ? err.message : "Artifact collection failed.");
    } finally {
      setCurrentSessionId(null);
      setCurrentRecordingStartIso(null);
      setSpeakers([]);
    }
  }

  function exportCollection(): void {
    if (!recordings.length) {
      setStatusMessage("Add at least one recording before export.");
      return;
    }

    const dataset: DataCollectionSet = {
      collectionId: collectionId.trim() || "dataset-set",
      title: collectionTitle.trim(),
      notes: collectionNotes.trim(),
      createdAtIso,
      deviceName,
      recordings,
    };

    const entries = [
      textFile(
        "collection.json",
        JSON.stringify(
          {
            collectionId: dataset.collectionId,
            title: dataset.title,
            notes: dataset.notes,
            createdAtIso: dataset.createdAtIso,
            deviceName: dataset.deviceName,
            recordings: dataset.recordings.map((recording) => ({
              recordingId: recording.recordingId,
              sessionId: recording.sessionId,
              startedAtIso: recording.startedAtIso,
              stoppedAtIso: recording.stoppedAtIso,
              status: recording.status,
              deviceName: recording.deviceName,
              error: recording.error ?? null,
              sampleRateHz: recording.artifacts?.sampleRateHz ?? null,
              channels:
                recording.artifacts?.channels.map((channel) => ({
                  channelIndex: channel.channelIndex,
                  filename: channel.filename,
                })) ?? [],
            })),
          },
          null,
          2
        )
      ),
      ...recordings.flatMap((recording) => {
        const files = [
          textFile(
            `recordings/${recording.recordingId}/metadata.json`,
            JSON.stringify(
              {
                recordingId: recording.recordingId,
                sessionId: recording.sessionId,
                startedAtIso: recording.startedAtIso,
                stoppedAtIso: recording.stoppedAtIso,
                status: recording.status,
                deviceName: recording.deviceName,
                error: recording.error ?? null,
                sampleRateHz: recording.artifacts?.sampleRateHz ?? null,
              },
              null,
              2
            )
          ),
        ];
        if (!recording.artifacts) {
          return files;
        }
        return [
          ...files,
          ...recording.artifacts.channels.map((channel) => ({
            path: `recordings/${recording.recordingId}/raw/${channel.filename}`,
            bytes: channel.bytes,
          })),
        ];
      }),
    ];

    downloadBlob(`${dataset.collectionId || "dataset-set"}.zip`, createZipBlob(entries));
    setStatusMessage(`Exported ${recordings.length} recordings.`);
  }

  return (
    <main className="app-shell">
      <h1>Data Collection</h1>
      <p className="page-copy">Enter collection details, pick the capture device, then record and stop runs into the current set.</p>

      <div className="layout data-layout">
        <section className="panel">
          <h2>Collection</h2>
          <label htmlFor="collection-id">Collection id</label>
          <input id="collection-id" value={collectionId} onChange={(e) => setCollectionId(e.target.value)} />

          <label htmlFor="collection-title">Title</label>
          <input id="collection-title" value={collectionTitle} onChange={(e) => setCollectionTitle(e.target.value)} />

          <label htmlFor="collection-notes">Notes</label>
          <textarea id="collection-notes" rows={4} value={collectionNotes} onChange={(e) => setCollectionNotes(e.target.value)} />

          <label htmlFor="device-name">Device</label>
          <select id="device-name" aria-label="Device" value={deviceName} onChange={(e) => setDeviceName(e.target.value)}>
            <option value="ReSpeaker">ReSpeaker</option>
            <option value="XVF3800">XVF3800</option>
          </select>

          <label htmlFor="mic-array-profile">Mic array profile</label>
          <select
            id="mic-array-profile"
            aria-label="Mic array profile"
            value={micArrayProfile}
            onChange={(e) => setMicArrayProfile(e.target.value)}
          >
            <option value="respeaker_xvf3800_0650">ReSpeaker XVF3800 (65.0 mm)</option>
            <option value="respeaker_v3_0457">ReSpeaker 4-Mic v3 (45.7 mm)</option>
          </select>

          <p className="status">Created: {new Date(createdAtIso).toLocaleString()}</p>
        </section>

        <section className="panel">
          <h2>Recorder</h2>
          <p className="status">Status: {sessionStatus}</p>
          {currentSessionId ? <p className="status">Active session: {currentSessionId}</p> : null}
          <div className="actions">
            <button type="button" onClick={() => void startRecording()} disabled={!canStartRecording}>
              Record
            </button>
            <button type="button" onClick={() => void stopRecording()} disabled={!canStopRecording}>
              Stop
            </button>
            <button type="button" onClick={() => exportCollection()} disabled={!recordings.length}>
              Export Set
            </button>
          </div>
          {statusMessage ? <p className="status">{statusMessage}</p> : null}
        </section>

        <div className="data-visual-stack">
          <SpeakerStage
            speakers={speakers}
            groundTruth={[]}
            processingMode="localize_and_beamform"
            selectedSpeakerId={null}
            onSpeakerTap={() => undefined}
            showGroundTruth={false}
          />
          <DirectionalityViz speakers={speakers} trail={directionTrail} />
        </div>
      </div>

      <section className="panel">
        <h2>Recordings</h2>
        {!recordings.length ? (
          <p className="empty-state">No recordings captured yet.</p>
        ) : (
          <div className="recording-list">
            {recordings.map((recording) => (
              <article key={recording.recordingId} className="recording-card">
                <h3>{recording.recordingId}</h3>
                <p className="recording-meta">{recording.status} · {recording.deviceName}</p>
                <p className="recording-meta">Session: {recording.sessionId}</p>
                <p className="recording-meta">
                  Raw channels: {recording.artifacts?.channels.length ?? 0}
                  {recording.artifacts ? ` @ ${recording.artifacts.sampleRateHz} Hz` : ""}
                </p>
                {recording.error ? <p className="status">{recording.error}</p> : null}
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
