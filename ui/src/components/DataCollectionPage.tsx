import { useEffect, useMemo, useState } from "react";

import { DemoWsClient } from "../api/ws";
import { DirectionalityViz } from "./DirectionalityViz";
import { SpeakerStage } from "./SpeakerStage";
import type { MicArrayProfile } from "../utils/direction";
import type {
  AnnotatedSpeaker,
  DataCollectionSet,
  RawChannelFile,
  RawChannelsResponse,
  RecordingArtifactManifest,
  RecordingEntry,
} from "../types/dataCollection";
import type { ServerMessage, Speaker } from "../types/contracts";
import { backendArrivalToUiSourceBearingDeg } from "../utils/direction";
import { createZipBlob, textFile } from "../utils/zip";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const DEFAULT_SAMPLE_RATE_HZ = 48000;
const DEFAULT_DEVICE = "XVF3800";
const DEFAULT_MIC_ARRAY_PROFILE: MicArrayProfile = "respeaker_xvf3800_0650";
const EMPTY_SPEAKERS: Speaker[] = [];
const ADJECTIVES = ["amber", "brisk", "calm", "daring", "ember", "fuzzy", "golden", "harbor"];
const ANIMALS = ["otter", "lynx", "falcon", "badger", "fox", "heron", "panda", "wren"];

type DirectionSample = {
  directionDeg: number;
  intensity: number;
  atMs: number;
};

type RawChannelPlayersProps = {
  artifacts: RecordingArtifactManifest;
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

function makeDefaultSpeakerName(index: number): string {
  const adjective = ADJECTIVES[index % ADJECTIVES.length];
  const animal = ANIMALS[Math.floor(index / ADJECTIVES.length) % ANIMALS.length];
  return `${adjective}-${animal}`;
}

function makeAnnotatedSpeaker(index: number): AnnotatedSpeaker {
  return {
    speakerName: makeDefaultSpeakerName(index),
    directionDeg: 0,
  };
}

function normalizeDirectionDeg(value: number): number {
  const wrapped = value % 360;
  return wrapped < 0 ? wrapped + 360 : wrapped;
}

function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function RawChannelPlayers({ artifacts }: RawChannelPlayersProps) {
  const channelEntries = useMemo(
    () =>
      artifacts.channels.map((channel) => ({
        channelIndex: channel.channelIndex,
        label: `raw ch${channel.channelIndex} · mic ${channel.channelIndex + 1}`,
        url: (() => {
          const audioBytes = new Uint8Array(channel.bytes.byteLength);
          audioBytes.set(channel.bytes);
          return URL.createObjectURL(new Blob([audioBytes], { type: "audio/wav" }));
        })(),
      })),
    [artifacts]
  );

  useEffect(
    () => () => {
      for (const entry of channelEntries) {
        URL.revokeObjectURL(entry.url);
      }
    },
    [channelEntries]
  );

  return (
    <div className="recording-list">
      {channelEntries.map((entry) => (
        <article key={`player-${entry.channelIndex}`} className="recording-card">
          <h3>{entry.label}</h3>
          <audio controls preload="none" src={entry.url}>
            <track kind="captions" />
          </audio>
        </article>
      ))}
    </div>
  );
}

async function fetchRawArtifacts(sessionId: string, speakers: AnnotatedSpeaker[]): Promise<RecordingArtifactManifest> {
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
  let rawChannelPlot: RecordingArtifactManifest["rawChannelPlot"];
  try {
    const subtitle = rawChannelPlotSubtitle(speakers);
    const query = subtitle ? `?subtitle=${encodeURIComponent(subtitle)}` : "";
    const plotResp = await fetch(apiUrl(`/api/session/${sessionId}/raw-channels-plot.png${query}`));
    if (plotResp.ok) {
      rawChannelPlot = {
        filename: "raw_channels.png",
        bytes: new Uint8Array(await plotResp.arrayBuffer()),
      };
    }
  } catch {
    rawChannelPlot = undefined;
  }
  return { sampleRateHz: manifest.sample_rate_hz, channels, rawChannelPlot };
}

function rawChannelPlotSubtitle(speakers: AnnotatedSpeaker[]): string {
  if (!speakers.length) {
    return "";
  }
  return speakers
    .map((speaker) => `${speaker.speakerName.trim() || "speaker"} ${normalizeDirectionDeg(speaker.directionDeg).toFixed(0)}°`)
    .join(" · ");
}

export function DataCollectionPage() {
  const [collectionId, setCollectionId] = useState("dataset-set-001");
  const [collectionTitle, setCollectionTitle] = useState("Capstone data collection");
  const [collectionNotes, setCollectionNotes] = useState("");
  const [deviceName, setDeviceName] = useState(DEFAULT_DEVICE);
  const [micArrayProfile, setMicArrayProfile] = useState<MicArrayProfile>(DEFAULT_MIC_ARRAY_PROFILE);
  const [createdAtIso] = useState(nowIso);
  const [sessionStatus, setSessionStatus] = useState("idle");
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentRecordingStartIso, setCurrentRecordingStartIso] = useState<string | null>(null);
  const [recordings, setRecordings] = useState<RecordingEntry[]>([]);
  const [statusMessage, setStatusMessage] = useState("");
  const [speakers, setSpeakers] = useState<Speaker[]>(EMPTY_SPEAKERS);
  const [directionTrail, setDirectionTrail] = useState<DirectionSample[]>([]);
  const [pendingRecordingNotes, setPendingRecordingNotes] = useState("");
  const [pendingRecordingSpeakers, setPendingRecordingSpeakers] = useState<AnnotatedSpeaker[]>([makeAnnotatedSpeaker(0)]);

  const ws = useMemo(
    () =>
      new DemoWsClient({
        onServerMessage: (msg: ServerMessage) => {
          if (msg.type !== "speaker_state") {
            return;
          }
          const displaySpeakers = msg.speakers.map((speaker) => ({
            ...speaker,
            direction_degrees: backendArrivalToUiSourceBearingDeg(speaker.direction_degrees, micArrayProfile),
          }));
          setSpeakers(displaySpeakers);
          const visibleSpeakers = displaySpeakers.filter((speaker) => speaker.active);
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
    [micArrayProfile]
  );

  useEffect(() => () => ws.close(), [ws]);

  function addPendingSpeaker(): void {
    setPendingRecordingSpeakers((prev) => [...prev, makeAnnotatedSpeaker(prev.length)]);
  }

  function updatePendingSpeaker(index: number, field: keyof AnnotatedSpeaker, value: string): void {
    setPendingRecordingSpeakers((prev) =>
      prev.map((speaker, speakerIdx) =>
        speakerIdx !== index
          ? speaker
          : {
              ...speaker,
              [field]:
                field === "directionDeg"
                  ? normalizeDirectionDeg(Number.parseFloat(value || "0") || 0)
                  : value,
            }
      )
    );
  }

  function removePendingSpeaker(index: number): void {
    setPendingRecordingSpeakers((prev) => prev.filter((_, speakerIdx) => speakerIdx !== index));
  }

  function updateRecordingNotes(recordingId: string, notes: string): void {
    setRecordings((prev) => prev.map((recording) => (recording.recordingId === recordingId ? { ...recording, notes } : recording)));
  }

  function updateRecordingSpeaker(recordingId: string, index: number, field: keyof AnnotatedSpeaker, value: string): void {
    setRecordings((prev) =>
      prev.map((recording) =>
        recording.recordingId !== recordingId
          ? recording
          : {
              ...recording,
              speakers: recording.speakers.map((speaker, speakerIdx) =>
                speakerIdx !== index
                  ? speaker
                  : {
                      ...speaker,
                      [field]:
                        field === "directionDeg"
                          ? normalizeDirectionDeg(Number.parseFloat(value || "0") || 0)
                          : value,
                    }
              ),
            }
      )
    );
  }

  function addRecordingSpeaker(recordingId: string): void {
    setRecordings((prev) =>
      prev.map((recording) =>
        recording.recordingId !== recordingId
          ? recording
          : { ...recording, speakers: [...recording.speakers, makeAnnotatedSpeaker(recording.speakers.length)] }
      )
    );
  }

  function removeRecordingSpeaker(recordingId: string, index: number): void {
    setRecordings((prev) =>
      prev.map((recording) =>
        recording.recordingId !== recordingId
          ? recording
          : { ...recording, speakers: recording.speakers.filter((_, speakerIdx) => speakerIdx !== index) }
      )
    );
  }

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
          monitor_source: "raw_mixed",
          sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
          audio_device_query: deviceName,
          mic_array_profile: micArrayProfile,
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
    const capturedSpeakers = pendingRecordingSpeakers.map((speaker, index) => ({
      speakerName: speaker.speakerName.trim() || makeDefaultSpeakerName(index),
      directionDeg: normalizeDirectionDeg(speaker.directionDeg),
    }));
    setSessionStatus("stopping");
    setStatusMessage(`Stopping session ${sessionId} and downloading raw channels.`);
    try {
      ws.close();
      await fetch(apiUrl(`/api/session/${sessionId}/stop`), { method: "POST" });
      const artifacts = await fetchRawArtifacts(sessionId, capturedSpeakers);
      setRecordings((prev) => [
        ...prev,
        {
          recordingId: makeId("recording"),
          sessionId,
          startedAtIso,
          stoppedAtIso: nowIso(),
          status: "ready",
          deviceName,
          micArrayProfile,
          notes: pendingRecordingNotes.trim(),
          speakers: capturedSpeakers,
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
          micArrayProfile,
          notes: pendingRecordingNotes.trim(),
          speakers: capturedSpeakers,
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
      micArrayProfile,
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
            micArrayProfile: dataset.micArrayProfile,
            recordings: dataset.recordings.map((recording) => ({
              recordingId: recording.recordingId,
              sessionId: recording.sessionId,
              startedAtIso: recording.startedAtIso,
              stoppedAtIso: recording.stoppedAtIso,
              status: recording.status,
              deviceName: recording.deviceName,
              micArrayProfile: recording.micArrayProfile,
              notes: recording.notes,
              speakers: recording.speakers.map((speaker) => ({
                speakerName: speaker.speakerName,
                directionDeg: speaker.directionDeg,
              })),
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
                micArrayProfile: recording.micArrayProfile,
                notes: recording.notes,
                speakers: recording.speakers.map((speaker) => ({
                  speakerName: speaker.speakerName,
                  directionDeg: speaker.directionDeg,
                })),
                error: recording.error ?? null,
                sampleRateHz: recording.artifacts?.sampleRateHz ?? null,
              },
              null,
              2
            )
          ),
        ];
        if (recording.artifacts?.rawChannelPlot) {
          files.push({
            path: `recordings/${recording.recordingId}/visualizations/${recording.artifacts.rawChannelPlot.filename}`,
            bytes: recording.artifacts.rawChannelPlot.bytes,
          });
        }
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
            <option value="XVF3800">XVF3800</option>
            <option value="ReSpeaker">ReSpeaker</option>
          </select>

          <label htmlFor="mic-array-profile">Mic array profile</label>
          <select
            id="mic-array-profile"
            aria-label="Mic array profile"
            value={micArrayProfile}
            onChange={(e) => setMicArrayProfile(e.target.value as MicArrayProfile)}
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

        <section className="panel">
          <h2>Next Recording Scene</h2>
          <label htmlFor="recording-notes">Recording notes</label>
          <textarea
            id="recording-notes"
            aria-label="Recording notes"
            rows={4}
            value={pendingRecordingNotes}
            onChange={(e) => setPendingRecordingNotes(e.target.value)}
          />
          <div className="recording-list">
            {pendingRecordingSpeakers.map((speaker, index) => (
              <article key={`pending-speaker-${index}`} className="recording-card">
                <h3>Speaker {index + 1}</h3>
                <label htmlFor={`pending-speaker-name-${index}`}>Name</label>
                <input
                  id={`pending-speaker-name-${index}`}
                  aria-label={`Pending speaker ${index + 1} name`}
                  value={speaker.speakerName}
                  onChange={(e) => updatePendingSpeaker(index, "speakerName", e.target.value)}
                />
                <label htmlFor={`pending-speaker-doa-${index}`}>DOA (deg)</label>
                <input
                  id={`pending-speaker-doa-${index}`}
                  aria-label={`Pending speaker ${index + 1} DOA`}
                  type="number"
                  value={speaker.directionDeg}
                  onChange={(e) => updatePendingSpeaker(index, "directionDeg", e.target.value)}
                />
                <div className="actions">
                  <button type="button" onClick={() => removePendingSpeaker(index)} disabled={pendingRecordingSpeakers.length <= 1}>
                    Remove Speaker
                  </button>
                </div>
              </article>
            ))}
          </div>
          <div className="actions">
            <button type="button" onClick={addPendingSpeaker}>
              Add Speaker
            </button>
          </div>
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
                <p className="recording-meta">Mic profile: {recording.micArrayProfile}</p>
                <p className="recording-meta">
                  Raw channels: {recording.artifacts?.channels.length ?? 0}
                  {recording.artifacts ? ` @ ${recording.artifacts.sampleRateHz} Hz` : ""}
                </p>
                {recording.artifacts ? <RawChannelPlayers artifacts={recording.artifacts} /> : null}
                <label htmlFor={`recording-notes-${recording.recordingId}`}>Recording notes</label>
                <textarea
                  id={`recording-notes-${recording.recordingId}`}
                  aria-label={`Recording notes ${recording.recordingId}`}
                  rows={3}
                  value={recording.notes}
                  onChange={(e) => updateRecordingNotes(recording.recordingId, e.target.value)}
                />
                <div className="recording-list">
                  {recording.speakers.map((speaker, index) => (
                    <article key={`${recording.recordingId}-speaker-${index}`} className="recording-card">
                      <h3>Speaker {index + 1}</h3>
                      <label htmlFor={`${recording.recordingId}-speaker-name-${index}`}>Name</label>
                      <input
                        id={`${recording.recordingId}-speaker-name-${index}`}
                        aria-label={`Speaker ${index + 1} name for ${recording.recordingId}`}
                        value={speaker.speakerName}
                        onChange={(e) => updateRecordingSpeaker(recording.recordingId, index, "speakerName", e.target.value)}
                      />
                      <label htmlFor={`${recording.recordingId}-speaker-doa-${index}`}>DOA (deg)</label>
                      <input
                        id={`${recording.recordingId}-speaker-doa-${index}`}
                        aria-label={`Speaker ${index + 1} DOA for ${recording.recordingId}`}
                        type="number"
                        value={speaker.directionDeg}
                        onChange={(e) => updateRecordingSpeaker(recording.recordingId, index, "directionDeg", e.target.value)}
                      />
                      <div className="actions">
                        <button
                          type="button"
                          onClick={() => removeRecordingSpeaker(recording.recordingId, index)}
                          disabled={recording.speakers.length <= 1}
                        >
                          Remove Speaker
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
                <div className="actions">
                  <button type="button" onClick={() => addRecordingSpeaker(recording.recordingId)}>
                    Add Speaker
                  </button>
                </div>
                {recording.error ? <p className="status">{recording.error}</p> : null}
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
