import { useMemo, useState } from "react";

import type { MonitorSource, ProcessingMode } from "../types/contracts";
import type {
  DataCollectionSet,
  RawChannelFile,
  RawChannelsResponse,
  RecordingArtifactManifest,
  RecordingEntry,
  SceneDefinition,
  SceneSessionConfig,
} from "../types/dataCollection";
import { createZipBlob, textFile } from "../utils/zip";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

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
  return {
    sampleRateHz: manifest.sample_rate_hz,
    channels,
  };
}

type Props = {
  defaultScenePath: string;
  defaultBackgroundNoisePath: string;
  defaultBackgroundNoiseGain: number;
  defaultProcessingMode: ProcessingMode;
  defaultMonitorSource: MonitorSource;
};

type SceneFormState = {
  sceneId: string;
  title: string;
  description: string;
  inputSource: SceneSessionConfig["inputSource"];
  scenePath: string;
  backgroundNoisePath: string;
  backgroundNoiseGain: number;
  audioDeviceQuery: string;
  monitorSource: MonitorSource;
  sampleRateHz: number;
  channelMap: string;
  processingMode: ProcessingMode;
};

export function DataCollectionPage({
  defaultScenePath,
  defaultBackgroundNoisePath,
  defaultBackgroundNoiseGain,
  defaultProcessingMode,
  defaultMonitorSource,
}: Props) {
  const [collectionId, setCollectionId] = useState("dataset-set-001");
  const [collectionTitle, setCollectionTitle] = useState("Capstone data collection");
  const [collectionNotes, setCollectionNotes] = useState("");
  const [createdAtIso] = useState(nowIso);
  const [scenes, setScenes] = useState<SceneDefinition[]>([]);
  const [selectedSceneId, setSelectedSceneId] = useState<string>("");
  const [editingSceneId, setEditingSceneId] = useState<string | null>(null);
  const [sessionStatus, setSessionStatus] = useState("idle");
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentRecordingStartIso, setCurrentRecordingStartIso] = useState<string | null>(null);
  const [currentSceneSnapshot, setCurrentSceneSnapshot] = useState<SceneDefinition | null>(null);
  const [recordings, setRecordings] = useState<RecordingEntry[]>([]);
  const [statusMessage, setStatusMessage] = useState("");
  const [sceneForm, setSceneForm] = useState<SceneFormState>({
    sceneId: "scene-001",
    title: "Library scene",
    description: "",
    inputSource: "simulation",
    scenePath: defaultScenePath,
    backgroundNoisePath: defaultBackgroundNoisePath,
    backgroundNoiseGain: defaultBackgroundNoiseGain,
    audioDeviceQuery: "ReSpeaker",
    monitorSource: defaultMonitorSource,
    sampleRateHz: 48000,
    channelMap: "0,1,2,3",
    processingMode: defaultProcessingMode,
  });

  const selectedScene = useMemo(
    () => scenes.find((scene) => scene.sceneId === selectedSceneId) ?? null,
    [scenes, selectedSceneId]
  );
  const isBusy = sessionStatus === "starting" || sessionStatus === "stopping" || sessionStatus === "capturing";
  const canStartRecording = Boolean(selectedScene) && !isBusy && !currentSessionId;
  const canStopRecording = Boolean(currentSessionId) && (sessionStatus === "capturing" || sessionStatus === "running");

  function resetSceneForm(nextId = `scene-${String(scenes.length + 1).padStart(3, "0")}`): void {
    setSceneForm({
      sceneId: nextId,
      title: "",
      description: "",
      inputSource: "simulation",
      scenePath: defaultScenePath,
      backgroundNoisePath: defaultBackgroundNoisePath,
      backgroundNoiseGain: defaultBackgroundNoiseGain,
      audioDeviceQuery: "ReSpeaker",
      monitorSource: defaultMonitorSource,
      sampleRateHz: 48000,
      channelMap: "0,1,2,3",
      processingMode: defaultProcessingMode,
    });
    setEditingSceneId(null);
  }

  function toSceneDefinition(form: SceneFormState): SceneDefinition {
    return {
      sceneId: form.sceneId.trim(),
      title: form.title.trim(),
      description: form.description.trim(),
      sessionConfig: {
        inputSource: form.inputSource,
        scenePath: form.scenePath.trim(),
        backgroundNoisePath: form.backgroundNoisePath.trim(),
        backgroundNoiseGain: form.backgroundNoiseGain,
        audioDeviceQuery: form.audioDeviceQuery.trim(),
        monitorSource: form.monitorSource,
        sampleRateHz: form.sampleRateHz,
        channelMap: form.channelMap.trim(),
        processingMode: form.processingMode,
      },
    };
  }

  function saveScene(): void {
    const nextScene = toSceneDefinition(sceneForm);
    if (!nextScene.sceneId || !nextScene.title) {
      setStatusMessage("Scene id and title are required.");
      return;
    }
    if (
      scenes.some((scene) => scene.sceneId === nextScene.sceneId && scene.sceneId !== editingSceneId)
    ) {
      setStatusMessage(`Scene id ${nextScene.sceneId} already exists.`);
      return;
    }
    setScenes((prev) => {
      if (editingSceneId) {
        return prev.map((scene) => (scene.sceneId === editingSceneId ? nextScene : scene));
      }
      return [...prev, nextScene];
    });
    setSelectedSceneId(nextScene.sceneId);
    setStatusMessage(editingSceneId ? `Updated ${nextScene.sceneId}.` : `Added ${nextScene.sceneId}.`);
    resetSceneForm();
  }

  function editScene(scene: SceneDefinition): void {
    setEditingSceneId(scene.sceneId);
    setSceneForm({
      sceneId: scene.sceneId,
      title: scene.title,
      description: scene.description,
      inputSource: scene.sessionConfig.inputSource,
      scenePath: scene.sessionConfig.scenePath,
      backgroundNoisePath: scene.sessionConfig.backgroundNoisePath,
      backgroundNoiseGain: scene.sessionConfig.backgroundNoiseGain,
      audioDeviceQuery: scene.sessionConfig.audioDeviceQuery,
      monitorSource: scene.sessionConfig.monitorSource,
      sampleRateHz: scene.sessionConfig.sampleRateHz,
      channelMap: scene.sessionConfig.channelMap,
      processingMode: scene.sessionConfig.processingMode,
    });
  }

  function deleteScene(sceneId: string): void {
    setScenes((prev) => prev.filter((scene) => scene.sceneId !== sceneId));
    if (selectedSceneId === sceneId) {
      setSelectedSceneId("");
    }
    if (editingSceneId === sceneId) {
      resetSceneForm();
    }
  }

  async function startRecording(): Promise<void> {
    if (!selectedScene) {
      return;
    }
    setSessionStatus("starting");
    setStatusMessage("");
    const config = selectedScene.sessionConfig;
    try {
      const resp = await fetch(apiUrl("/api/session/start"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_source: config.inputSource,
          scene_config_path: config.scenePath,
          separation_mode: "mock",
          processing_mode: config.processingMode,
          monitor_source: config.monitorSource,
          sample_rate_hz: config.inputSource === "respeaker_live" ? config.sampleRateHz : undefined,
          background_noise_audio_path: config.backgroundNoisePath,
          background_noise_gain: config.backgroundNoiseGain,
          audio_device_query: config.inputSource === "respeaker_live" ? config.audioDeviceQuery : undefined,
          channel_map:
            config.inputSource === "respeaker_live" && config.channelMap.trim()
              ? config.channelMap
                  .split(",")
                  .map((value) => Number(value.trim()))
                  .filter((value) => Number.isFinite(value))
              : undefined,
        }),
      });
      if (!resp.ok) {
        throw new Error(`start failed (${resp.status})`);
      }
      const payload = (await resp.json()) as { session_id: string };
      setCurrentSessionId(payload.session_id);
      setCurrentRecordingStartIso(nowIso());
      setCurrentSceneSnapshot(selectedScene);
      setSessionStatus("capturing");
      setStatusMessage(`Recording ${selectedScene.sceneId} into session ${payload.session_id}.`);
    } catch (err) {
      setSessionStatus("error");
      setStatusMessage(err instanceof Error ? err.message : "Failed to start recording.");
    }
  }

  async function stopRecording(): Promise<void> {
    if (!currentSessionId || !currentSceneSnapshot) {
      return;
    }
    const sessionId = currentSessionId;
    const startedAtIso = currentRecordingStartIso ?? nowIso();
    const sceneSnapshot = currentSceneSnapshot;
    setSessionStatus("stopping");
    setStatusMessage(`Stopping session ${sessionId} and downloading raw channels.`);
    try {
      await fetch(apiUrl(`/api/session/${sessionId}/stop`), { method: "POST" });
      const artifacts = await fetchRawArtifacts(sessionId);
      setRecordings((prev) => [
        ...prev,
        {
          recordingId: makeId("recording"),
          sceneId: sceneSnapshot.sceneId,
          sceneSnapshot,
          sessionId,
          startedAtIso,
          stoppedAtIso: nowIso(),
          status: "ready",
          artifacts,
        },
      ]);
      setStatusMessage(`Saved recording from ${sceneSnapshot.sceneId} with ${artifacts.channels.length} raw channels.`);
      setSessionStatus("idle");
    } catch (err) {
      setRecordings((prev) => [
        ...prev,
        {
          recordingId: makeId("recording"),
          sceneId: sceneSnapshot.sceneId,
          sceneSnapshot,
          sessionId,
          startedAtIso,
          stoppedAtIso: nowIso(),
          status: "failed",
          error: err instanceof Error ? err.message : "Artifact collection failed.",
        },
      ]);
      setSessionStatus("error");
      setStatusMessage(err instanceof Error ? err.message : "Artifact collection failed.");
    } finally {
      setCurrentSessionId(null);
      setCurrentRecordingStartIso(null);
      setCurrentSceneSnapshot(null);
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
      scenes,
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
            scenes: dataset.scenes.map((scene) => ({
              sceneId: scene.sceneId,
              title: scene.title,
              description: scene.description,
              sessionConfig: scene.sessionConfig,
            })),
            recordings: dataset.recordings.map((recording) => ({
              recordingId: recording.recordingId,
              sceneId: recording.sceneId,
              sessionId: recording.sessionId,
              startedAtIso: recording.startedAtIso,
              stoppedAtIso: recording.stoppedAtIso,
              status: recording.status,
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
      ...scenes.map((scene) => textFile(`scenes/${scene.sceneId}.json`, JSON.stringify(scene, null, 2))),
      ...recordings.flatMap((recording) => {
        const files = [
          textFile(
            `recordings/${recording.recordingId}/metadata.json`,
            JSON.stringify(
              {
                recordingId: recording.recordingId,
                sceneId: recording.sceneId,
                sessionId: recording.sessionId,
                startedAtIso: recording.startedAtIso,
                stoppedAtIso: recording.stoppedAtIso,
                status: recording.status,
                error: recording.error ?? null,
                sceneSnapshot: recording.sceneSnapshot,
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
      <p className="page-copy">
        Build scene definitions, capture multiple runs into one collection set, and export the raw channels together.
      </p>

      <div className="layout data-layout">
        <section className="panel">
          <h2>Collection Set</h2>
          <label htmlFor="collection-id">Collection id</label>
          <input id="collection-id" value={collectionId} onChange={(e) => setCollectionId(e.target.value)} />

          <label htmlFor="collection-title">Title</label>
          <input id="collection-title" value={collectionTitle} onChange={(e) => setCollectionTitle(e.target.value)} />

          <label htmlFor="collection-notes">Notes</label>
          <textarea
            id="collection-notes"
            value={collectionNotes}
            onChange={(e) => setCollectionNotes(e.target.value)}
            rows={4}
          />

          <p className="status">Created: {new Date(createdAtIso).toLocaleString()}</p>
          <div className="actions">
            <button type="button" onClick={() => exportCollection()} disabled={!recordings.length}>
              Export Set
            </button>
          </div>
        </section>

        <section className="panel">
          <h2>{editingSceneId ? "Edit Scene" : "Add Scene"}</h2>

          <label htmlFor="scene-id">Scene id</label>
          <input
            id="scene-id"
            value={sceneForm.sceneId}
            onChange={(e) => setSceneForm((prev) => ({ ...prev, sceneId: e.target.value }))}
          />

          <label htmlFor="scene-title">Scene title</label>
          <input
            id="scene-title"
            value={sceneForm.title}
            onChange={(e) => setSceneForm((prev) => ({ ...prev, title: e.target.value }))}
          />

          <label htmlFor="scene-description">Description</label>
          <textarea
            id="scene-description"
            rows={4}
            value={sceneForm.description}
            onChange={(e) => setSceneForm((prev) => ({ ...prev, description: e.target.value }))}
          />

          <label htmlFor="scene-input-source">Input source</label>
          <select
            id="scene-input-source"
            value={sceneForm.inputSource}
            onChange={(e) =>
              setSceneForm((prev) => ({
                ...prev,
                inputSource: e.target.value as SceneSessionConfig["inputSource"],
              }))
            }
          >
            <option value="simulation">Simulation</option>
            <option value="respeaker_live">ReSpeaker live</option>
          </select>

          {sceneForm.inputSource === "simulation" ? (
            <>
              <label htmlFor="scene-path">Scene config path</label>
              <input
                id="scene-path"
                value={sceneForm.scenePath}
                onChange={(e) => setSceneForm((prev) => ({ ...prev, scenePath: e.target.value }))}
              />

              <label htmlFor="scene-noise-path">Background noise path</label>
              <input
                id="scene-noise-path"
                value={sceneForm.backgroundNoisePath}
                onChange={(e) => setSceneForm((prev) => ({ ...prev, backgroundNoisePath: e.target.value }))}
              />

              <label htmlFor="scene-noise-gain">Background noise gain</label>
              <input
                id="scene-noise-gain"
                type="number"
                min={0}
                max={2}
                step={0.05}
                value={sceneForm.backgroundNoiseGain}
                onChange={(e) =>
                  setSceneForm((prev) => ({ ...prev, backgroundNoiseGain: Number(e.target.value) || 0 }))
                }
              />
            </>
          ) : (
            <>
              <label htmlFor="scene-device-query">Audio device query</label>
              <input
                id="scene-device-query"
                value={sceneForm.audioDeviceQuery}
                onChange={(e) => setSceneForm((prev) => ({ ...prev, audioDeviceQuery: e.target.value }))}
              />

              <label htmlFor="scene-channel-map">Channel map</label>
              <input
                id="scene-channel-map"
                value={sceneForm.channelMap}
                onChange={(e) => setSceneForm((prev) => ({ ...prev, channelMap: e.target.value }))}
              />

              <label htmlFor="scene-sample-rate">Sample rate (Hz)</label>
              <input
                id="scene-sample-rate"
                type="number"
                min={8000}
                max={96000}
                step={1000}
                value={sceneForm.sampleRateHz}
                onChange={(e) => setSceneForm((prev) => ({ ...prev, sampleRateHz: Number(e.target.value) || 16000 }))}
              />
            </>
          )}

          <label htmlFor="scene-processing-mode">Processing mode</label>
          <select
            id="scene-processing-mode"
            value={sceneForm.processingMode}
            onChange={(e) =>
              setSceneForm((prev) => ({ ...prev, processingMode: e.target.value as ProcessingMode }))
            }
          >
            <option value="specific_speaker_enhancement">Specific speaker enhancement</option>
            <option value="localize_and_beamform">Localize and beamform</option>
            <option value="beamform_from_ground_truth">Beamform from ground truth</option>
          </select>

          <label htmlFor="scene-monitor-source">Monitor source</label>
          <select
            id="scene-monitor-source"
            value={sceneForm.monitorSource}
            onChange={(e) =>
              setSceneForm((prev) => ({ ...prev, monitorSource: e.target.value as MonitorSource }))
            }
          >
            <option value="processed">Processed</option>
            <option value="raw_mixed">Raw mixed</option>
          </select>

          <div className="actions">
            <button type="button" onClick={() => saveScene()}>
              {editingSceneId ? "Update Scene" : "Add Scene"}
            </button>
            <button type="button" onClick={() => resetSceneForm()} disabled={!editingSceneId && !sceneForm.title}>
              Clear
            </button>
          </div>
        </section>

        <section className="panel">
          <h2>Scenes</h2>
          {!scenes.length ? (
            <p className="empty-state">No scenes yet.</p>
          ) : (
            <div className="scene-list">
              {scenes.map((scene) => (
                <article
                  key={scene.sceneId}
                  className={`scene-card ${selectedSceneId === scene.sceneId ? "selected" : ""}`.trim()}
                >
                  <button type="button" className="scene-card-select" onClick={() => setSelectedSceneId(scene.sceneId)}>
                    <span className="scene-card-title">{scene.title}</span>
                    <span className="scene-card-id">{scene.sceneId}</span>
                    <span className="scene-card-copy">{scene.description || "No description."}</span>
                  </button>
                  <div className="actions">
                    <button type="button" onClick={() => editScene(scene)}>
                      Edit
                    </button>
                    <button type="button" onClick={() => deleteScene(scene.sceneId)}>
                      Delete
                    </button>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>

      <section className="panel">
        <h2>Recording Runner</h2>
        <p className="status">Status: {sessionStatus}</p>
        <p className="status">
          Selected scene: {selectedScene ? `${selectedScene.title} (${selectedScene.sceneId})` : "none"}
        </p>
        {currentSessionId ? <p className="status">Active session: {currentSessionId}</p> : null}
        <div className="actions">
          <button type="button" onClick={() => void startRecording()} disabled={!canStartRecording}>
            Start Recording
          </button>
          <button type="button" onClick={() => void stopRecording()} disabled={!canStopRecording}>
            Stop Recording
          </button>
        </div>
        {statusMessage ? <p className="status">{statusMessage}</p> : null}
      </section>

      <section className="panel">
        <h2>Recorded Items</h2>
        {!recordings.length ? (
          <p className="empty-state">No recordings captured yet.</p>
        ) : (
          <div className="recording-list">
            {recordings.map((recording) => (
              <article key={recording.recordingId} className="recording-card">
                <h3>{recording.sceneSnapshot.title}</h3>
                <p className="recording-meta">
                  {recording.recordingId} · {recording.status} · {recording.sceneId}
                </p>
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
