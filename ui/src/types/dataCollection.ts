import type { MonitorSource, ProcessingMode } from "./contracts";

export type InputSource = "simulation" | "respeaker_live";

export type SceneSessionConfig = {
  inputSource: InputSource;
  scenePath: string;
  backgroundNoisePath: string;
  backgroundNoiseGain: number;
  audioDeviceQuery: string;
  monitorSource: MonitorSource;
  sampleRateHz: number;
  channelMap: string;
  processingMode: ProcessingMode;
};

export type SceneDefinition = {
  sceneId: string;
  title: string;
  description: string;
  sessionConfig: SceneSessionConfig;
};

export type RawChannelFile = {
  channelIndex: number;
  filename: string;
  bytes: Uint8Array;
};

export type RecordingArtifactManifest = {
  sampleRateHz: number;
  channels: RawChannelFile[];
};

export type RecordingStatus = "capturing" | "ready" | "incomplete" | "failed";

export type RecordingEntry = {
  recordingId: string;
  sceneId: string;
  sceneSnapshot: SceneDefinition;
  sessionId: string;
  startedAtIso: string;
  stoppedAtIso: string;
  status: RecordingStatus;
  error?: string;
  artifacts?: RecordingArtifactManifest;
};

export type DataCollectionSet = {
  collectionId: string;
  title: string;
  notes: string;
  createdAtIso: string;
  scenes: SceneDefinition[];
  recordings: RecordingEntry[];
};

export type RawChannelDescriptor = {
  channel_index: number;
  filename: string;
};

export type RawChannelsResponse = {
  session_id: string;
  sample_rate_hz: number;
  channel_count: number;
  channels: RawChannelDescriptor[];
};
