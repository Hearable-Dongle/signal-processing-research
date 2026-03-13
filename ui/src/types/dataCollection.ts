export type RawChannelFile = {
  channelIndex: number;
  filename: string;
  bytes: Uint8Array;
};

export type RecordingArtifactManifest = {
  sampleRateHz: number;
  channels: RawChannelFile[];
};

export type AnnotatedSpeaker = {
  speakerName: string;
  directionDeg: number;
};

export type RecordingStatus = "capturing" | "ready" | "incomplete" | "failed";

export type RecordingEntry = {
  recordingId: string;
  sessionId: string;
  startedAtIso: string;
  stoppedAtIso: string;
  status: RecordingStatus;
  deviceName: string;
  micArrayProfile: string;
  notes: string;
  speakers: AnnotatedSpeaker[];
  error?: string;
  artifacts?: RecordingArtifactManifest;
};

export type DataCollectionSet = {
  collectionId: string;
  title: string;
  notes: string;
  createdAtIso: string;
  deviceName: string;
  micArrayProfile: string;
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
