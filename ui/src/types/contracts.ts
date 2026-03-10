export const SCHEMA_VERSION = "v1" as const;

export type SeparationMode = "auto" | "mock" | "single_dominant_no_separator";

export type ProcessingMode =
  | "specific_speaker_enhancement"
  | "localize_and_beamform"
  | "beamform_from_ground_truth";

export type MonitorSource = "processed" | "raw_mixed";

export type Speaker = {
  speaker_id: number;
  direction_degrees: number;
  confidence: number;
  active: boolean;
  activity_confidence: number;
  gain_weight: number;
};

export type GroundTruthSpeaker = {
  source_id: number;
  direction_degrees: number;
};

export type SpeakerStateMessage = {
  schema_version: "v1";
  type: "speaker_state";
  timestamp_ms: number;
  speakers: Speaker[];
  ground_truth: GroundTruthSpeaker[];
};

export type MetricsMessage = {
  schema_version: "v1";
  type: "metrics";
  timestamp_ms: number;
  fast_rtf: number;
  slow_rtf: number;
  fast_stage_avg_ms: Record<string, number>;
  slow_stage_avg_ms: Record<string, number>;
  startup_lock_ms: number;
  reacquire_catchup_ms_median: number;
  nearest_change_catchup_ms_median: number;
};

export type SessionEventMessage = {
  schema_version: "v1";
  type: "session_event";
  event: "started" | "stopped" | "error";
  detail: string;
  timestamp_ms: number;
};

export type ErrorMessage = {
  schema_version: "v1";
  type: "error";
  error: string;
  timestamp_ms: number;
};

export type ServerMessage = SpeakerStateMessage | MetricsMessage | SessionEventMessage | ErrorMessage;

export type SelectSpeakerMessage = {
  schema_version: "v1";
  type: "select_speaker";
  speaker_id: number;
};

export type AdjustSpeakerGainMessage = {
  schema_version: "v1";
  type: "adjust_speaker_gain";
  speaker_id: number;
  delta_db_step: 1 | -1;
};

export type ClearFocusMessage = {
  schema_version: "v1";
  type: "clear_focus";
};

export type SetMonitorSourceMessage = {
  schema_version: "v1";
  type: "set_monitor_source";
  monitor_source: MonitorSource;
};

export type StopSessionMessage = {
  schema_version: "v1";
  type: "stop_session";
};

export type ClientMessage =
  | SelectSpeakerMessage
  | AdjustSpeakerGainMessage
  | ClearFocusMessage
  | SetMonitorSourceMessage
  | StopSessionMessage;
