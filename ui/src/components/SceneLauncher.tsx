import { useState, type ReactNode } from "react";

import type { MonitorSource } from "../types/contracts";

export type InputSource = "simulation" | "respeaker_live";
export type MicArrayProfile = "respeaker_v3_0457" | "respeaker_xvf3800_0650";
export type LocalizationBackend =
  | "srp_phat_legacy"
  | "srp_phat_localization"
  | "srp_phat_mvdr_refine"
  | "capon_1src"
  | "capon_multisrc"
  | "capon_mvdr_refine_1src"
  | "music_1src";
export type BeamformingMode = "mvdr_fd" | "sd_mvdr_fd" | "gsc_fd" | "delay_sum";
export type OwnVoiceSuppressionMode = "off" | "lcmv_null_hysteresis" | "soft_output_gate";
export type TrackingMode = "doa_centroid_v1";
export type CentroidAssociationMode = "hard_window" | "gaussian";
export type OutputEnhancerMode = "off" | "wiener";

export type SessionLaunchConfig = {
  inputSource: InputSource;
  scenePath: string;
  backgroundNoisePath: string;
  backgroundNoiseGain: number;
  useGroundTruthLocation: boolean;
  useGroundTruthSpeakerSources: boolean;
  audioDeviceQuery: string;
  monitorSource: MonitorSource;
  livePlaybackEnabled: boolean;
  sampleRateHz: number;
  micArrayProfile: MicArrayProfile;
  fastPath: {
    localizationBackend: LocalizationBackend;
    localizationHopMs: number;
    localizationWindowMs: number;
    localizationOverlap: number;
    localizationFreqLowHz: number;
    localizationFreqHighHz: number;
    localizationVadEnabled: boolean;
    assumeSingleSpeaker: boolean;
    beamformingMode: BeamformingMode;
    ownVoiceSuppressionMode: OwnVoiceSuppressionMode;
    outputEnhancerMode: OutputEnhancerMode;
    postfilterEnabled: boolean;
  };
  slowPath: {
    enabled: boolean;
    trackingMode: TrackingMode;
    singleActive: boolean;
    speakerHistorySize: number;
    speakerActivationMinPredictions: number;
    speakerMatchWindowDeg: number;
    centroidAssociationMode: CentroidAssociationMode;
    centroidAssociationSigmaDeg: number;
    centroidAssociationMinScore: number;
    longMemoryEnabled: boolean;
    longMemoryWindowMs: number;
  };
  sessionOverrides: Record<string, unknown>;
  fastPathOverrides: Record<string, unknown>;
  slowPathOverrides: Record<string, unknown>;
};

type Props = {
  status: string;
  hasActiveSession: boolean;
  defaultScenePath: string;
  defaultBackgroundNoisePath: string;
  defaultBackgroundNoiseGain: number;
  onStart: (config: SessionLaunchConfig) => void;
  onStop: () => void;
  onKillRun: () => void;
  canKillRun: boolean;
  onStopActiveSession: () => void;
  onDownloadWav: () => void;
  canDownloadWav: boolean;
  latencyMs: number;
  onLatencyMsChange: (latencyMs: number) => void;
  monitorSource: MonitorSource;
  onMonitorSourceChange: (source: MonitorSource) => void;
};

const LOCALIZATION_BACKENDS: Array<{ value: LocalizationBackend; label: string }> = [
  { value: "srp_phat_legacy", label: "SRP-PHAT legacy" },
  { value: "capon_1src", label: "Capon 1src" },
  { value: "capon_mvdr_refine_1src", label: "Capon MVDR refine 1src" },
  { value: "capon_multisrc", label: "Capon multisrc" },
  { value: "srp_phat_localization", label: "SRP-PHAT" },
  { value: "srp_phat_mvdr_refine", label: "SRP MVDR refine" },
  { value: "music_1src", label: "MUSIC 1src" },
];

const BEAMFORMING_MODES: Array<{ value: BeamformingMode; label: string }> = [
  { value: "delay_sum", label: "Delay-and-sum" },
  { value: "mvdr_fd", label: "MVDR FD" },
  { value: "sd_mvdr_fd", label: "Superdirective MVDR FD" },
  { value: "gsc_fd", label: "GSC FD" },
];

const OWN_VOICE_SUPPRESSION_MODES: Array<{ value: OwnVoiceSuppressionMode; label: string }> = [
  { value: "lcmv_null_hysteresis", label: "LCMV null + hysteresis" },
  { value: "soft_output_gate", label: "Soft output gate" },
  { value: "off", label: "Off" },
];

const OUTPUT_ENHANCER_MODES: Array<{ value: OutputEnhancerMode; label: string }> = [
  { value: "off", label: "Off" },
  { value: "wiener", label: "Wiener" },
];

type CollapsibleSectionProps = {
  title: string;
  expanded: boolean;
  onToggle: () => void;
  children: ReactNode;
};

function CollapsibleSection({ title, expanded, onToggle, children }: CollapsibleSectionProps) {
  return (
    <div className={`launcher-section ${expanded ? "expanded" : "collapsed"}`.trim()}>
      <button
        type="button"
        className="launcher-section-toggle"
        aria-expanded={expanded}
        aria-label={title}
        onClick={onToggle}
      >
        <span className={`launcher-section-arrow ${expanded ? "expanded" : ""}`.trim()} aria-hidden="true">
          ▾
        </span>
        <span className="launcher-section-title">{title}</span>
      </button>
      {expanded ? <div className="launcher-section-body">{children}</div> : null}
    </div>
  );
}

export function SceneLauncher({
  status,
  hasActiveSession,
  defaultScenePath,
  defaultBackgroundNoisePath,
  defaultBackgroundNoiseGain,
  onStart,
  onStop,
  onKillRun,
  canKillRun,
  onStopActiveSession,
  onDownloadWav,
  canDownloadWav,
  latencyMs,
  onLatencyMsChange,
  monitorSource,
  onMonitorSourceChange,
}: Props) {
  const [inputSource, setInputSource] = useState<InputSource | null>(null);
  const [localizationBackend, setLocalizationBackend] = useState<LocalizationBackend>("capon_1src");
  const [localizationHopMs, setLocalizationHopMs] = useState(95);
  const [localizationWindowMs, setLocalizationWindowMs] = useState(300);
  const [localizationOverlap, setLocalizationOverlap] = useState(0.2);
  const [localizationFreqLowHz, setLocalizationFreqLowHz] = useState(1200);
  const [localizationFreqHighHz, setLocalizationFreqHighHz] = useState(5400);
  const [localizationVadEnabled, setLocalizationVadEnabled] = useState(true);
  const [assumeSingleSpeaker, setAssumeSingleSpeaker] = useState(true);
  const [beamformingMode, setBeamformingMode] = useState<BeamformingMode>("mvdr_fd");
  const [ownVoiceSuppressionMode, setOwnVoiceSuppressionMode] = useState<OwnVoiceSuppressionMode>("lcmv_null_hysteresis");
  const [outputEnhancerMode, setOutputEnhancerMode] = useState<OutputEnhancerMode>("off");
  const [postfilterEnabled, setPostfilterEnabled] = useState(true);
  const [livePlaybackEnabled, setLivePlaybackEnabled] = useState(true);
  const [slowPathEnabled, setSlowPathEnabled] = useState(false);
  const [singleActive, setSingleActive] = useState(true);
  const [speakerHistorySize, setSpeakerHistorySize] = useState(8);
  const [speakerActivationMinPredictions, setSpeakerActivationMinPredictions] = useState(3);
  const [speakerMatchWindowDeg, setSpeakerMatchWindowDeg] = useState(30);
  const [centroidAssociationMode, setCentroidAssociationMode] = useState<CentroidAssociationMode>("hard_window");
  const [centroidAssociationSigmaDeg, setCentroidAssociationSigmaDeg] = useState(10);
  const [centroidAssociationMinScore, setCentroidAssociationMinScore] = useState(0.15);
  const [longMemoryEnabled, setLongMemoryEnabled] = useState(false);
  const [longMemoryWindowMs, setLongMemoryWindowMs] = useState(60000);
  const [scenePath, setScenePath] = useState(defaultScenePath);
  const [backgroundNoisePath, setBackgroundNoisePath] = useState(defaultBackgroundNoisePath);
  const [backgroundNoiseGain, setBackgroundNoiseGain] = useState(defaultBackgroundNoiseGain);
  const [useGroundTruthLocation, setUseGroundTruthLocation] = useState(false);
  const [useGroundTruthSpeakerSources, setUseGroundTruthSpeakerSources] = useState(false);
  const [audioDeviceQuery, setAudioDeviceQuery] = useState("ReSpeaker");
  const [sampleRateHz, setSampleRateHz] = useState(48000);
  const [micArrayProfile, setMicArrayProfile] = useState<MicArrayProfile>("respeaker_xvf3800_0650");
  const [sessionOverridesText, setSessionOverridesText] = useState("{}");
  const [fastPathOverridesText, setFastPathOverridesText] = useState("{}");
  const [slowPathOverridesText, setSlowPathOverridesText] = useState("{}");
  const [overrideError, setOverrideError] = useState("");
  const [localizationExpanded, setLocalizationExpanded] = useState(true);
  const [beamformingExpanded, setBeamformingExpanded] = useState(true);
  const [postProcExpanded, setPostProcExpanded] = useState(true);
  const [slowPathExpanded, setSlowPathExpanded] = useState(false);
  const [advancedOverridesExpanded, setAdvancedOverridesExpanded] = useState(false);
  const isBusy = hasActiveSession || status === "running" || status === "starting" || status === "stopping";
  const showSimulationSettings = inputSource === "simulation";
  const showLiveSettings = inputSource === "respeaker_live";
  const canStart = inputSource !== null && !isBusy;
  const canStop = isBusy;

  function applyLatency(v: number): void {
    const clamped = Math.max(80, Math.min(2000, Math.round(v)));
    onLatencyMsChange(clamped);
  }

  function parseOverrideObject(label: string, text: string): Record<string, unknown> {
    const trimmed = text.trim();
    if (!trimmed) {
      return {};
    }
    const parsed = JSON.parse(trimmed) as unknown;
    if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
      throw new Error(`${label} must be a JSON object`);
    }
    return parsed as Record<string, unknown>;
  }

  return (
    <section className="panel">
      <h2>Scene Launcher</h2>
      <div className="mode-picker" role="group" aria-label="Input mode">
        <button
          type="button"
          className={`mode-card ${showSimulationSettings ? "selected" : ""}`}
          aria-pressed={showSimulationSettings}
          disabled={isBusy}
          onClick={() => setInputSource("simulation")}
        >
          <span className="mode-card-title">Simulation</span>
          <span className="mode-card-copy">Scene file plus optional background noise.</span>
        </button>
        <button
          type="button"
          className={`mode-card ${showLiveSettings ? "selected" : ""}`}
          aria-pressed={showLiveSettings}
          disabled={isBusy}
          onClick={() => setInputSource("respeaker_live")}
        >
          <span className="mode-card-title">ReSpeaker Live</span>
          <span className="mode-card-copy">Direct capture from the local USB microphone array.</span>
        </button>
      </div>

      <div className="actions actions-persistent">
        <button onClick={onKillRun} disabled={!canKillRun}>
          Kill Current Run
        </button>
        <button onClick={onStopActiveSession}>Stop Active Session</button>
      </div>

      {!inputSource ? (
        <p className="launcher-hint">Choose a mode to reveal the relevant session settings.</p>
      ) : (
        <div className="launcher-settings">
          <div className="switch-row">
            <span>Live playback</span>
            <button
              aria-label="Live playback"
              aria-checked={livePlaybackEnabled}
              className={`switch ${livePlaybackEnabled ? "on" : "off"}`.trim()}
              disabled={isBusy}
              role="switch"
              type="button"
              onClick={() => setLivePlaybackEnabled((prev) => !prev)}
            >
              <span className="switch-thumb" />
            </button>
          </div>

          <CollapsibleSection
            title="Localization"
            expanded={localizationExpanded}
            onToggle={() => setLocalizationExpanded((prev) => !prev)}
          >
              <label htmlFor="localization-backend">Localization backend</label>
              <select
                id="localization-backend"
                aria-label="Localization backend"
                value={localizationBackend}
                disabled={isBusy}
                onChange={(e) => setLocalizationBackend(e.target.value as LocalizationBackend)}
              >
                {LOCALIZATION_BACKENDS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              <div className="switch-row">
                <span>Single-speaker assumption</span>
                <button
                  aria-label="Single-speaker assumption"
                  aria-checked={assumeSingleSpeaker}
                  className={`switch ${assumeSingleSpeaker ? "on" : "off"}`.trim()}
                  disabled={isBusy}
                  role="switch"
                  type="button"
                  onClick={() => setAssumeSingleSpeaker((prev) => !prev)}
                >
                  <span className="switch-thumb" />
                </button>
              </div>

              <div className="switch-row">
                <span>Localization VAD</span>
                <button
                  aria-label="Localization VAD"
                  aria-checked={localizationVadEnabled}
                  className={`switch ${localizationVadEnabled ? "on" : "off"}`.trim()}
                  disabled={isBusy}
                  role="switch"
                  type="button"
                  onClick={() => setLocalizationVadEnabled((prev) => !prev)}
                >
                  <span className="switch-thumb" />
                </button>
              </div>

              <label htmlFor="localization-hop-ms">Localization hop (ms)</label>
              <input
                id="localization-hop-ms"
                aria-label="Localization hop (ms)"
                type="number"
                min={10}
                max={500}
                step={10}
                value={localizationHopMs}
                disabled={isBusy}
                onChange={(e) => {
                  const nextHop = Math.max(10, Math.min(500, Number(e.target.value) || 10));
                  setLocalizationHopMs(nextHop);
                  setLocalizationWindowMs((prev) => Math.max(prev, nextHop));
                }}
              />

              <label htmlFor="localization-window-ms">Localization window (ms)</label>
              <input
                id="localization-window-ms"
                aria-label="Localization window (ms)"
                type="number"
                min={20}
                max={2000}
                step={10}
                value={localizationWindowMs}
                disabled={isBusy}
                onChange={(e) => {
                  const nextWindow = Math.min(2000, Number(e.target.value) || 160);
                  setLocalizationWindowMs(Math.max(Math.max(20, localizationHopMs), nextWindow));
                }}
              />

              <label htmlFor="localization-overlap">Localization overlap</label>
              <input
                id="localization-overlap"
                aria-label="Localization overlap"
                type="number"
                min={0}
                max={0.99}
                step={0.01}
                value={localizationOverlap}
                disabled={isBusy}
                onChange={(e) => {
                  const nextOverlap = Number(e.target.value);
                  setLocalizationOverlap(Math.max(0, Math.min(0.99, Number.isFinite(nextOverlap) ? nextOverlap : 0)));
                }}
              />

              <label htmlFor="localization-freq-low-hz">Localization freq low (Hz)</label>
              <input
                id="localization-freq-low-hz"
                aria-label="Localization freq low (Hz)"
                type="number"
                min={0}
                max={24000}
                step={50}
                value={localizationFreqLowHz}
                disabled={isBusy}
                onChange={(e) => {
                  const nextLow = Math.max(0, Math.min(24000, Number(e.target.value) || 0));
                  setLocalizationFreqLowHz(nextLow);
                  setLocalizationFreqHighHz((prev) => Math.max(prev, nextLow));
                }}
              />

              <label htmlFor="localization-freq-high-hz">Localization freq high (Hz)</label>
              <input
                id="localization-freq-high-hz"
                aria-label="Localization freq high (Hz)"
                type="number"
                min={0}
                max={24000}
                step={50}
                value={localizationFreqHighHz}
                disabled={isBusy}
                onChange={(e) => {
                  const nextHigh = Math.max(0, Math.min(24000, Number(e.target.value) || 0));
                  setLocalizationFreqHighHz(Math.max(nextHigh, localizationFreqLowHz));
                }}
              />
          </CollapsibleSection>

          <CollapsibleSection
            title="Beamforming"
            expanded={beamformingExpanded}
            onToggle={() => setBeamformingExpanded((prev) => !prev)}
          >
              <label htmlFor="beamforming-mode">Beamforming mode</label>
              <select
                id="beamforming-mode"
                aria-label="Beamforming mode"
                value={beamformingMode}
                disabled={isBusy}
                onChange={(e) => setBeamformingMode(e.target.value as BeamformingMode)}
              >
                {BEAMFORMING_MODES.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              <label htmlFor="own-voice-suppression-mode">Own voice suppression</label>
              <select
                id="own-voice-suppression-mode"
                aria-label="Own voice suppression"
                value={ownVoiceSuppressionMode}
                disabled={isBusy}
                onChange={(e) => setOwnVoiceSuppressionMode(e.target.value as OwnVoiceSuppressionMode)}
              >
                {OWN_VOICE_SUPPRESSION_MODES.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
          </CollapsibleSection>

          <CollapsibleSection
            title="Post-proc"
            expanded={postProcExpanded}
            onToggle={() => setPostProcExpanded((prev) => !prev)}
          >
              <label htmlFor="output-enhancer-mode">Output enhancer mode</label>
              <select
                id="output-enhancer-mode"
                aria-label="Output enhancer mode"
                value={outputEnhancerMode}
                disabled={isBusy}
                onChange={(e) => setOutputEnhancerMode(e.target.value as OutputEnhancerMode)}
              >
                {OUTPUT_ENHANCER_MODES.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              <div className="switch-row">
                <span>Postfilter enabled</span>
                <button
                  aria-label="Postfilter enabled"
                  aria-checked={postfilterEnabled}
                  className={`switch ${postfilterEnabled ? "on" : "off"}`.trim()}
                  disabled={isBusy}
                  role="switch"
                  type="button"
                  onClick={() => setPostfilterEnabled((prev) => !prev)}
                >
                  <span className="switch-thumb" />
                </button>
              </div>
          </CollapsibleSection>

          <CollapsibleSection
            title="Slow path"
            expanded={slowPathExpanded}
            onToggle={() => setSlowPathExpanded((prev) => !prev)}
          >
              <div className="switch-row">
                <span>Enable slow path</span>
                <button
                  aria-label="Enable slow path"
                  aria-checked={slowPathEnabled}
                  className={`switch ${slowPathEnabled ? "on" : "off"}`.trim()}
                  disabled={isBusy}
                  role="switch"
                  type="button"
                  onClick={() => setSlowPathEnabled((prev) => !prev)}
                >
                  <span className="switch-thumb" />
                </button>
              </div>

              <div className="switch-row">
                <span>Single active speaker</span>
                <button
                  id="single-active-speaker"
                  aria-label="Single active speaker"
                  aria-checked={singleActive}
                  className={`switch ${singleActive ? "on" : "off"}`.trim()}
                  disabled={isBusy}
                  role="switch"
                  type="button"
                  onClick={() => setSingleActive((prev) => !prev)}
                >
                  <span className="switch-thumb" />
                </button>
              </div>

              <div className="switch-row">
                <span>Long memory</span>
                <button
                  aria-label="Long memory"
                  aria-checked={longMemoryEnabled}
                  className={`switch ${longMemoryEnabled ? "on" : "off"}`.trim()}
                  disabled={isBusy}
                  role="switch"
                  type="button"
                  onClick={() => setLongMemoryEnabled((prev) => !prev)}
                >
                  <span className="switch-thumb" />
                </button>
              </div>

              <label htmlFor="speaker-history-size">Speaker centroid history (M)</label>
              <input
                id="speaker-history-size"
                aria-label="Speaker centroid history (M)"
                type="number"
                min={1}
                max={64}
                step={1}
                value={speakerHistorySize}
                disabled={isBusy}
                onChange={(e) => {
                  const nextHistory = Math.max(1, Math.min(64, Number(e.target.value) || 1));
                  setSpeakerHistorySize(nextHistory);
                  setSpeakerActivationMinPredictions((prev) => Math.min(prev, nextHistory));
                }}
              />

              <label htmlFor="speaker-activation-min-predictions">Speaker activation observations (N)</label>
              <input
                id="speaker-activation-min-predictions"
                aria-label="Speaker activation observations (N)"
                type="number"
                min={1}
                max={64}
                step={1}
                value={speakerActivationMinPredictions}
                disabled={isBusy}
                onChange={(e) => {
                  const nextCount = Math.max(1, Math.min(64, Number(e.target.value) || 1));
                  setSpeakerActivationMinPredictions(Math.min(nextCount, speakerHistorySize));
                }}
              />

              <label htmlFor="speaker-match-window-deg">Speaker match window (deg)</label>
              <input
                id="speaker-match-window-deg"
                aria-label="Speaker match window (deg)"
                type="number"
                min={1}
                max={180}
                step={1}
                value={speakerMatchWindowDeg}
                disabled={isBusy}
                onChange={(e) => setSpeakerMatchWindowDeg(Math.max(1, Math.min(180, Number(e.target.value) || 1)))}
              />

              <label htmlFor="centroid-association-mode">Centroid association</label>
              <select
                id="centroid-association-mode"
                aria-label="Centroid association"
                value={centroidAssociationMode}
                disabled={isBusy}
                onChange={(e) => setCentroidAssociationMode(e.target.value as CentroidAssociationMode)}
              >
                <option value="hard_window">Hard window</option>
                <option value="gaussian">Gaussian</option>
              </select>

              <label htmlFor="centroid-association-sigma-deg">Centroid sigma (deg)</label>
              <input
                id="centroid-association-sigma-deg"
                aria-label="Centroid sigma (deg)"
                type="number"
                min={1}
                max={90}
                step={1}
                value={centroidAssociationSigmaDeg}
                disabled={isBusy}
                onChange={(e) => setCentroidAssociationSigmaDeg(Math.max(1, Math.min(90, Number(e.target.value) || 1)))}
              />

              <label htmlFor="centroid-association-min-score">Centroid min score</label>
              <input
                id="centroid-association-min-score"
                aria-label="Centroid min score"
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={centroidAssociationMinScore}
                disabled={isBusy}
                onChange={(e) => {
                  const nextValue = Number(e.target.value);
                  setCentroidAssociationMinScore(Math.max(0, Math.min(1, Number.isFinite(nextValue) ? nextValue : 0)));
                }}
              />

              <label htmlFor="long-memory-window-ms">Long-memory window (ms)</label>
              <input
                id="long-memory-window-ms"
                aria-label="Long-memory window (ms)"
                type="number"
                min={1000}
                max={120000}
                step={1000}
                value={longMemoryWindowMs}
                disabled={isBusy}
                onChange={(e) => setLongMemoryWindowMs(Math.max(1000, Math.min(120000, Number(e.target.value) || 1000)))}
              />
          </CollapsibleSection>

          <CollapsibleSection
            title="Advanced overrides"
            expanded={advancedOverridesExpanded}
            onToggle={() => setAdvancedOverridesExpanded((prev) => !prev)}
          >
              <label htmlFor="session-overrides-json">Session overrides JSON</label>
              <textarea
                id="session-overrides-json"
                aria-label="Session overrides JSON"
                value={sessionOverridesText}
                disabled={isBusy}
                rows={4}
                onChange={(e) => setSessionOverridesText(e.target.value)}
              />

              <label htmlFor="fast-path-overrides-json">Fast path overrides JSON</label>
              <textarea
                id="fast-path-overrides-json"
                aria-label="Fast path overrides JSON"
                value={fastPathOverridesText}
                disabled={isBusy}
                rows={4}
                onChange={(e) => setFastPathOverridesText(e.target.value)}
              />

              <label htmlFor="slow-path-overrides-json">Slow path overrides JSON</label>
              <textarea
                id="slow-path-overrides-json"
                aria-label="Slow path overrides JSON"
                value={slowPathOverridesText}
                disabled={isBusy}
                rows={4}
                onChange={(e) => setSlowPathOverridesText(e.target.value)}
              />
              {overrideError ? <p className="launcher-hint">{overrideError}</p> : null}
          </CollapsibleSection>

          {showSimulationSettings && (
            <>
              <label htmlFor="scene-path">Scene path</label>
              <input id="scene-path" aria-label="Scene path" value={scenePath} disabled={isBusy} onChange={(e) => setScenePath(e.target.value)} />

              <label htmlFor="background-noise-path">Background noise path</label>
              <input
                id="background-noise-path"
                aria-label="Background noise path"
                value={backgroundNoisePath}
                disabled={isBusy}
                onChange={(e) => setBackgroundNoisePath(e.target.value)}
              />

              <label htmlFor="background-noise-gain">Background noise gain</label>
              <input
                id="background-noise-gain"
                aria-label="Background noise gain"
                type="number"
                min={0}
                max={2}
                step={0.01}
                value={backgroundNoiseGain}
                disabled={isBusy}
                onChange={(e) => setBackgroundNoiseGain(Math.max(0, Math.min(2, Number(e.target.value) || 0)))}
              />

              <div className="switch-row">
                <span>Use ground truth location</span>
                <button
                  aria-label="Use ground truth location"
                  aria-checked={useGroundTruthLocation}
                  className={`switch ${useGroundTruthLocation ? "on" : "off"}`.trim()}
                  disabled={isBusy}
                  role="switch"
                  type="button"
                  onClick={() => setUseGroundTruthLocation((prev) => !prev)}
                >
                  <span className="switch-thumb" />
                </button>
              </div>

              <div className="switch-row">
                <span>Use ground truth speaker sources</span>
                <button
                  aria-label="Use ground truth speaker sources"
                  aria-checked={useGroundTruthSpeakerSources}
                  className={`switch ${useGroundTruthSpeakerSources ? "on" : "off"}`.trim()}
                  disabled={isBusy}
                  role="switch"
                  type="button"
                  onClick={() => setUseGroundTruthSpeakerSources((prev) => !prev)}
                >
                  <span className="switch-thumb" />
                </button>
              </div>
            </>
          )}

          {showLiveSettings && (
            <>
              <label htmlFor="audio-device-query">Audio device query</label>
              <input
                id="audio-device-query"
                aria-label="Audio device query"
                value={audioDeviceQuery}
                disabled={isBusy}
                onChange={(e) => setAudioDeviceQuery(e.target.value)}
              />

              <label htmlFor="sample-rate-hz">Sample rate (Hz)</label>
              <input
                id="sample-rate-hz"
                aria-label="Sample rate (Hz)"
                type="number"
                min={8000}
                max={96000}
                step={1000}
                value={sampleRateHz}
                disabled={isBusy}
                onChange={(e) => setSampleRateHz(Math.max(8000, Math.min(96000, Number(e.target.value) || 8000)))}
              />

              <label htmlFor="mic-array-profile">Mic array profile</label>
              <select
                id="mic-array-profile"
                aria-label="Mic array profile"
                value={micArrayProfile}
                disabled={isBusy}
                onChange={(e) => setMicArrayProfile(e.target.value as MicArrayProfile)}
              >
                <option value="respeaker_xvf3800_0650">ReSpeaker XVF3800 65mm</option>
                <option value="respeaker_v3_0457">ReSpeaker v3 45.7mm</option>
              </select>
            </>
          )}

          <label htmlFor="launcher-monitor-source">Monitor source</label>
          <select
            id="launcher-monitor-source"
            aria-label="Monitor source"
            value={monitorSource}
            disabled={isBusy}
            onChange={(e) => onMonitorSourceChange(e.target.value as MonitorSource)}
          >
            <option value="processed">Processed</option>
            <option value="raw_mixed">Raw mixed</option>
          </select>

          <label htmlFor="latency-ms">Playback latency target (ms)</label>
          <input
            id="latency-ms"
            aria-label="Playback latency target (ms)"
            type="number"
            min={80}
            max={2000}
            step={10}
            value={latencyMs}
            disabled={isBusy}
            onChange={(e) => applyLatency(Number(e.target.value) || latencyMs)}
          />

          <div className="actions">
            {showLiveSettings && isBusy ? (
              <span className="status-chip">Recording mode</span>
            ) : (
              <button
                onClick={() => {
                  try {
                    const sessionOverrides = parseOverrideObject("Session overrides JSON", sessionOverridesText);
                    const fastPathOverrides = parseOverrideObject("Fast path overrides JSON", fastPathOverridesText);
                    const slowPathOverrides = parseOverrideObject("Slow path overrides JSON", slowPathOverridesText);
                    setOverrideError("");
                    onStart({
                      inputSource,
                      scenePath,
                      backgroundNoisePath,
                      backgroundNoiseGain,
                      useGroundTruthLocation,
                      useGroundTruthSpeakerSources,
                      audioDeviceQuery,
                      monitorSource,
                      livePlaybackEnabled,
                      sampleRateHz,
                      micArrayProfile,
                      fastPath: {
                        localizationBackend,
                        localizationHopMs,
                        localizationWindowMs,
                        localizationOverlap,
                        localizationFreqLowHz,
                        localizationFreqHighHz,
                        localizationVadEnabled,
                        assumeSingleSpeaker,
                        beamformingMode,
                        ownVoiceSuppressionMode,
                        outputEnhancerMode,
                        postfilterEnabled,
                      },
                      slowPath: {
                        enabled: slowPathEnabled,
                        trackingMode: "doa_centroid_v1",
                        singleActive,
                        speakerHistorySize,
                        speakerActivationMinPredictions,
                        speakerMatchWindowDeg,
                        centroidAssociationMode,
                        centroidAssociationSigmaDeg,
                        centroidAssociationMinScore,
                        longMemoryEnabled,
                        longMemoryWindowMs,
                      },
                      sessionOverrides,
                      fastPathOverrides,
                      slowPathOverrides,
                    });
                  } catch (error) {
                    setOverrideError(error instanceof Error ? error.message : "Invalid JSON override");
                  }
                }}
                disabled={!canStart}
              >
                Start
              </button>
            )}
            <button onClick={onStop} disabled={!canStop}>
              Stop
            </button>
            <button onClick={onDownloadWav} disabled={!canDownloadWav}>
              Download WAV
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
