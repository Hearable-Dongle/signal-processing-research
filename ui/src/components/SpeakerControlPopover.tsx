type Props = {
  speakerId: number;
  deltaDb: number;
  onAdjust: (speakerId: number, step: 1 | -1) => void;
};

export function SpeakerControlPopover({ speakerId, deltaDb, onAdjust }: Props) {
  return (
    <div className="popover" role="dialog" aria-label={`speaker-${speakerId}-control`}>
      <h4>Speaker {speakerId}</h4>
      <p>Relative gain: {deltaDb.toFixed(1)} dB</p>
      <div className="actions">
        <button onClick={() => onAdjust(speakerId, -1)}>-</button>
        <button onClick={() => onAdjust(speakerId, 1)}>+</button>
      </div>
    </div>
  );
}
