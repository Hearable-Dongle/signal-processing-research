## Commit Message
`Revamp UI flow and fix frontend session transport`

## Summary
- Gate scene launcher settings behind an explicit Simulation vs ReSpeaker Live mode picker
- Add beamforming weight visibility, a polar beam visualization, and smoother waveform traces
- Make metrics collapse into a horizontal rail and route API and WS traffic through the frontend origin

## Changes
- UI: new mode-picker flow in the launcher with always-visible kill controls
- UI: new `BeamformerViz` plus speaker-stage weight display and updated metrics panel behavior
- UI: waveform rendering switched from filled envelope blocks to line traces
- Frontend transport: same-origin `/api` and `/ws` defaults with Vite dev proxy support

## Testing
- `cd ui && npm test`
- `cd ui && npm run build`
