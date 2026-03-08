# Mic Array Forwarder

FastAPI backend bridge for the realtime speaker UI demo.

It serves two input modes behind the same REST + WebSocket contract:

- `simulation`: existing scene-driven demo path backed by the realtime pipeline.
- `respeaker_live`: live 4-channel microphone capture path intended for a ReSpeaker-style USB array on the same machine.

The frontend still talks to `http://localhost:8000` and `ws://localhost:8000/ws/session/{id}` either way.

Dockerfile location: `mic_array_forwarder/Dockerfile`.

## How it works

### Simulation mode

- `POST /api/session/start` with `input_source="simulation"` creates the existing `DemoSession`.
- The backend loads a scene config, runs simulated multichannel audio through the realtime pipeline, and streams:
  - `speaker_state` JSON messages
  - `metrics` JSON messages
  - mono audio monitor chunks as binary WebSocket frames
- `GET /api/session/{id}/raw-mix-wav` returns the raw mixed mono WAV used by the UI waveform panel.

### Live ReSpeaker mode

- `POST /api/session/start` with `input_source="respeaker_live"` creates `LiveDemoSession`.
- The backend opens a local multichannel input device with `sounddevice`.
- It captures 4 channels, runs SRP peak tracking for direction estimates, assigns stable pseudo-speaker ids, and sends the same `speaker_state` / `metrics` / audio stream shape the UI already expects.
- The audio returned over the binary WebSocket stream is a mono monitor output. If a speaker is selected, it is delay-and-sum steered toward that direction; otherwise it falls back to a simple mono mix.
- `GET /api/session/{id}/raw-mix-wav` returns a rolling mono capture buffer for the UI waveform panel and playback.

Current live-mode assumptions:

- Linux host with a PortAudio-compatible input device
- 4-channel ReSpeaker-style array
- Device is discoverable by name substring, defaulting to `ReSpeaker`
- Mic geometry is hardcoded for a small 4-mic cross layout and may need one calibration pass on real hardware

## Install

From repo root:

```bash
python -m pip install -U fastapi uvicorn websockets sounddevice
```

If you want the broader backend dependencies used by the demo server image:

```bash
python -m pip install -r mic_array_forwarder/requirements-docker.txt
```

`sounddevice` requires a working PortAudio setup on the host machine.

## Run locally

Start the backend from the repo root:

```bash
PYTHONPATH=. uvicorn mic_array_forwarder.app:app --reload --port 8000
```

Read live mic array messages in a terminal:

```bash
python -m mic_array_forwarder.tools.ws_read --audio-device-query "ReSpeaker"
```

Monitor output source:

```bash
python -m mic_array_forwarder.tools.ws_read --monitor-source raw_mixed
```

Start the frontend in another terminal:

```bash
cd ui
npm install
npm run dev
```

Open `http://localhost:5173`.

## Run with Docker

Build from repo root:

```bash
docker build -f mic_array_forwarder/Dockerfile -t speaker-demo-backend .
```

Run from repo root:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/realtime_demo/output:/app/realtime_demo/output" \
  speaker-demo-backend
```

Use Docker for simulation mode. Live hardware capture is expected to run directly on the host machine, not inside the container.

## UI demo flow

### Simulation

1. Open `http://localhost:5173`.
2. Leave `Input source` as `Simulation`.
3. Use a scene path such as `simulation/simulations/configs/library_scene/library_k1_scene00.json`.
4. Click `Start`.
5. Tap a speaker and use `+` / `-` controls.

### Live ReSpeaker

1. Connect the USB mic array to the same machine running the backend.
2. Start the backend locally, not in Docker.
3. Open `http://localhost:5173`.
4. Change `Input source` to `ReSpeaker live`.
5. Leave `Audio device query` as `ReSpeaker` unless your device enumerates under a different name.
6. Click `Start`.

If startup fails, the most likely cause is device discovery. Change `Audio device query` to a substring that matches the actual PortAudio device name.

## API summary

- `POST /api/session/start`
- `GET /api/session/{id}/status`
- `POST /api/session/{id}/stop`
- `GET /api/session/{id}/raw-mix-wav`
- `GET /api/scenes`
- `WS /ws/session/{id}`

## Start request notes

Important request fields:

- `input_source`: `simulation` or `respeaker_live`
- `scene_config_path`: required for `simulation`, ignored for `respeaker_live`
- `audio_device_query`: optional device-name substring for `respeaker_live`
- `channel_count`: defaults to `4`
- `sample_rate_hz`: defaults to `16000`
- `processing_mode`: same UI processing modes as before

The current UI still sends `separation_mode="mock"` by default for predictable startup.
