# Realtime Demo Server

FastAPI backend bridge for the realtime speaker UI demo.

Dockerfile location: `realtime_demo_server/Dockerfile` (service-owned, keeps backend deploy assets together).

## Install

From repo root:

```bash
python -m pip install -U fastapi uvicorn websockets
```

## Start backend

```bash
PYTHONPATH=. uvicorn realtime_demo_server.app:app --reload --port 8000
```

## Run backend with Docker

Build from repo root:

```bash
docker build -f realtime_demo_server/Dockerfile -t speaker-demo-backend .
```

Run from repo root:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/realtime_demo/output:/app/realtime_demo/output" \
  speaker-demo-backend
```

## Start frontend

```bash
cd ui
npm install
npm run dev
```

## Demo flow

1. Open `http://localhost:5173`.
2. Use a scene path, for example:
   `simulation/simulations/configs/library_scene/library_k1_scene00.json`
3. Click start.
4. Tap a speaker and use `+` / `-` controls.

Notes:
- UI uses `separation_mode=\"mock\"` by default for reliable startup.
- To try real backends, call `POST /api/session/start` with `\"separation_mode\": \"auto\"`.

## API summary

- `POST /api/session/start`
- `GET /api/session/{id}/status`
- `POST /api/session/{id}/stop`
- `GET /api/session/{id}/raw-mix-wav`
- `GET /api/scenes`
- `WS /ws/session/{id}`
