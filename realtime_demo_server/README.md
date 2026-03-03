# Realtime Demo Server

FastAPI backend bridge for the realtime speaker UI demo.

## Install

From repo root:

```bash
python -m pip install -U fastapi uvicorn websockets
```

## Start backend

```bash
PYTHONPATH=. uvicorn realtime_demo_server.app:app --reload --port 8000
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

## API summary

- `POST /api/session/start`
- `GET /api/session/{id}/status`
- `POST /api/session/{id}/stop`
- `GET /api/scenes`
- `WS /ws/session/{id}`
