from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .manager import manager
from .models import (
    AdjustSpeakerGainMessage,
    ClearFocusMessage,
    ErrorMessage,
    SCHEMA_VERSION,
    SelectSpeakerMessage,
    SessionStartRequest,
    SessionStartResponse,
    SessionStatusResponse,
    SessionStopResponse,
    StopSessionMessage,
)
from .scenes import list_sample_scenes

app = FastAPI(title="Realtime Speaker Demo Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"ok": "true"}


@app.post("/api/session/start", response_model=SessionStartResponse)
def start_session(req: SessionStartRequest) -> SessionStartResponse:
    scene_path = Path(req.scene_config_path)
    if not scene_path.exists():
        raise HTTPException(status_code=400, detail=f"scene_config_path does not exist: {scene_path}")
    session = manager.start_session(req)
    return SessionStartResponse(session_id=session.session_id, status=session.status, config_echo=req)


@app.get("/api/session/{session_id}/status", response_model=SessionStatusResponse)
def status_session(session_id: str) -> SessionStatusResponse:
    return manager.get_session(session_id).get_status()


@app.post("/api/session/{session_id}/stop", response_model=SessionStopResponse)
def stop_session(session_id: str) -> SessionStopResponse:
    session = manager.stop_session(session_id)
    return SessionStopResponse(session_id=session.session_id, status="stopped")


@app.get("/api/scenes")
def get_scenes() -> dict[str, list[str]]:
    return {"scenes": list_sample_scenes()}


def _parse_client_message(raw: dict) -> SelectSpeakerMessage | AdjustSpeakerGainMessage | ClearFocusMessage | StopSessionMessage:
    msg_type = raw.get("type")
    if msg_type == "select_speaker":
        return SelectSpeakerMessage.model_validate(raw)
    if msg_type == "adjust_speaker_gain":
        return AdjustSpeakerGainMessage.model_validate(raw)
    if msg_type == "clear_focus":
        return ClearFocusMessage.model_validate(raw)
    if msg_type == "stop_session":
        return StopSessionMessage.model_validate(raw)
    raise ValueError(f"Unsupported message type: {msg_type}")


@app.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    try:
        session = manager.get_session(session_id)
    except HTTPException as exc:
        await websocket.send_json(ErrorMessage(schema_version=SCHEMA_VERSION, error=str(exc.detail), timestamp_ms=0.0).model_dump())
        await websocket.close(code=4404)
        return

    audio_seq = 0
    speaker_version = 0
    metrics_version = 0
    event_seq = 0

    while True:
        try:
            raw_text = None
            try:
                raw_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.03)
            except TimeoutError:
                raw_text = None

            if raw_text:
                payload = json.loads(raw_text)
                msg = _parse_client_message(payload)
                if isinstance(msg, SelectSpeakerMessage):
                    session.select_speaker(msg.speaker_id)
                elif isinstance(msg, AdjustSpeakerGainMessage):
                    session.adjust_speaker_gain(msg.speaker_id, msg.delta_db_step)
                elif isinstance(msg, ClearFocusMessage):
                    session.clear_focus()
                elif isinstance(msg, StopSessionMessage):
                    session.stop()

            event_seq, events = session.iter_events_since(event_seq)
            for item in events:
                await websocket.send_json(item)

            speaker_version, speaker_state = session.get_speaker_state_if_new(speaker_version)
            if speaker_state is not None:
                await websocket.send_json(speaker_state)

            metrics_version, metrics_state = session.get_metrics_if_new(metrics_version)
            if metrics_state is not None:
                await websocket.send_json(metrics_state)

            audio_seq, chunks = session.iter_audio_since(audio_seq)
            for chunk in chunks:
                await websocket.send_bytes(chunk)

            await asyncio.sleep(0.01)
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_json(
                ErrorMessage(schema_version=SCHEMA_VERSION, error=str(exc), timestamp_ms=0.0).model_dump()
            )
            await asyncio.sleep(0.05)
