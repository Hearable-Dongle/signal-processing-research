from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .manager import manager
from .models import (
    AdjustSpeakerGainMessage,
    ClearFocusMessage,
    ErrorMessage,
    RawChannelDescriptor,
    RawChannelsResponse,
    SCHEMA_VERSION,
    SetMonitorSourceMessage,
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
    if req.input_source == "simulation":
        scene_path = Path(req.scene_config_path)
        if not scene_path.exists():
            raise HTTPException(status_code=400, detail=f"scene_config_path does not exist: {scene_path}")
    session = manager.start_session(req)
    return SessionStartResponse(session_id=session.session_id, status=session.status, config_echo=req)


@app.get("/api/session/{session_id}/status", response_model=SessionStatusResponse)
def status_session(session_id: str) -> SessionStatusResponse:
    return manager.get_session(session_id).get_status()


@app.get("/api/session/active")
def active_session() -> dict[str, str | None]:
    sess = manager.get_active_session()
    if sess is None:
        return {"session_id": None, "status": None}
    return {"session_id": sess.session_id, "status": sess.status}


@app.post("/api/session/active/stop", response_model=SessionStopResponse)
def stop_active_session() -> SessionStopResponse:
    sess = manager.get_active_session()
    if sess is None:
        raise HTTPException(status_code=404, detail="No active session")
    sess.stop()
    return SessionStopResponse(session_id=sess.session_id, status="stopped")


@app.post("/api/session/{session_id}/stop", response_model=SessionStopResponse)
def stop_session(session_id: str) -> SessionStopResponse:
    session = manager.stop_session(session_id)
    return SessionStopResponse(session_id=session.session_id, status="stopped")


@app.get("/api/scenes")
def get_scenes() -> dict[str, list[str]]:
    return {"scenes": list_sample_scenes()}


@app.get("/api/session/{session_id}/raw-mix-wav")
def get_raw_mix_wav(session_id: str) -> Response:
    session = manager.get_session(session_id)
    wav_bytes = session.get_raw_mix_wav_bytes()
    if not wav_bytes:
        raise HTTPException(status_code=404, detail="raw mixed audio not available yet")
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/api/session/{session_id}/raw-channels", response_model=RawChannelsResponse)
def get_raw_channels(session_id: str) -> RawChannelsResponse:
    session = manager.get_session(session_id)
    channel_count = session.get_raw_channel_count()
    if channel_count <= 0:
        raise HTTPException(status_code=404, detail="raw channel audio not available yet")
    return RawChannelsResponse(
        session_id=session_id,
        sample_rate_hz=session.get_raw_sample_rate_hz(),
        channel_count=channel_count,
        channels=[
            RawChannelDescriptor(channel_index=index, filename=f"channel_{index:03d}.wav")
            for index in range(channel_count)
        ],
    )


@app.get("/api/session/{session_id}/raw-channel/{channel_index}.wav")
def get_raw_channel_wav(session_id: str, channel_index: int) -> Response:
    session = manager.get_session(session_id)
    wav_bytes = session.get_raw_channel_wav_bytes(channel_index)
    if not wav_bytes:
        raise HTTPException(status_code=404, detail=f"raw channel {channel_index} not available")
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/api/session/{session_id}/raw-channels-plot.png")
def get_raw_channels_plot(session_id: str, subtitle: str = Query(default="")) -> Response:
    session = manager.get_session(session_id)
    png_bytes = session.get_raw_channel_plot_png_bytes(subtitle=subtitle)
    if not png_bytes:
        raise HTTPException(status_code=404, detail="raw channel plot not available")
    return Response(content=png_bytes, media_type="image/png")


def _parse_client_message(
    raw: dict,
) -> SelectSpeakerMessage | AdjustSpeakerGainMessage | ClearFocusMessage | SetMonitorSourceMessage | StopSessionMessage:
    msg_type = raw.get("type")
    if msg_type == "select_speaker":
        return SelectSpeakerMessage.model_validate(raw)
    if msg_type == "adjust_speaker_gain":
        return AdjustSpeakerGainMessage.model_validate(raw)
    if msg_type == "clear_focus":
        return ClearFocusMessage.model_validate(raw)
    if msg_type == "set_monitor_source":
        return SetMonitorSourceMessage.model_validate(raw)
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
                elif isinstance(msg, SetMonitorSourceMessage):
                    session.set_monitor_source(msg.monitor_source)
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
from fastapi import Query
