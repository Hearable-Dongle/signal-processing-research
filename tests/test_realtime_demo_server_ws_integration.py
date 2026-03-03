import json
import time

from fastapi.testclient import TestClient

from realtime_demo_server.app import app
from realtime_demo_server.manager import manager
from realtime_demo_server.ws_codec import decode_audio_chunk


client = TestClient(app)


def _cleanup() -> None:
    sess = getattr(manager, "_session", None)
    if sess is not None:
        sess.stop()
        sess.join(timeout=2.0)


def test_ws_receives_state_metrics_and_audio_and_applies_controls() -> None:
    _cleanup()
    start = client.post(
        "/api/session/start",
        json={
            "scene_config_path": "simulation/simulations/configs/library_scene/library_k1_scene00.json",
            "separation_mode": "mock",
            "slow_chunk_ms": 200,
        },
    )
    assert start.status_code == 200
    sid = start.json()["session_id"]

    got_speaker = False
    got_metrics = False
    got_audio = False

    with client.websocket_connect(f"/ws/session/{sid}") as ws:
        ws.send_text(json.dumps({"schema_version": "v1", "type": "select_speaker", "speaker_id": 1}))
        ws.send_text(json.dumps({"schema_version": "v1", "type": "adjust_speaker_gain", "speaker_id": 1, "delta_db_step": 1}))
        deadline = time.time() + 5.0
        while time.time() < deadline and not (got_speaker and got_metrics and got_audio):
            packet = ws.receive()
            msg_type = packet.get("type")
            if msg_type in {"websocket.receive", "websocket.send"} and "text" in packet:
                payload = json.loads(packet["text"])
                if payload.get("type") == "speaker_state":
                    got_speaker = True
                if payload.get("type") == "metrics":
                    got_metrics = True
            if msg_type in {"websocket.receive", "websocket.send"} and "bytes" in packet:
                ts_ms, audio = decode_audio_chunk(packet["bytes"])
                assert ts_ms >= 0.0
                assert audio.size > 0
                got_audio = True

    assert got_speaker
    assert got_metrics
    assert got_audio

    status = client.get(f"/api/session/{sid}/status")
    assert status.status_code == 200
    assert int(status.json()["selected_speaker_id"]) == 1
    assert float(status.json()["speaker_gain_delta_db"]["1"]) == 1.0

    _cleanup()
