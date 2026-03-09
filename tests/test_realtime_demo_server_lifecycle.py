import time

from fastapi.testclient import TestClient

from mic_array_forwarder.app import app
from mic_array_forwarder.manager import manager


client = TestClient(app)


def _cleanup() -> None:
    sess = getattr(manager, "_session", None)
    if sess is not None:
        sess.stop()
        sess.join(timeout=2.0)


def test_start_status_stop_lifecycle() -> None:
    _cleanup()
    resp = client.post(
        "/api/session/start",
        json={
            "scene_config_path": "simulation/simulations/configs/library_scene/library_k1_scene00.json",
            "separation_mode": "mock",
            "slow_chunk_ms": 200,
        },
    )
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    time.sleep(0.2)
    status = client.get(f"/api/session/{sid}/status")
    assert status.status_code == 200
    assert status.json()["session_id"] == sid

    stop = client.post(f"/api/session/{sid}/stop")
    assert stop.status_code == 200

    _cleanup()


def test_second_start_returns_409_while_running() -> None:
    _cleanup()
    first = client.post(
        "/api/session/start",
        json={
            "scene_config_path": "simulation/simulations/configs/library_scene/library_k1_scene00.json",
            "separation_mode": "mock",
            "slow_chunk_ms": 200,
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/api/session/start",
        json={
            "scene_config_path": "simulation/simulations/configs/library_scene/library_k1_scene00.json",
            "separation_mode": "mock",
            "slow_chunk_ms": 200,
        },
    )
    assert second.status_code == 409

    _cleanup()


def test_raw_mix_wav_endpoint_available_after_start() -> None:
    _cleanup()
    resp = client.post(
        "/api/session/start",
        json={
            "scene_config_path": "simulation/simulations/configs/library_scene/library_k1_scene00.json",
            "separation_mode": "mock",
            "slow_chunk_ms": 200,
        },
    )
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    wav_resp = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        candidate = client.get(f"/api/session/{sid}/raw-mix-wav")
        if candidate.status_code == 200:
            wav_resp = candidate
            break
        time.sleep(0.05)

    assert wav_resp is not None
    assert wav_resp.headers["content-type"].startswith("audio/wav")
    assert wav_resp.content[:4] == b"RIFF"

    _cleanup()


def test_raw_channel_endpoints_available_after_start() -> None:
    _cleanup()
    resp = client.post(
        "/api/session/start",
        json={
            "scene_config_path": "simulation/simulations/configs/library_scene/library_k1_scene00.json",
            "separation_mode": "mock",
            "slow_chunk_ms": 200,
        },
    )
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    manifest = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        candidate = client.get(f"/api/session/{sid}/raw-channels")
        if candidate.status_code == 200:
            manifest = candidate
            break
        time.sleep(0.05)

    assert manifest is not None
    payload = manifest.json()
    assert payload["session_id"] == sid
    assert payload["channel_count"] >= 1
    assert len(payload["channels"]) == payload["channel_count"]

    first = client.get(f"/api/session/{sid}/raw-channel/0.wav")
    assert first.status_code == 200
    assert first.headers["content-type"].startswith("audio/wav")
    assert first.content[:4] == b"RIFF"

    missing = client.get(f"/api/session/{sid}/raw-channel/999.wav")
    assert missing.status_code == 404

    _cleanup()
