from __future__ import annotations

import json
from pathlib import Path

from beamforming.benchmark.data_collection_benchmark import _discover_recordings


def test_discover_recordings_from_collection_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    recordings = root / "recordings"
    (recordings / "recording-a" / "raw").mkdir(parents=True)
    (recordings / "recording-b" / "raw").mkdir(parents=True)
    (root / "collection.json").write_text(json.dumps({"collectionId": "set-1"}), encoding="utf-8")

    resolved_root, found = _discover_recordings(root)

    assert resolved_root == root
    assert [name for name, _path in found] == ["recording-a", "recording-b"]


def test_discover_recordings_from_specific_recording_dir(tmp_path: Path) -> None:
    recording = tmp_path / "recordings" / "recording-a"
    (recording / "raw").mkdir(parents=True)

    resolved_root, found = _discover_recordings(recording)

    assert resolved_root == recording
    assert found == [("recording-a", recording)]
