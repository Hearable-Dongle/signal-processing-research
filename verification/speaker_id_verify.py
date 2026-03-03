from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .contracts import MetricRecord, SanityArtifactRecord, SubsystemVerificationResult


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _macro_f1(labels: list[int], preds: list[int]) -> float:
    classes = sorted(set(labels) | set(preds))
    f1s = []
    for c in classes:
        tp = sum(1 for y, p in zip(labels, preds) if y == c and p == c)
        fp = sum(1 for y, p in zip(labels, preds) if y != c and p == c)
        fn = sum(1 for y, p in zip(labels, preds) if y == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * prec * rec / (prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def _id_switch_rate(rows: list[dict[str, str]]) -> float:
    by_oracle: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        sid_s = r.get("pred_speaker_id", "").strip()
        if sid_s == "":
            continue
        oracle = int(r["oracle_label"])
        chunk = int(r["chunk_id"])
        sid = int(sid_s)
        by_oracle[oracle].append((chunk, sid))

    switches = 0
    trans = 0
    for seq in by_oracle.values():
        seq.sort(key=lambda x: x[0])
        for i in range(1, len(seq)):
            trans += 1
            if seq[i][1] != seq[i - 1][1]:
                switches += 1
    return float(switches / trans) if trans else 0.0


def _plot_confusion(labels: list[int], preds: list[int], out_path: Path) -> None:
    classes = sorted(set(labels) | set(preds))
    idx = {c: i for i, c in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=int)
    for y, p in zip(labels, preds):
        mat[idx[y], idx[p]] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted speaker_id")
    ax.set_ylabel("Oracle label")
    ax.set_title("Speaker ID Confusion")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def verify_speaker_identification(out_root: Path, num_scenes: int = 12) -> SubsystemVerificationResult:
    out_dir = out_root / "speaker_identification"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use seed_embeddings mode on synthetic scenes to force explicit identification mapping objective.
    cmd = [
        sys.executable,
        "-m",
        "direction_assignment.validate",
        "--num-scenes",
        str(num_scenes),
        "--identity-mode",
        "enroll_audio",
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return SubsystemVerificationResult(
            subsystem="speaker_identification",
            status="error",
            details={"stderr": proc.stderr[-4000:], "stdout": proc.stdout[-4000:]},
        )

    # collect all track assignment rows
    scene_dirs = [p for p in out_dir.iterdir() if p.is_dir() and (p / "track_assignments.csv").exists()]
    rows: list[dict[str, str]] = []
    for d in scene_dirs:
        rows.extend(_read_csv(d / "track_assignments.csv"))

    pairs = []
    conf_correct = []
    conf_wrong = []
    for r in rows:
        sid_s = r.get("pred_speaker_id", "").strip()
        if sid_s == "":
            continue
        y = int(r["oracle_label"])
        p = int(sid_s)
        pairs.append((y, p))
        conf = float(r.get("identity_conf", "0") or 0.0)
        if y == p:
            conf_correct.append(conf)
        else:
            conf_wrong.append(conf)

    if not pairs:
        return SubsystemVerificationResult(subsystem="speaker_identification", status="error", details={"error": "no labeled pairs"})

    labels = [y for y, _ in pairs]
    preds = [p for _, p in pairs]
    acc = float(sum(int(y == p) for y, p in pairs) / len(pairs))
    macro_f1 = _macro_f1(labels, preds)
    switch = _id_switch_rate(rows)

    conf_path = out_dir / "confusion_matrix.png"
    _plot_confusion(labels, preds, conf_path)

    metrics = [
        MetricRecord("id_accuracy", acc, True, 0.90, acc >= 0.90),
        MetricRecord("macro_f1", macro_f1, True, 0.85, macro_f1 >= 0.85),
        MetricRecord("id_switch_rate", switch, False, 0.10, switch <= 0.10),
        MetricRecord("mean_conf_correct", float(np.mean(conf_correct)) if conf_correct else 0.0, True),
        MetricRecord("mean_conf_wrong", float(np.mean(conf_wrong)) if conf_wrong else 0.0, False),
    ]

    artifacts = [
        SanityArtifactRecord("plot", str(conf_path)),
        SanityArtifactRecord("csv", str(out_dir / "per_scene_metrics.csv")),
        SanityArtifactRecord("csv", str(out_dir / "per_chunk_metrics.csv")),
    ]

    status = "pass" if all(m.passed for m in metrics if m.passed is not None) else "warn"
    return SubsystemVerificationResult(
        subsystem="speaker_identification",
        status=status,
        metrics=metrics,
        artifacts=artifacts,
        details={"results_dir": str(out_dir.resolve()), "num_pairs": len(pairs)},
    )
