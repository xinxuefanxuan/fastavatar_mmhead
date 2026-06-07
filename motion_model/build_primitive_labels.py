#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_label(text: str) -> str:
    t = (text or "").lower()
    if "left" in t and ("turn" in t or "head" in t or "look" in t):
        return "turn_left"
    if "right" in t and ("turn" in t or "head" in t or "look" in t):
        return "turn_right"
    if "nod" in t:
        return "nod"
    if any(x in t for x in ["smile", "smiling", "grin", "happy"]):
        return "smile"
    if any(x in t for x in ["mouth open", "open mouth", "jaw", "speak", "talk"]):
        return "mouth_open"
    if any(x in t for x in ["neutral", "still", "static", "rest"]):
        return "neutral"
    return "other"


def process_rows(
    rows: list[dict],
    yaw_threshold: float,
    pitch_threshold: float,
    jaw_threshold: float,
    neutral_threshold: float,
) -> tuple[list[dict], dict[str, int]]:
    out = []
    dist: dict[str, int] = {}
    for r in rows:
        text = " ".join([
            str(r.get("searchable_text", "")),
            str((r.get("annotations", {}) or {}).get("detail_head_pose", "")),
            str((r.get("annotations", {}) or {}).get("detail_expression", "")),
            str((r.get("annotations", {}) or {}).get("action", "")),
        ]).strip()
        npz_path = Path(r["npz_path"])
        motion = np.asarray(np.load(npz_path, allow_pickle=True)["motion"], dtype=np.float32)
        head = motion[:, 50:53]
        yaw = head[:, 1]
        pitch = head[:, 0]
        jaw = motion[:, 53:56]
        total_norm_max = float(np.linalg.norm(motion, axis=1).max())
        jaw_norm_max = float(np.linalg.norm(jaw, axis=1).max())
        yaw_mean = float(yaw.mean())
        yaw_max = float(yaw.max())
        yaw_min = float(yaw.min())
        pitch_range = float(pitch.max() - pitch.min())

        if total_norm_max < neutral_threshold:
            label = "neutral"
        elif yaw_mean > yaw_threshold or yaw_max > yaw_threshold:
            label = "turn_left"
        elif yaw_mean < -yaw_threshold or yaw_min < -yaw_threshold:
            label = "turn_right"
        elif pitch_range > pitch_threshold:
            label = "nod"
        elif any(x in text.lower() for x in ["mouth", "open", "jaw"]) or jaw_norm_max > jaw_threshold:
            label = "mouth_open"
        elif any(x in text.lower() for x in ["smile", "smiling", "happy", "grin"]):
            label = "smile"
        else:
            label = infer_label(text)
        dist[label] = dist.get(label, 0) + 1
        nr = dict(r)
        nr["primitive_label"] = label
        out.append(nr)
    return out, dist


def cap_per_label(rows: list[dict], max_per_label: int | None) -> list[dict]:
    if max_per_label is None or max_per_label <= 0:
        return rows
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r["primitive_label"], []).append(r)
    out = []
    for _, items in buckets.items():
        out.extend(items[:max_per_label])
    return out


def stratified_split(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r["primitive_label"], []).append(r)
    tr, va = [], []
    rng = random.Random(seed)
    for _, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
        va.extend(items[:n_val])
        tr.extend(items[n_val:])
    return tr, va


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, required=True)
    ap.add_argument("--yaw_threshold", type=float, default=0.015)
    ap.add_argument("--pitch_threshold", type=float, default=0.015)
    ap.add_argument("--jaw_threshold", type=float, default=0.01)
    ap.add_argument("--neutral_threshold", type=float, default=0.02)
    ap.add_argument("--max_per_label", type=int, default=None)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels = ["turn_left", "turn_right", "nod", "smile", "mouth_open", "neutral", "other"]
    label_map = {k: i for i, k in enumerate(labels)}

    all_rows = read_jsonl(args.dataset_root / "manifest.jsonl")
    all_labeled, _ = process_rows(
        all_rows,
        yaw_threshold=args.yaw_threshold,
        pitch_threshold=args.pitch_threshold,
        jaw_threshold=args.jaw_threshold,
        neutral_threshold=args.neutral_threshold,
    )
    all_labeled = cap_per_label(all_labeled, args.max_per_label)
    tr2, va2 = stratified_split(all_labeled, args.val_ratio, args.seed)
    _, dtr = process_rows(
        tr2,
        yaw_threshold=args.yaw_threshold,
        pitch_threshold=args.pitch_threshold,
        jaw_threshold=args.jaw_threshold,
        neutral_threshold=args.neutral_threshold,
    )
    _, dva = process_rows(
        va2,
        yaw_threshold=args.yaw_threshold,
        pitch_threshold=args.pitch_threshold,
        jaw_threshold=args.jaw_threshold,
        neutral_threshold=args.neutral_threshold,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "train_labeled.jsonl", tr2)
    write_jsonl(args.output_root / "val_labeled.jsonl", va2)
    (args.output_root / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    print("[Final label distribution train]")
    for k in labels:
        print(f"  {k}: {dtr.get(k, 0)}")
    print("[Label distribution val]")
    for k in labels:
        print(f"  {k}: {dva.get(k, 0)}")


if __name__ == "__main__":
    main()
