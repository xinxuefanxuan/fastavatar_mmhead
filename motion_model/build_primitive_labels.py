#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def process_rows(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    out = []
    dist: dict[str, int] = {}
    for r in rows:
        text = " ".join([
            str(r.get("searchable_text", "")),
            str((r.get("annotations", {}) or {}).get("detail_head_pose", "")),
            str((r.get("annotations", {}) or {}).get("detail_expression", "")),
            str((r.get("annotations", {}) or {}).get("action", "")),
        ]).strip()
        label = infer_label(text)
        dist[label] = dist.get(label, 0) + 1
        nr = dict(r)
        nr["primitive_label"] = label
        out.append(nr)
    return out, dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, required=True)
    args = ap.parse_args()

    labels = ["turn_left", "turn_right", "nod", "smile", "mouth_open", "neutral", "other"]
    label_map = {k: i for i, k in enumerate(labels)}

    tr = read_jsonl(args.dataset_root / "train.jsonl")
    va = read_jsonl(args.dataset_root / "val.jsonl")
    tr2, dtr = process_rows(tr)
    va2, dva = process_rows(va)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "train_labeled.jsonl", tr2)
    write_jsonl(args.output_root / "val_labeled.jsonl", va2)
    (args.output_root / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    print("[Label distribution train]")
    for k in labels:
        print(f"  {k}: {dtr.get(k, 0)}")
    print("[Label distribution val]")
    for k in labels:
        print(f"  {k}: {dva.get(k, 0)}")


if __name__ == "__main__":
    main()
