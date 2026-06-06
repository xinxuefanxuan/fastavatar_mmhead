#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_stats(x: np.ndarray) -> tuple[float, float]:
    norms = np.linalg.norm(x, axis=1)
    return float(norms.mean()), float(norms.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--max_examples", type=int, default=5)
    args = ap.parse_args()

    root = args.dataset_root
    manifest = read_jsonl(root / "manifest.jsonl")
    train_rows = read_jsonl(root / "train.jsonl")
    val_rows = read_jsonl(root / "val.jsonl")

    print(f"dataset_root: {root}")
    print(f"num_samples: {len(manifest)}")
    print(f"train_count: {len(train_rows)}")
    print(f"val_count: {len(val_rows)}")

    if not manifest:
        return

    first = np.load(Path(manifest[0]["npz_path"]), allow_pickle=True)
    motion_shape = tuple(first["motion"].shape)
    print(f"motion_shape(example): {motion_shape}")

    expr_means, expr_maxs = [], []
    head_means, head_maxs = [], []
    jaw_means, jaw_maxs = [], []

    for rec in manifest:
        data = np.load(Path(rec["npz_path"]), allow_pickle=True)
        expr = np.asarray(data["expr_delta"], dtype=np.float32)
        head = np.asarray(data["head_delta"], dtype=np.float32)
        jaw = np.asarray(data["jaw_delta"], dtype=np.float32)
        m, x = norm_stats(expr)
        expr_means.append(m)
        expr_maxs.append(x)
        m, x = norm_stats(head)
        head_means.append(m)
        head_maxs.append(x)
        m, x = norm_stats(jaw)
        jaw_means.append(m)
        jaw_maxs.append(x)

    print(f"expr_norm mean={np.mean(expr_means):.6f}, max={np.max(expr_maxs):.6f}")
    print(f"head_norm mean={np.mean(head_means):.6f}, max={np.max(head_maxs):.6f}")
    print(f"jaw_norm  mean={np.mean(jaw_means):.6f}, max={np.max(jaw_maxs):.6f}")

    print("examples:")
    for rec in manifest[: args.max_examples]:
        print(f"  sample_id={rec.get('sample_id','')} npz_path={rec.get('npz_path','')}")


if __name__ == "__main__":
    main()
