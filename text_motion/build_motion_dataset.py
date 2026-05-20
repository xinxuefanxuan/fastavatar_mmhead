#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip invalid jsonl line {i}: {exc}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_axis_signs(s: str) -> np.ndarray:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 3:
        raise SystemExit(f"--head_axis_signs expects 3 comma-separated ints, got: {s}")
    for v in vals:
        if v not in (-1, 1):
            raise SystemExit(f"--head_axis_signs only supports -1 or 1, got: {vals}")
    return np.asarray(vals, dtype=np.float32)


def parse_channels(s: str) -> list[str]:
    channels = [x.strip() for x in s.split(",") if x.strip()]
    valid = {"expr", "head", "jaw"}
    if not channels:
        raise SystemExit("--include_channels must not be empty")
    for ch in channels:
        if ch not in valid:
            raise SystemExit(f"invalid channel: {ch}, valid={sorted(valid)}")
    return channels


def load_native_mmhead_pkl(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError("pkl root is not dict")
    if "expcodes" not in obj or "posecodes" not in obj:
        raise ValueError("missing expcodes/posecodes in native MMHead pkl")

    expr = np.asarray(obj["expcodes"], dtype=np.float32)
    pose = np.asarray(obj["posecodes"], dtype=np.float32)

    if expr.ndim != 2:
        expr = expr.reshape(expr.shape[0], -1)
    if pose.ndim != 2:
        pose = pose.reshape(pose.shape[0], -1)
    if pose.shape[1] < 3:
        raise ValueError(f"posecodes dim < 3: {pose.shape}")

    head = pose[:, 0:3].astype(np.float32)
    if pose.shape[1] >= 6:
        jaw = pose[:, 3:6].astype(np.float32)
    else:
        jaw = np.zeros_like(head, dtype=np.float32)
        print(f"[WARN] posecodes has dim={pose.shape[1]} (<6), using zero jaw: {path}")
    return expr.astype(np.float32), head, jaw


def sanitize_td(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def delta_from_ref(x: np.ndarray, ref_n: int) -> np.ndarray:
    x = sanitize_td(x)
    n = x.shape[0]
    r = max(1, min(ref_n, n))
    ref = x[:r].mean(axis=0, keepdims=True)
    return x - ref


def max_velocity_norm(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    v = x[1:] - x[:-1]
    return float(np.linalg.norm(v, axis=1).max())


def resample_or_pad(x: np.ndarray, target_len: int, mode: str) -> np.ndarray:
    x = sanitize_td(x)
    t, d = x.shape
    if t == target_len:
        return x
    if t > target_len:
        if mode == "uniform":
            idx = np.linspace(0, t - 1, target_len).round().astype(np.int64)
            return x[idx]
        return x[:target_len]
    pad = np.repeat(x[-1:, :], target_len - t, axis=0)
    return np.concatenate([x, pad], axis=0)


def build_record(entry: dict, out_npz: Path, target_len: int, dims: dict[str, int]) -> dict:
    return {
        "sample_id": str(entry.get("sample_id", "")),
        "npz_path": str(out_npz),
        "motion_path": str(entry.get("motion_path", "")),
        "searchable_text": str(entry.get("searchable_text", "")),
        "annotations": entry.get("annotations", {}),
        "motion_stats": entry.get("motion_stats", {}),
        "target_len": int(target_len),
        "dims": {k: int(v) for k, v in dims.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook_jsonl", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, required=True)
    ap.add_argument("--target_len", type=int, default=90)
    ap.add_argument("--min_frames", type=int, default=32)
    ap.add_argument("--ref_n", type=int, default=5)
    ap.add_argument("--max_head_velocity", type=float, default=None)
    ap.add_argument("--max_expr_velocity", type=float, default=None)
    ap.add_argument("--include_channels", type=str, default="expr,head,jaw")
    ap.add_argument("--split_ratio", type=float, default=0.9)
    ap.add_argument("--head_axis_signs", type=str, default="1,-1,1")
    ap.add_argument("--sample_mode", choices=["uniform", "first"], default="uniform")
    args = ap.parse_args()

    if not (0.0 < args.split_ratio < 1.0):
        raise SystemExit("--split_ratio must be in (0,1)")

    channels = parse_channels(args.include_channels)
    head_signs = parse_axis_signs(args.head_axis_signs)

    output_root = args.output_root
    motions_dir = output_root / "motions"
    motions_dir.mkdir(parents=True, exist_ok=True)

    entries = read_jsonl(args.codebook_jsonl)
    records: list[dict] = []

    n_total = 0
    n_saved = 0
    n_skipped = 0
    for entry in entries:
        n_total += 1
        sample_id = str(entry.get("sample_id", "")).strip()
        motion_path = Path(str(entry.get("motion_path", "")))
        if not sample_id or not motion_path.exists():
            print(f"[WARN] skip missing sample_id/motion: sample_id={sample_id} motion_path={motion_path}")
            n_skipped += 1
            continue
        if motion_path.suffix.lower() != ".pkl":
            print(f"[WARN] skip non-pkl sample: {sample_id} path={motion_path}")
            n_skipped += 1
            continue

        try:
            expr, head, jaw = load_native_mmhead_pkl(motion_path)
        except Exception as exc:
            print(f"[WARN] skip malformed pkl sample={sample_id}: {exc}")
            n_skipped += 1
            continue

        t = min(expr.shape[0], head.shape[0], jaw.shape[0])
        if t < args.min_frames:
            print(f"[WARN] skip short sample={sample_id} frames={t} < min_frames={args.min_frames}")
            n_skipped += 1
            continue

        expr = expr[:t]
        head = head[:t]
        jaw = jaw[:t]

        if not np.isfinite(expr).all() or not np.isfinite(head).all() or not np.isfinite(jaw).all():
            print(f"[WARN] skip non-finite sample={sample_id}")
            n_skipped += 1
            continue

        if args.max_expr_velocity is not None and max_velocity_norm(expr) > args.max_expr_velocity:
            print(f"[WARN] skip high expr velocity sample={sample_id}")
            n_skipped += 1
            continue
        if args.max_head_velocity is not None and max_velocity_norm(head) > args.max_head_velocity:
            print(f"[WARN] skip high head velocity sample={sample_id}")
            n_skipped += 1
            continue

        expr_delta = delta_from_ref(expr, args.ref_n)
        head_delta = delta_from_ref(head, args.ref_n)
        jaw_delta = delta_from_ref(jaw, args.ref_n)

        head_delta = head_delta * head_signs[None, :]

        expr_delta = resample_or_pad(expr_delta, args.target_len, args.sample_mode)
        head_delta = resample_or_pad(head_delta, args.target_len, args.sample_mode)
        jaw_delta = resample_or_pad(jaw_delta, args.target_len, args.sample_mode)

        motion_parts: list[np.ndarray] = []
        dims = {"expr": 50, "head": 3, "jaw": 3}
        if "expr" in channels:
            motion_parts.append(expr_delta)
        if "head" in channels:
            motion_parts.append(head_delta)
        if "jaw" in channels:
            motion_parts.append(jaw_delta)
        motion = np.concatenate(motion_parts, axis=1).astype(np.float32)

        out_npz = motions_dir / f"{sample_id}.npz"
        np.savez(
            out_npz,
            motion=motion,
            expr_delta=expr_delta.astype(np.float32),
            head_delta=head_delta.astype(np.float32),
            jaw_delta=jaw_delta.astype(np.float32),
            original_length=np.int32(t),
            sample_id=np.asarray(sample_id),
        )

        records.append(build_record(entry, out_npz, args.target_len, dims))
        n_saved += 1

    records = sorted(records, key=lambda x: x["sample_id"])
    n_train = int(len(records) * args.split_ratio)
    train_rows = records[:n_train]
    val_rows = records[n_train:]

    write_jsonl(output_root / "manifest.jsonl", records)
    write_jsonl(output_root / "train.jsonl", train_rows)
    write_jsonl(output_root / "val.jsonl", val_rows)

    meta = {
        "num_total": n_total,
        "num_saved": n_saved,
        "num_skipped": n_skipped,
        "target_len": int(args.target_len),
        "min_frames": int(args.min_frames),
        "ref_n": int(args.ref_n),
        "channels": channels,
        "head_axis_signs": head_signs.tolist(),
        "split_ratio": float(args.split_ratio),
        "sample_mode": args.sample_mode,
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
