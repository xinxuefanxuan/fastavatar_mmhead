#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SAMPLE_ID_CANDIDATES = ["sample_id", "id", "name", "motion_id", "file_id"]
EXPR_KEYS = ["expression", "expressions", "expr", "exp", "flame_expression"]
JAW_KEYS = ["jaw_pose", "jaw", "jaw_params"]
HEAD_KEYS = ["global_orient", "head_pose", "head", "pose", "root_pose"]


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(f"numpy is required for codebook building: {e}")
    return np



def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                warn(f"skip invalid manifest line {i}: {e}")
    return rows


def infer_sample_id(row: dict) -> Optional[str]:
    for k in SAMPLE_ID_CANDIDATES:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return Path(v.strip()).stem
        if isinstance(v, (int, float)):
            return str(v)
    return None


def read_annotation(root: Path, category: str, sample_id: str) -> str:
    p = root / "text_annotations" / category / f"{sample_id}.txt"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore").strip()


def _collect_arrays(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    np = _require_numpy()
    found: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            found.extend(_collect_arrays(v, name))
        return found
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            name = f"{prefix}[{i}]" if prefix else f"[{i}]"
            found.extend(_collect_arrays(v, name))
        return found
    if isinstance(obj, np.ndarray):
        found.append((prefix, obj))
        return found
    return found


def _canonical_leaf_key(path_key: str) -> str:
    if not path_key:
        return ""
    leaf = path_key.split(".")[-1]
    leaf = leaf.split("[")[0]
    return leaf.lower()


def _reshape_to_td(arr: Any) -> Any:
    np = _require_numpy()
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    t = a.shape[0]
    return a.reshape(t, -1)


def _find_component(arrs: List[Tuple[str, Any]], candidates: List[str]) -> Optional[np.ndarray]:
    candidates_l = [c.lower() for c in candidates]

    # exact leaf match first
    for c in candidates_l:
        for name, arr in arrs:
            if _canonical_leaf_key(name) == c:
                return _reshape_to_td(arr)
    # substring fallback
    for c in candidates_l:
        for name, arr in arrs:
            if c in name.lower():
                return _reshape_to_td(arr)
    return None


def _delta_stats(arr: Any) -> Dict[str, float]:
    np = _require_numpy()
    if arr.shape[0] <= 0:
        return {"mean": 0.0, "max": 0.0, "std": 0.0}
    delta = arr - arr[0:1]
    norms = np.linalg.norm(delta, axis=1)
    return {
        "mean": float(np.mean(norms)),
        "max": float(np.max(norms)),
        "std": float(np.std(norms)),
    }


def load_motion(path: Path) -> Any:
    np = _require_numpy()
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    raise ValueError(f"unsupported motion format: {path.suffix}")


def compute_motion_stats(motion_obj: Any, sample_id: str) -> Tuple[Dict[str, float], bool, int]:
    np = _require_numpy()
    arrays = _collect_arrays(motion_obj)
    expr = _find_component(arrays, EXPR_KEYS)
    jaw = _find_component(arrays, JAW_KEYS)
    head = _find_component(arrays, HEAD_KEYS)

    stats_valid = True

    if expr is None:
        warn(f"sample={sample_id} missing expression-like array; set expr stats to 0")
        expr_stats = {"mean": 0.0, "max": 0.0, "std": 0.0}
        stats_valid = False
    else:
        expr_stats = _delta_stats(expr)

    if jaw is None:
        warn(f"sample={sample_id} missing jaw-like array; set jaw stats to 0")
        jaw_stats = {"mean": 0.0, "max": 0.0, "std": 0.0}
        stats_valid = False
    else:
        jaw_stats = _delta_stats(jaw)

    if head is None:
        warn(f"sample={sample_id} missing head-like array; set head stats to 0")
        head_stats = {"mean": 0.0, "max": 0.0, "std": 0.0}
        stats_valid = False
    else:
        head_stats = _delta_stats(head)

    frame_candidates = [x.shape[0] for x in (expr, jaw, head) if hasattr(x, "shape") and x.ndim >= 1]
    num_frames = int(max(frame_candidates)) if frame_candidates else 0

    intensity = 0.5 * expr_stats["max"] + 0.3 * head_stats["max"] + 0.2 * jaw_stats["max"]

    stats = {
        "num_frames": num_frames,
        "expr_delta_mean_norm": expr_stats["mean"],
        "expr_delta_max_norm": expr_stats["max"],
        "expr_delta_std_norm": expr_stats["std"],
        "jaw_delta_mean_norm": jaw_stats["mean"],
        "jaw_delta_max_norm": jaw_stats["max"],
        "jaw_delta_std_norm": jaw_stats["std"],
        "head_delta_mean_norm": head_stats["mean"],
        "head_delta_max_norm": head_stats["max"],
        "head_delta_std_norm": head_stats["std"],
        "motion_intensity_score": float(intensity),
    }
    return stats, stats_valid, num_frames


def build_entry(mmhead_root: Path, sample_id: str, verbose: bool = False) -> Optional[dict]:
    pkl_path = mmhead_root / "facial_motion" / f"{sample_id}.pkl"
    npz_path = mmhead_root / "facial_motion" / f"{sample_id}.npz"

    if pkl_path.exists():
        motion_path = pkl_path
        motion_format = "pkl"
    elif npz_path.exists():
        motion_path = npz_path
        motion_format = "npz"
    else:
        warn(f"sample={sample_id} motion missing (.pkl/.npz), skipped")
        return None

    try:
        motion_obj = load_motion(motion_path)
    except Exception as e:
        warn(f"sample={sample_id} failed loading motion {motion_path}: {e}")
        return None

    action = read_annotation(mmhead_root, "action", sample_id)
    detail_expression = read_annotation(mmhead_root, "detail_expression", sample_id)
    detail_head_pose = read_annotation(mmhead_root, "detail_head_pose", sample_id)
    emotion = read_annotation(mmhead_root, "emotion", sample_id)
    emotion_scenario = read_annotation(mmhead_root, "emotion_scenario", sample_id)

    searchable_text = " ".join(
        x for x in [sample_id, action, detail_expression, detail_head_pose, emotion, emotion_scenario] if x
    ).strip()

    stats, stats_valid, _ = compute_motion_stats(motion_obj, sample_id)

    entry = {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "motion_path": str(motion_path),
        "motion_format": motion_format,
        "searchable_text": searchable_text,
        "annotations": {
            "action": action,
            "detail_expression": detail_expression,
            "detail_head_pose": detail_head_pose,
            "emotion": emotion,
            "emotion_scenario": emotion_scenario,
        },
        "motion_stats": stats,
        "stats_valid": bool(stats_valid),
    }

    if verbose:
        print(
            f"[OK] sample={sample_id} fmt={motion_format} frames={stats['num_frames']} "
            f"intensity={stats['motion_intensity_score']:.4f}"
        )
    return entry


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MMHead text-to-motion codebook JSONL.")
    ap.add_argument("--mmhead_root", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output_jsonl", required=True, type=Path)
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.mmhead_root.exists():
        raise SystemExit(f"mmhead_root not found: {args.mmhead_root}")
    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    manifest_rows = read_jsonl(args.manifest)
    if not manifest_rows:
        raise SystemExit("manifest is empty or invalid")

    out_rows: List[dict] = []
    seen = set()
    skip_no_id = 0
    skip_missing_motion = 0

    for row in manifest_rows:
        sample_id = infer_sample_id(row)
        if not sample_id:
            skip_no_id += 1
            warn("manifest row missing sample id candidates, skipped")
            continue
        if sample_id in seen:
            continue
        seen.add(sample_id)

        entry = build_entry(args.mmhead_root, sample_id, verbose=args.verbose)
        if entry is None:
            skip_missing_motion += 1
            continue
        out_rows.append(entry)

        if args.max_samples > 0 and len(out_rows) >= args.max_samples:
            break

    write_jsonl(args.output_jsonl, out_rows)
    print("[Done] codebook built")
    print(f"  manifest rows: {len(manifest_rows)}")
    print(f"  unique sample ids seen: {len(seen)}")
    print(f"  written entries: {len(out_rows)}")
    print(f"  skipped no sample_id: {skip_no_id}")
    print(f"  skipped missing/bad motion: {skip_missing_motion}")
    print(f"  output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
