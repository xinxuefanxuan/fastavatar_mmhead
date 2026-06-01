#!/usr/bin/env python3
"""Inspect FastAvatar train config dataset IDs before launching P9.2 smoke training."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_config(path: Path) -> Any:
    try:
        from omegaconf import OmegaConf
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("OmegaConf is required to read FastAvatar yaml configs") from exc
    return OmegaConf.load(path)


def resolve_path(value: str | Path, base: Path) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_meta(meta_path: Path) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metadata JSON object/dict, got {type(data).__name__}: {meta_path}")
    return data


def filter_for_dataset(all_frame_groups: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    """Mirror MixerDataset._filter_data_for_subset for inspection without importing torch."""
    filtered: dict[str, Any] = {}
    prefix = f"{dataset_name}/"
    for path, frame_data in all_frame_groups.items():
        if path.startswith(prefix):
            filtered[path[len(prefix):]] = frame_data
            continue
        if "/" in path and path.split("/")[0] not in ["nersemble", "vfhq"]:
            if dataset_name == "vfhq":
                parts = path.split("/")
                if (len(parts) >= 2 and "EXP-" in parts[1]) or (len(parts) == 2 and parts[0].isdigit() and "EXP-" in path):
                    filtered[path] = frame_data
            # Nersemble legacy fallback intentionally matches current MixerDataset: no-op.
    return filtered


def extract_nersemble_id(path: str) -> str:
    return path.split("/")[0] if path else path


def selected_val_ids(all_ids: list[str], configured_val_id: list[str] | None) -> tuple[list[str], list[str]]:
    configured_val_id = configured_val_id or []
    if not configured_val_id:
        if len(all_ids) < 5:
            return list(all_ids), []
        rng = random.Random(42)
        return rng.sample(list(all_ids), 5), []
    valid = [uid for uid in configured_val_id if uid in set(all_ids)]
    invalid = [uid for uid in configured_val_id if uid not in set(all_ids)]
    if not valid:
        # Mirrors NersembleDataset fallback that makes all IDs validation and therefore can make train empty.
        return list(all_ids), invalid
    return valid, invalid


def infer_counts(filtered: dict[str, Any], configured_val_id: list[str] | None) -> dict[str, Any]:
    id_to_count = Counter(extract_nersemble_id(path) for path in filtered)
    all_ids = sorted(id_to_count)
    val_ids, invalid = selected_val_ids(all_ids, configured_val_id)
    val_count = sum(id_to_count[uid] for uid in val_ids)
    train_count = len(filtered) - val_count
    return {
        "available_ids": all_ids,
        "id_to_count": dict(id_to_count),
        "valid_val_id": val_ids,
        "invalid_val_id": invalid,
        "expected_train_count": train_count,
        "expected_val_candidate_count": val_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect FastAvatar dataset IDs and train/val split for a config.")
    parser.add_argument("--config", type=Path, default=Path("configs/train/fastavatar_motion_zero_token_overfit.yaml"))
    parser.add_argument("--dataset", default="nersemble")
    parser.add_argument("--require_train", action="store_true", help="Exit non-zero if expected train count is zero or metadata is missing.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg_path = resolve_path(args.config, repo_root)
    print(f"[InspectDataset] config={cfg_path}")
    cfg = load_config(cfg_path)

    meta_path = resolve_path(cfg.dataset.meta_path, repo_root)
    dataset_cfg = cfg.dataset.datasets[args.dataset]
    root_dir = resolve_path(dataset_cfg.root_dir, repo_root)
    configured_val_id = list(getattr(dataset_cfg, "val_id", []) or [])

    print(f"[InspectDataset] resolved meta_path={meta_path}")
    print(f"[InspectDataset] meta_path exists={meta_path.exists()}")
    print(f"[InspectDataset] resolved {args.dataset} root_dir={root_dir}")
    print(f"[InspectDataset] root_dir exists={root_dir.exists()}")
    print(f"[InspectDataset] configured val_id={configured_val_id}")

    if not meta_path.exists():
        print(f"[InspectDataset][ERROR] metadata does not exist: {meta_path}")
        return 2 if args.require_train else 0

    all_meta = load_meta(meta_path)
    filtered = filter_for_dataset(all_meta, args.dataset)
    counts = infer_counts(filtered, configured_val_id)

    print(f"[InspectDataset] loaded frame groups/items total={len(all_meta)}")
    print(f"[InspectDataset] filtered frame groups/items for {args.dataset}={len(filtered)}")
    print(f"[InspectDataset] available IDs ({len(counts['available_ids'])}): {counts['available_ids'][:50]}")
    if len(counts["available_ids"]) > 50:
        print(f"[InspectDataset] ... {len(counts['available_ids']) - 50} more IDs omitted")
    print(f"[InspectDataset] valid_val_id={counts['valid_val_id']}")
    print(f"[InspectDataset] invalid_val_id={counts['invalid_val_id']}")
    print(f"[InspectDataset] expected train count={counts['expected_train_count']}")
    print(f"[InspectDataset] expected val candidate count={counts['expected_val_candidate_count']}")
    print(f"[InspectDataset] top ID counts={Counter(counts['id_to_count']).most_common(10)}")

    if args.require_train and counts["expected_train_count"] <= 0:
        print("[InspectDataset][ERROR] expected train count is 0. Create overfit metadata or choose a valid val_id split first.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
