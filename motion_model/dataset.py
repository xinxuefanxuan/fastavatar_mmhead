#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class MotionDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str = "train"):
        self.dataset_root = Path(dataset_root)
        self.rows = read_jsonl(self.dataset_root / f"{split}.jsonl")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        data = np.load(Path(row["npz_path"]), allow_pickle=True)
        motion = np.asarray(data["motion_norm"], dtype=np.float32)
        return {
            "motion": torch.from_numpy(motion),
            "sample_id": row.get("sample_id", ""),
            "npz_path": row.get("npz_path", ""),
        }
