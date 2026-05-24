#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", required=True, type=Path)
    args = ap.parse_args()
    frames = sorted((args.motion_dir / "flame_param").glob("*.npz"))
    if not frames:
        raise SystemExit("no frame npz")
    payloads = [np.load(p, allow_pickle=True) for p in frames]

    for key in ["rotation", "neck_pose", "expr", "jaw_pose"]:
        if key not in payloads[0].files:
            print(f"{key} delta_norm max: N/A")
            continue
        seq = np.stack([np.asarray(p[key], dtype=np.float32).reshape(-1) for p in payloads], axis=0)
        delta = seq - seq[0:1]
        norms = np.linalg.norm(delta, axis=1)
        print(f"{key} delta_norm max: {float(np.max(norms)):.6f}")


if __name__ == "__main__":
    main()
