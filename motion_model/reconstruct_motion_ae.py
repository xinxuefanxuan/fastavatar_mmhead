#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_model.models import TemporalConvAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--motion_key", type=str, default="motion_norm")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    latent_dim = int(ckpt.get("args", {}).get("latent_dim", 64))
    model = TemporalConvAE(in_dim=56, latent_dim=latent_dim).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = np.load(args.input_npz, allow_pickle=True)
    if args.motion_key not in data:
        raise SystemExit(f"missing key {args.motion_key} in {args.input_npz}")
    motion_norm = np.asarray(data[args.motion_key], dtype=np.float32)

    x = torch.from_numpy(motion_norm[None, ...]).to(args.device)
    with torch.no_grad():
        pred_norm, _ = model(x)
    pred_norm = pred_norm[0].cpu().numpy().astype(np.float32)

    stats = json.loads(args.norm_stats.read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    motion_raw = pred_norm * std[None, :] + mean[None, :]

    out = {
        "motion": motion_raw.astype(np.float32),
        "motion_raw": motion_raw.astype(np.float32),
        "motion_norm": pred_norm.astype(np.float32),
        "expr_delta": motion_raw[:, 0:50].astype(np.float32),
        "head_delta": motion_raw[:, 50:53].astype(np.float32),
        "jaw_delta": motion_raw[:, 53:56].astype(np.float32),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)
    print(f"saved reconstruction npz: {args.output_npz}")


if __name__ == "__main__":
    main()
