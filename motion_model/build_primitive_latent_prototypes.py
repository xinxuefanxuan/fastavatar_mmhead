#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_model.models import TemporalConvVAE


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_root", type=Path, required=True)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--output_path", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = read_jsonl(args.labeled_root / "train_labeled.jsonl")
    if args.top_k is not None and args.top_k > 0:
        rows = rows[: args.top_k]

    ckpt = torch.load(args.vae_checkpoint, map_location=args.device)
    latent_dim = int(ckpt.get("args", {}).get("latent_dim", 64))
    vae = TemporalConvVAE(in_dim=56, latent_dim=latent_dim).to(args.device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    label_map = json.loads((args.labeled_root / "label_map.json").read_text(encoding="utf-8"))
    buckets: dict[str, list[np.ndarray]] = {k: [] for k in label_map.keys()}

    with torch.no_grad():
        for r in rows:
            label = r.get("primitive_label", "other")
            if label not in buckets:
                continue
            npz = np.load(Path(r["npz_path"]), allow_pickle=True)
            motion = np.asarray(npz["motion_norm"], dtype=np.float32)
            x = torch.from_numpy(motion[None, ...]).to(args.device)
            mu, _ = vae.encode(x)  # [1,latent,T]
            mu_mean = mu.mean(dim=2)[0].cpu().numpy().astype(np.float32)
            buckets[label].append(mu_mean)

    label_to_mean_mu = {}
    label_to_std_mu = {}
    counts = {}
    for label in label_map.keys():
        arrs = buckets.get(label, [])
        if len(arrs) == 0:
            label_to_mean_mu[label] = np.zeros((latent_dim,), dtype=np.float32)
            label_to_std_mu[label] = np.ones((latent_dim,), dtype=np.float32)
            counts[label] = 0
        else:
            stack = np.stack(arrs, axis=0)
            label_to_mean_mu[label] = stack.mean(axis=0).astype(np.float32)
            label_to_std_mu[label] = stack.std(axis=0).astype(np.float32)
            counts[label] = int(stack.shape[0])

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "label_map": label_map,
            "label_to_mean_mu": label_to_mean_mu,
            "label_to_std_mu": label_to_std_mu,
            "counts": counts,
            "latent_dim": latent_dim,
        },
        args.output_path,
    )

    print(f"saved prototypes: {args.output_path}")
    for k in label_map.keys():
        print(f"  {k}: count={counts[k]}")


if __name__ == "__main__":
    main()
