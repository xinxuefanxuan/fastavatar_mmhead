#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from motion_model.channel_vae import ChannelTemporalVAE

CHANNEL_SLICE = {"expr": (0, 50), "head": (50, 53), "jaw": (53, 56)}
DEFAULT_LATENT = {"expr": 64, "head": 16, "jaw": 16}


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fit_length(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] >= target_len:
        return x[:target_len]
    pad = np.repeat(x[-1:], target_len - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0)


class ChannelDataset(Dataset):
    def __init__(self, manifest: Path, norm_stats: Path, channel: str, target_len: int):
        rows = read_jsonl(manifest)
        st = json.loads(norm_stats.read_text())
        self.mean = np.asarray(st["mean"], dtype=np.float32)
        self.std = np.asarray(st["std"], dtype=np.float32)
        s, e = CHANNEL_SLICE[channel]
        self.items = []
        for r in rows:
            p = r.get("npz_path") or r.get("motion_path")
            if not p:
                continue
            p = Path(p)
            if not p.exists():
                continue
            arrs = np.load(p, allow_pickle=False)
            if "motion_norm" in arrs:
                m = np.asarray(arrs["motion_norm"], dtype=np.float32)
            else:
                raw = np.asarray(arrs.get("motion") if "motion" in arrs else arrs["motion_raw"], dtype=np.float32)
                m = (raw - self.mean[None, :]) / np.maximum(self.std[None, :], 1e-8)
            if m.ndim != 2 or m.shape[1] < 56:
                continue
            m = fit_length(m[:, :56], target_len)
            self.items.append(m[:, s:e])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return torch.from_numpy(self.items[idx].astype(np.float32))


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def channel_stats(x):
    norms = torch.linalg.norm(x, dim=2)
    d = {"norm_mean": norms.mean().item(), "norm_max": norms.max().item()}
    if x.shape[2] == 3:
        yaw = x[:, :, 1]
        d["yaw_min"] = yaw.min().item()
        d["yaw_max"] = yaw.max().item()
    return d


def run_epoch(model, loader, opt, beta, device):
    train = opt is not None
    model.train(train)
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    tgt_all, rec_all = [], []
    for x in loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_l = F.mse_loss(recon, x)
        kl = kl_loss(mu, logvar)
        loss = recon_l + beta * kl
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        bs = x.shape[0]
        n += bs
        totals["total"] += loss.item() * bs
        totals["recon"] += recon_l.item() * bs
        totals["kl"] += kl.item() * bs
        tgt_all.append(x.detach().cpu())
        rec_all.append(recon.detach().cpu())
    for k in totals:
        totals[k] = totals[k] / max(1, n)
    tgt = torch.cat(tgt_all, 0)
    rec = torch.cat(rec_all, 0)
    return totals, channel_stats(tgt), channel_stats(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", type=Path, default=Path("outputs/mmhead_debug/motion_dataset_v1_ae_debug/train.jsonl"))
    ap.add_argument("--val_manifest", type=Path, default=Path("outputs/mmhead_debug/motion_dataset_v1_ae_debug/val.jsonl"))
    ap.add_argument("--norm_stats", type=Path, default=Path("outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json"))
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--channel", type=str, choices=["expr", "head", "jaw"], required=True)
    ap.add_argument("--target_len", type=int, default=64)
    ap.add_argument("--latent_dim", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_every", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    latent_dim = args.latent_dim if args.latent_dim is not None else DEFAULT_LATENT[args.channel]
    in_dim = CHANNEL_SLICE[args.channel][1] - CHANNEL_SLICE[args.channel][0]

    train_ds = ChannelDataset(args.train_manifest, args.norm_stats, args.channel, args.target_len)
    val_ds = ChannelDataset(args.val_manifest, args.norm_stats, args.channel, args.target_len)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ChannelTemporalVAE(in_dim, args.target_len, latent_dim, args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update({"latent_dim": latent_dim, "input_dim": in_dim})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best = 1e9
    logf = (args.output_dir / "train_log.jsonl").open("w", encoding="utf-8")
    for ep in range(1, args.epochs + 1):
        tr, tr_tgt, tr_rec = run_epoch(model, train_ld, opt, args.beta, device)
        va, va_tgt, va_rec = run_epoch(model, val_ld, None, args.beta, device)
        row = {
            "epoch": ep,
            "train_total": tr["total"], "train_recon": tr["recon"], "train_kl": tr["kl"],
            "val_total": va["total"], "val_recon": va["recon"], "val_kl": va["kl"],
            "target_stats": va_tgt, "recon_stats": va_rec,
        }
        logf.write(json.dumps(row) + "\n"); logf.flush()
        print(f"epoch={ep} train={tr['total']:.6f} val={va['total']:.6f}")

        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "channel": args.channel,
            "latent_dim": latent_dim,
            "hidden_dim": args.hidden_dim,
            "target_len": args.target_len,
            "input_dim": in_dim,
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        if va["total"] < best:
            best = va["total"]
            torch.save(ckpt, args.output_dir / "best.pt")
        if args.save_every and ep % args.save_every == 0:
            torch.save(ckpt, args.output_dir / f"epoch_{ep:04d}.pt")

    logf.close()


if __name__ == "__main__":
    main()
