#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from motion_model.dataset import MotionDataset
from motion_model.models import TemporalConvVAE


def weighted_recon_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    w = torch.ones((1, 1, gt.shape[-1]), device=gt.device, dtype=gt.dtype)
    w[..., 50:53] = 10.0
    w[..., 53:56] = 10.0
    return ((pred - gt) ** 2 * w).mean()


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # mean over all dims and timesteps
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def beta_for_epoch(epoch: int, total_warmup_epochs: int, base_beta: float, warmup: bool) -> float:
    if not warmup:
        return base_beta
    if total_warmup_epochs <= 0:
        return base_beta
    scale = min(1.0, float(epoch) / float(total_warmup_epochs))
    return float(base_beta * scale)


def evaluate(model, loader, device, beta: float):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["motion"].to(device)
            y, mu, logvar, _ = model(x)
            rec = weighted_recon_loss(y, x)
            kl = kl_loss(mu, logvar)
            loss = rec + beta * kl
            total += float(loss.item())
            n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1e-4)
    ap.add_argument("--kl_warmup", action="store_true", default=True)
    ap.add_argument("--kl_warmup_epochs", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ds = MotionDataset(args.dataset_root, "train")
    val_ds = MotionDataset(args.dataset_root, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TemporalConvVAE(in_dim=56, latent_dim=args.latent_dim).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 1e18
    logs = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tr = 0.0
        n = 0
        beta_ep = beta_for_epoch(ep, args.kl_warmup_epochs, args.beta, args.kl_warmup)
        for batch in train_loader:
            x = batch["motion"].to(args.device)
            y, mu, logvar, _ = model(x)
            rec = weighted_recon_loss(y, x)
            kl = kl_loss(mu, logvar)
            loss = rec + beta_ep * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += float(loss.item())
            n += 1
        train_loss = tr / max(n, 1)
        val_loss = evaluate(model, val_loader, args.device, beta_ep)
        logs.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "beta": beta_ep})
        print(f"epoch={ep} beta={beta_ep:.8f} train={train_loss:.6f} val={val_loss:.6f}")

        ckpt = {
            "model": model.state_dict(),
            "epoch": ep,
            "args": vars(args),
            "val_loss": val_loss,
            "beta": beta_ep,
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, args.output_dir / "best.pt")

    (args.output_dir / "train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
