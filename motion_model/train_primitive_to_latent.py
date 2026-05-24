#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from motion_model.models import TemporalConvVAE


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PrimitiveLatentDataset(Dataset):
    def __init__(self, rows: list[dict], label_map: dict[str, int], vae: TemporalConvVAE, device: str):
        self.items = []
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        with torch.no_grad():
            for r in rows:
                npz = np.load(Path(r["npz_path"]), allow_pickle=True)
                motion = np.asarray(npz["motion_norm"], dtype=np.float32)
                x = torch.from_numpy(motion[None, ...]).to(device)
                mu, _ = vae.encode(x)
                mu_mean = mu.mean(dim=2)[0].cpu()  # [latent_dim]
                label = r.get("primitive_label", "other")
                idx = label_map.get(label, label_map["other"])
                self.items.append((idx, mu_mean))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        l, z = self.items[idx]
        return torch.tensor(l, dtype=torch.long), z.float()


class PrimitiveMLP(nn.Module):
    def __init__(self, num_labels: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_labels, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, latent_dim))

    def forward(self, label_idx: torch.Tensor) -> torch.Tensor:
        h = self.embedding(label_idx)
        return self.mlp(h)


def eval_loss(model, loader, device):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for y, z in loader:
            y = y.to(device)
            z = z.to(device)
            p = model(y)
            loss = ((p - z) ** 2).mean()
            tot += float(loss.item())
            n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_root", type=Path, required=True)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    label_map = json.loads((args.labeled_root / "label_map.json").read_text(encoding="utf-8"))
    tr = read_jsonl(args.labeled_root / "train_labeled.jsonl")
    va = read_jsonl(args.labeled_root / "val_labeled.jsonl")

    ckpt = torch.load(args.vae_checkpoint, map_location=args.device)
    latent_dim = int(ckpt.get("args", {}).get("latent_dim", 64))
    vae = TemporalConvVAE(in_dim=56, latent_dim=latent_dim).to(args.device)
    vae.load_state_dict(ckpt["model"])

    ds_tr = PrimitiveLatentDataset(tr, label_map, vae, args.device)
    ds_va = PrimitiveLatentDataset(va, label_map, vae, args.device)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)

    model = PrimitiveMLP(num_labels=len(label_map), latent_dim=latent_dim).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = 1e18
    logs = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        n = 0
        for y, z in dl_tr:
            y, z = y.to(args.device), z.to(args.device)
            p = model(y)
            loss = ((p - z) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            n += 1
        tr_loss /= max(n, 1)
        va_loss = eval_loss(model, dl_va, args.device)
        logs.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"epoch={ep} train={tr_loss:.6f} val={va_loss:.6f}")

        ckpt_out = {"model": model.state_dict(), "args": vars(args), "label_map": label_map, "val_loss": va_loss}
        torch.save(ckpt_out, args.output_dir / "last.pt")
        if va_loss < best:
            best = va_loss
            torch.save(ckpt_out, args.output_dir / "best.pt")

    (args.output_dir / "train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
