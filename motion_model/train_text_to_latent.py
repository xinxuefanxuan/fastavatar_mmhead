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


def build_id_to_embedding(emb_dir: Path) -> dict[str, np.ndarray]:
    emb = np.load(emb_dir / "embeddings.npy").astype(np.float32)
    ids = json.loads((emb_dir / "sample_ids.json").read_text(encoding="utf-8"))
    if len(ids) != emb.shape[0]:
        raise SystemExit(f"embedding rows != sample_ids: {emb.shape[0]} vs {len(ids)}")
    return {str(sid): emb[i] for i, sid in enumerate(ids)}


def weighted_motion_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    w = torch.ones((1, 1, gt.shape[-1]), dtype=gt.dtype, device=gt.device)
    w[..., 50:53] = 10.0
    w[..., 53:56] = 10.0
    return ((pred - gt) ** 2 * w).mean()


class TextLatentDataset(Dataset):
    def __init__(self, manifest_rows: list[dict], id2emb: dict[str, np.ndarray], vae: TemporalConvVAE, device: str):
        self.items = []
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        with torch.no_grad():
            for r in manifest_rows:
                sid = str(r.get("sample_id", ""))
                if sid not in id2emb:
                    continue
                npz = np.load(Path(r["npz_path"]), allow_pickle=True)
                motion_norm = np.asarray(npz["motion_norm"], dtype=np.float32)
                x = torch.from_numpy(motion_norm[None, ...]).to(device)
                mu, _ = vae.encode(x)  # [1,latent,T]
                z_mu = mu.mean(dim=2)[0].detach().cpu().float()  # [latent]
                self.items.append((id2emb[sid], motion_norm, z_mu, sid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        emb, motion_norm, z_mu, sid = self.items[idx]
        return {
            "embedding": torch.from_numpy(np.asarray(emb, dtype=np.float32)),
            "motion_norm": torch.from_numpy(np.asarray(motion_norm, dtype=np.float32)),
            "z_mu": z_mu,
            "sample_id": sid,
        }


class TextToLatentMLP(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, num_layers - 1)):
            layers.extend([nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)])
            d = hidden_dim
        layers.append(nn.Linear(d, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model, vae, loader, device):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for b in loader:
            emb = b["embedding"].to(device)
            motion = b["motion_norm"].to(device)
            z_mu = b["z_mu"].to(device)
            z_pred = model(emb)
            latent_mse = ((z_pred - z_mu) ** 2).mean()
            dec = vae.decode(z_pred[:, :, None].repeat(1, 1, motion.shape[1]))
            motion_loss = weighted_motion_mse(dec, motion)
            loss = latent_mse + motion_loss
            tot += float(loss.item())
            n += 1
    return tot / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", type=Path, required=True)
    ap.add_argument("--val_manifest", type=Path, required=True)
    ap.add_argument("--text_embeddings_dir", type=Path, required=True)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    tr_rows = read_jsonl(args.train_manifest)
    va_rows = read_jsonl(args.val_manifest)
    tr_emb = build_id_to_embedding(args.text_embeddings_dir / "train")
    va_emb = build_id_to_embedding(args.text_embeddings_dir / "val")

    ckpt = torch.load(args.vae_checkpoint, map_location=args.device)
    latent_dim_ckpt = int(ckpt.get("args", {}).get("latent_dim", args.latent_dim))
    latent_dim = int(args.latent_dim if args.latent_dim else latent_dim_ckpt)
    vae = TemporalConvVAE(in_dim=56, latent_dim=latent_dim_ckpt).to(args.device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    ds_tr = TextLatentDataset(tr_rows, tr_emb, vae, args.device)
    ds_va = TextLatentDataset(va_rows, va_emb, vae, args.device)
    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise SystemExit("empty dataset after sample_id matching")

    emb_dim = ds_tr[0]["embedding"].numel()
    model = TextToLatentMLP(emb_dim, latent_dim, args.hidden_dim, args.num_layers).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    best = 1e18
    log_path = args.output_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as lf:
        for ep in range(1, args.epochs + 1):
            model.train()
            tr_tot = 0.0
            n = 0
            for b in dl_tr:
                emb = b["embedding"].to(args.device)
                motion = b["motion_norm"].to(args.device)
                z_mu = b["z_mu"].to(args.device)
                z_pred = model(emb)
                latent_mse = ((z_pred - z_mu) ** 2).mean()
                dec = vae.decode(z_pred[:, :, None].repeat(1, 1, motion.shape[1]))
                motion_loss = weighted_motion_mse(dec, motion)
                loss = latent_mse + motion_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                tr_tot += float(loss.item())
                n += 1
            tr_loss = tr_tot / max(1, n)
            va_loss = evaluate(model, vae, dl_va, args.device)
            rec = {"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss}
            lf.write(json.dumps(rec) + "\n")
            lf.flush()
            print(f"epoch={ep} train={tr_loss:.6f} val={va_loss:.6f}")

            out_ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "embedding_dim": emb_dim,
                "latent_dim": latent_dim,
                "vae_checkpoint": str(args.vae_checkpoint),
            }
            torch.save(out_ckpt, args.output_dir / "last.pt")
            if va_loss < best:
                best = va_loss
                torch.save(out_ckpt, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
