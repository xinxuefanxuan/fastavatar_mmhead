#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from motion_model.models import TemporalConvVAE

DEFAULT_DEBUG_PROMPTS = [
    "turn left",
    "turn right",
    "smile",
    "turn left and smile",
]


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


def build_label_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id", ""))
        label = row.get("label")
        if sid and isinstance(label, str) and label:
            mapping[sid] = label
    return mapping


def weighted_motion_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    w = torch.ones((1, 1, gt.shape[-1]), dtype=gt.dtype, device=gt.device)
    w[..., 50:53] = 10.0
    w[..., 53:56] = 10.0
    return ((pred - gt) ** 2 * w).mean()


def parse_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


class TextLatentDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "embedding": torch.from_numpy(np.asarray(it["embedding"], dtype=np.float32)),
            "motion_norm": torch.from_numpy(np.asarray(it["motion_norm"], dtype=np.float32)),
            "z_mu": torch.from_numpy(np.asarray(it["z_mu"], dtype=np.float32)),
            "sample_id": it["sample_id"],
            "label": it.get("label", "unknown"),
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


def build_items(
    manifest_rows: list[dict],
    id2emb: dict[str, np.ndarray],
    vae: TemporalConvVAE,
    device: str,
    label_map: dict[str, str],
    include_labels: set[str] | None,
    max_per_label: int | None,
    max_samples: int | None,
) -> list[dict]:
    items: list[dict] = []
    counts = defaultdict(int)
    with torch.no_grad():
        for r in manifest_rows:
            sid = str(r.get("sample_id", ""))
            if sid not in id2emb:
                continue
            label = label_map.get(sid, "unknown")
            if include_labels is not None and label not in include_labels:
                continue
            if max_per_label is not None and counts[label] >= max_per_label:
                continue

            npz = np.load(Path(r["npz_path"]), allow_pickle=True)
            motion_norm = np.asarray(npz["motion_norm"], dtype=np.float32)
            x = torch.from_numpy(motion_norm[None, ...]).to(device)
            mu, _ = vae.encode(x)  # [1,latent,T]
            z_mu = mu.mean(dim=2)[0].detach().cpu().numpy().astype(np.float32)
            items.append(
                {
                    "embedding": id2emb[sid],
                    "motion_norm": motion_norm,
                    "z_mu": z_mu,
                    "sample_id": sid,
                    "label": label,
                }
            )
            counts[label] += 1
            if max_samples is not None and len(items) >= max_samples:
                break
    return items


def summarize_label_distribution(items: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter([it.get("label", "unknown") for it in items]).items(), key=lambda kv: kv[0]))


def make_balanced_sampler(items: list[dict]) -> WeightedRandomSampler:
    labels = [it.get("label", "unknown") for it in items]
    c = Counter(labels)
    weights = np.asarray([1.0 / max(1, c[l]) for l in labels], dtype=np.float64)
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)


def tensor_stats(t: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "norm_mean": float(t.norm(dim=1).mean().item()),
    }


def eval_epoch(model, vae, loader, device):
    model.eval()
    n = 0
    acc = defaultdict(float)
    all_z_pred, all_z_mu = [], []
    dec_stats_acc = defaultdict(float)

    with torch.no_grad():
        for b in loader:
            emb = b["embedding"].to(device)
            motion = b["motion_norm"].to(device)
            z_mu = b["z_mu"].to(device)
            z_pred = model(emb)
            dec = vae.decode(z_pred[:, :, None].repeat(1, 1, motion.shape[1]))

            latent_mse = ((z_pred - z_mu) ** 2).mean()
            motion_loss = weighted_motion_mse(dec, motion)
            total_loss = latent_mse + motion_loss

            acc["latent_mse"] += float(latent_mse.item())
            acc["decoded_motion_loss"] += float(motion_loss.item())
            acc["total_loss"] += float(total_loss.item())

            expr = dec[:, :, :50].norm(dim=2)
            head = dec[:, :, 50:53].norm(dim=2)
            jaw = dec[:, :, 53:56].norm(dim=2)
            yaw = dec[:, :, 51]

            dec_stats_acc["expr_norm_mean"] += float(expr.mean().item())
            dec_stats_acc["expr_norm_max"] += float(expr.max().item())
            dec_stats_acc["head_norm_mean"] += float(head.mean().item())
            dec_stats_acc["head_norm_max"] += float(head.max().item())
            dec_stats_acc["jaw_norm_mean"] += float(jaw.mean().item())
            dec_stats_acc["jaw_norm_max"] += float(jaw.max().item())
            dec_stats_acc["yaw_min"] += float(yaw.min().item())
            dec_stats_acc["yaw_max"] += float(yaw.max().item())

            all_z_pred.append(z_pred.detach().cpu())
            all_z_mu.append(z_mu.detach().cpu())
            n += 1

    if n == 0:
        raise RuntimeError("validation loader is empty")
    metrics = {k: v / n for k, v in acc.items()}
    dec_metrics = {k: v / n for k, v in dec_stats_acc.items()}

    z_pred = torch.cat(all_z_pred, dim=0)
    z_mu = torch.cat(all_z_mu, dim=0)
    cos = F.cosine_similarity(z_pred, z_mu, dim=1).mean().item()
    z_stats = {
        "z_mu": tensor_stats(z_mu),
        "z_pred": tensor_stats(z_pred),
        "z_pred_var_mean": float(z_pred.var(dim=0, unbiased=False).mean().item()),
        "cosine_similarity": float(cos),
    }
    return metrics, z_stats, dec_metrics


def eval_debug_prompts(model, vae, prompts: list[str], device: str, emb_model):
    if not prompts:
        return {}
    embs = emb_model.encode(prompts, convert_to_numpy=True, batch_size=min(64, len(prompts)), normalize_embeddings=False)
    out = {}
    with torch.no_grad():
        for p, e in zip(prompts, embs):
            z = model(torch.from_numpy(np.asarray(e, dtype=np.float32)).to(device).unsqueeze(0))
            dec = vae.decode(z[:, :, None].repeat(1, 1, 64))[0]
            expr = dec[:, :50].norm(dim=1)
            head = dec[:, 50:53].norm(dim=1)
            jaw = dec[:, 53:56].norm(dim=1)
            yaw = dec[:, 51]
            out[p] = {
                "expr_norm_mean": float(expr.mean().item()),
                "expr_norm_max": float(expr.max().item()),
                "head_norm_mean": float(head.mean().item()),
                "head_norm_max": float(head.max().item()),
                "jaw_norm_mean": float(jaw.mean().item()),
                "jaw_norm_max": float(jaw.max().item()),
                "yaw_min": float(yaw.min().item()),
                "yaw_max": float(yaw.max().item()),
            }
    return out


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
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--debug_eval_prompts", type=str, default=",".join(DEFAULT_DEBUG_PROMPTS))
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_val_samples", type=int, default=None)
    ap.add_argument("--label_jsonl", type=Path, default=None)
    ap.add_argument("--include_labels", type=str, default=None)
    ap.add_argument("--max_per_label", type=int, default=None)
    ap.add_argument("--balanced_sampler", action="store_true")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    tr_rows = read_jsonl(args.train_manifest)
    va_rows = read_jsonl(args.val_manifest)
    tr_emb = build_id_to_embedding(args.text_embeddings_dir / "train")
    va_emb = build_id_to_embedding(args.text_embeddings_dir / "val")
    label_map = build_label_map(args.label_jsonl)
    include_labels = set(parse_list(args.include_labels)) if args.include_labels else None

    print(f"[Debug] max_train_samples={args.max_train_samples}")
    print(f"[Debug] max_val_samples={args.max_val_samples}")

    ckpt = torch.load(args.vae_checkpoint, map_location=args.device)
    latent_dim_ckpt = int(ckpt.get("args", {}).get("latent_dim", args.latent_dim))
    latent_dim = int(args.latent_dim if args.latent_dim else latent_dim_ckpt)
    vae = TemporalConvVAE(in_dim=56, latent_dim=latent_dim_ckpt).to(args.device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    tr_items = build_items(tr_rows, tr_emb, vae, args.device, label_map, include_labels, args.max_per_label, args.max_train_samples)
    va_items = build_items(va_rows, va_emb, vae, args.device, label_map, include_labels, args.max_per_label, args.max_val_samples)

    if len(tr_items) == 0 or len(va_items) == 0:
        raise SystemExit("empty dataset after sample_id matching/filtering")

    print("[LabelDist][train]", summarize_label_distribution(tr_items))
    print("[LabelDist][val]", summarize_label_distribution(va_items))

    ds_tr = TextLatentDataset(tr_items)
    ds_va = TextLatentDataset(va_items)

    emb_dim = ds_tr[0]["embedding"].numel()
    model = TextToLatentMLP(emb_dim, latent_dim, args.hidden_dim, args.num_layers).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    tr_sampler = make_balanced_sampler(tr_items) if args.balanced_sampler else None
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=(tr_sampler is None), sampler=tr_sampler)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)

    debug_prompts = parse_list(args.debug_eval_prompts)
    debug_encoder = SentenceTransformer("/home/yuanyuhao/models/all-MiniLM-L6-v2", device=args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    best = 1e18
    log_path = args.output_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as lf:
        for ep in range(1, args.epochs + 1):
            model.train()
            n = 0
            sums = defaultdict(float)
            z_pred_batches, z_mu_batches = [], []
            for b in dl_tr:
                emb = b["embedding"].to(args.device)
                motion = b["motion_norm"].to(args.device)
                z_mu = b["z_mu"].to(args.device)
                z_pred = model(emb)
                dec = vae.decode(z_pred[:, :, None].repeat(1, 1, motion.shape[1]))
                latent_mse = ((z_pred - z_mu) ** 2).mean()
                motion_loss = weighted_motion_mse(dec, motion)
                total_loss = latent_mse + motion_loss
                opt.zero_grad()
                total_loss.backward()
                opt.step()

                sums["latent_mse"] += float(latent_mse.item())
                sums["decoded_motion_loss"] += float(motion_loss.item())
                sums["total_loss"] += float(total_loss.item())
                z_pred_batches.append(z_pred.detach().cpu())
                z_mu_batches.append(z_mu.detach().cpu())
                n += 1

            tr_metrics = {k: v / max(1, n) for k, v in sums.items()}
            z_pred_cat = torch.cat(z_pred_batches, dim=0)
            z_mu_cat = torch.cat(z_mu_batches, dim=0)
            tr_z_stats = {
                "z_mu": tensor_stats(z_mu_cat),
                "z_pred": tensor_stats(z_pred_cat),
                "z_pred_var_mean": float(z_pred_cat.var(dim=0, unbiased=False).mean().item()),
                "cosine_similarity": float(F.cosine_similarity(z_pred_cat, z_mu_cat, dim=1).mean().item()),
            }

            va_metrics, va_z_stats, va_dec_stats = eval_epoch(model, vae, dl_va, args.device)
            rec = {
                "epoch": ep,
                "train": tr_metrics,
                "val": va_metrics,
                "train_z_stats": tr_z_stats,
                "val_z_stats": va_z_stats,
                "val_decoded_motion_stats": va_dec_stats,
            }

            if args.eval_every > 0 and (ep % args.eval_every == 0 or ep == 1):
                rec["debug_eval_prompts"] = eval_debug_prompts(model, vae, debug_prompts, args.device, debug_encoder)

            lf.write(json.dumps(rec) + "\n")
            lf.flush()

            print(
                f"epoch={ep} train_total={tr_metrics['total_loss']:.6f} val_total={va_metrics['total_loss']:.6f} "
                f"train_latent={tr_metrics['latent_mse']:.6f} val_latent={va_metrics['latent_mse']:.6f}"
            )

            out_ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "embedding_dim": emb_dim,
                "latent_dim": latent_dim,
                "vae_checkpoint": str(args.vae_checkpoint),
            }
            torch.save(out_ckpt, args.output_dir / "last.pt")
            if va_metrics["total_loss"] < best:
                best = va_metrics["total_loss"]
                torch.save(out_ckpt, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
