#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_model.channel_vae import ChannelTemporalVAE


def fit_length(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] >= target_len:
        return x[:target_len]
    return np.concatenate([x, np.repeat(x[-1:], target_len - x.shape[0], axis=0)], axis=0)


def load_motion_norm(npz_path: Path, norm_stats: Path, motion_key: str, target_len: int):
    arrs = np.load(npz_path, allow_pickle=False)
    st = json.loads(norm_stats.read_text())
    mean = np.asarray(st["mean"], dtype=np.float32)
    std = np.asarray(st["std"], dtype=np.float32)
    if "motion_norm" in arrs:
        m = np.asarray(arrs["motion_norm"], dtype=np.float32)
    else:
        raw = np.asarray(arrs[motion_key], dtype=np.float32)
        m = (raw - mean[None, :]) / np.maximum(std[None, :], 1e-8)
    m = fit_length(m[:, :56], target_len)
    return m, mean, std


def stats(m):
    e = np.linalg.norm(m[:, :50], axis=1)
    h = np.linalg.norm(m[:, 50:53], axis=1)
    j = np.linalg.norm(m[:, 53:56], axis=1)
    y = m[:, 51]
    return {
        "expr norm mean/max": (float(e.mean()), float(e.max())),
        "head norm mean/max": (float(h.mean()), float(h.max())),
        "jaw norm mean/max": (float(j.mean()), float(j.max())),
        "yaw min/max": (float(y.min()), float(y.max())),
    }


def load_channel_model(path: Path, device: torch.device):
    ck = torch.load(path, map_location=device)
    model = ChannelTemporalVAE(int(ck["input_dim"]), int(ck["target_len"]), int(ck["latent_dim"]), int(ck["hidden_dim"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model


def recon_channel(model, x, device, use_mu: bool):
    xt = torch.from_numpy(x[None].astype(np.float32)).to(device)
    with torch.no_grad():
        mu, logvar = model.encode(xt)
        z = mu if use_mu else model.reparameterize(mu, logvar)
        rec = model.decode(z)[0].cpu().numpy().astype(np.float32)
    return rec


def reconstruct_one(input_npz: Path, out_npz: Path, expr_model, head_model, jaw_model, norm_stats: Path, target_len: int, device, use_mu: bool, motion_key: str):
    m_norm, mean, std = load_motion_norm(input_npz, norm_stats, motion_key, target_len)
    expr, head, jaw = m_norm[:, :50], m_norm[:, 50:53], m_norm[:, 53:56]
    expr_r = recon_channel(expr_model, expr, device, use_mu)
    head_r = recon_channel(head_model, head, device, use_mu)
    jaw_r = recon_channel(jaw_model, jaw, device, use_mu)
    recon_norm = np.concatenate([expr_r, head_r, jaw_r], axis=1)
    recon_raw = recon_norm * std[None, :56] + mean[None, :56]

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        motion=recon_raw.astype(np.float32),
        motion_raw=recon_raw.astype(np.float32),
        motion_norm=recon_norm.astype(np.float32),
        expr_delta=recon_raw[:, :50].astype(np.float32),
        head_delta=recon_raw[:, 50:53].astype(np.float32),
        jaw_delta=recon_raw[:, 53:56].astype(np.float32),
        motion_input=(m_norm * std[None, :56] + mean[None, :56]).astype(np.float32),
        motion_input_norm=m_norm.astype(np.float32),
        motion_recon_norm=recon_norm.astype(np.float32),
    )

    print(f"[Input] {input_npz}")
    for k, v in stats(m_norm).items():
        print(f"  {k}: {v[0]:.6f} / {v[1]:.6f}")
    print("[Recon]")
    for k, v in stats(recon_norm).items():
        print(f"  {k}: {v[0]:.6f} / {v[1]:.6f}")
    mse = ((recon_norm - m_norm) ** 2).mean()
    print(f"[Error] channel MSE={mse:.8f}, expr={((expr_r-expr)**2).mean():.8f}, head={((head_r-head)**2).mean():.8f}, jaw={((jaw_r-jaw)**2).mean():.8f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", type=Path, default=None)
    ap.add_argument("--input_manifest", type=Path, default=None)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--expr_checkpoint", type=Path, required=True)
    ap.add_argument("--head_checkpoint", type=Path, required=True)
    ap.add_argument("--jaw_checkpoint", type=Path, required=True)
    ap.add_argument("--norm_stats", type=Path, default=Path("outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json"))
    ap.add_argument("--output_npz", type=Path, default=None)
    ap.add_argument("--target_len", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_mu", action="store_true", default=True)
    ap.add_argument("--motion_key", type=str, default="motion")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    expr_model = load_channel_model(args.expr_checkpoint, device)
    head_model = load_channel_model(args.head_checkpoint, device)
    jaw_model = load_channel_model(args.jaw_checkpoint, device)

    if args.input_manifest is not None:
        if args.output_dir is None:
            raise SystemExit("--output_dir is required with --input_manifest")
        rows = []
        with args.input_manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        n = 0
        for r in rows:
            p = r.get("npz_path") or r.get("motion_path")
            sid = r.get("sample_id", f"sample_{n:06d}")
            if not p:
                continue
            ip = Path(p)
            if not ip.exists():
                continue
            op = args.output_dir / f"{sid}.npz"
            reconstruct_one(ip, op, expr_model, head_model, jaw_model, args.norm_stats, args.target_len, device, args.use_mu, args.motion_key)
            n += 1
            if args.max_samples is not None and n >= args.max_samples:
                break
    else:
        if args.input_npz is None or args.output_npz is None:
            raise SystemExit("single mode requires --input_npz and --output_npz")
        reconstruct_one(args.input_npz, args.output_npz, expr_model, head_model, jaw_model, args.norm_stats, args.target_len, device, args.use_mu, args.motion_key)


if __name__ == "__main__":
    main()
