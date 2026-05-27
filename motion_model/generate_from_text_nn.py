#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(x: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(x, axis=-1, keepdims=True)
    d = np.maximum(d, 1e-8)
    return x / d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--train_manifest", type=Path, required=True)
    ap.add_argument("--text_embeddings_dir", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--encoder_name", type=str, default="/home/yuanyuhao/models/all-MiniLM-L6-v2")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    rows = read_jsonl(args.train_manifest)
    id_to_row = {str(r.get("sample_id", "")): r for r in rows}

    emb = np.load(args.text_embeddings_dir / "train" / "embeddings.npy").astype(np.float32)
    sample_ids = json.loads((args.text_embeddings_dir / "train" / "sample_ids.json").read_text(encoding="utf-8"))
    sample_ids = [str(x) for x in sample_ids]
    if emb.shape[0] != len(sample_ids):
        raise RuntimeError("embeddings rows and sample_ids size mismatch")

    encoder = SentenceTransformer(args.encoder_name)
    q = encoder.encode([args.prompt], convert_to_numpy=True).astype(np.float32)[0]

    sims = (norm(emb) @ norm(q[None, :]).T).squeeze(1)
    order = np.argsort(-sims)
    top_k = max(1, min(args.top_k, len(order)))

    print(f"[NN] prompt={args.prompt}")
    print(f"[NN] top_k={top_k}")
    for rank in range(top_k):
        i = int(order[rank])
        sid = sample_ids[i]
        r = id_to_row.get(sid, {})
        print(
            f"#{rank+1} sample_id={sid} sim={float(sims[i]):.6f} "
            f"text={r.get('searchable_text', '')} "
            f"npz_path={r.get('npz_path', r.get('motion_path', ''))}"
        )

    best_sid = sample_ids[int(order[0])]
    best_row = id_to_row.get(best_sid)
    if best_row is None:
        raise RuntimeError(f"top-1 sample_id not found in manifest: {best_sid}")

    src_npz = Path(best_row.get("npz_path", ""))
    if not src_npz.exists():
        raise RuntimeError(f"npz_path not found for top-1 sample: {src_npz}")

    src = np.load(src_npz, allow_pickle=True)
    motion = np.asarray(src["motion"], dtype=np.float32)
    motion_raw = np.asarray(src["motion_raw"], dtype=np.float32) if "motion_raw" in src else motion.copy()
    motion_norm = np.asarray(src["motion_norm"], dtype=np.float32) if "motion_norm" in src else motion.copy()

    expr = motion[:, :50]
    head = motion[:, 50:53]
    jaw = motion[:, 53:56]

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        motion=motion,
        motion_raw=motion_raw,
        motion_norm=motion_norm,
        expr_delta=expr,
        head_delta=head,
        jaw_delta=jaw,
    )
    print(f"[NN] output_npz={args.output_npz}")


if __name__ == "__main__":
    main()
