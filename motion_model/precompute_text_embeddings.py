#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid jsonl at line {i}: {exc}")
    return rows


def batched_encode_sentence_transformer(
    texts: list[str],
    encoder_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise SystemExit(f"sentence-transformers is required for encoder_type=sentence_transformer: {exc}")

    model = SentenceTransformer(encoder_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    return np.asarray(emb, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--text_field", type=str, default="searchable_text")
    ap.add_argument("--encoder_type", type=str, default="sentence_transformer")
    ap.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    rows = read_jsonl(args.manifest)
    texts: list[str] = []
    sample_ids: list[str] = []

    for r in rows:
        sample_ids.append(str(r.get("sample_id", "")))
        texts.append(str(r.get(args.text_field, "")))

    if args.encoder_type != "sentence_transformer":
        raise SystemExit(f"unsupported encoder_type: {args.encoder_type}")

    embeddings = batched_encode_sentence_transformer(
        texts=texts,
        encoder_name=args.encoder_name,
        batch_size=args.batch_size,
        device=args.device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = args.output_dir / "embeddings.npy"
    sample_ids_path = args.output_dir / "sample_ids.json"
    texts_path = args.output_dir / "texts.json"
    meta_path = args.output_dir / "meta.json"

    np.save(emb_path, embeddings)
    sample_ids_path.write_text(json.dumps(sample_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    texts_path.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "num_samples": int(len(rows)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "manifest": str(args.manifest),
        "text_field": args.text_field,
        "encoder_type": args.encoder_type,
        "encoder_name": args.encoder_name,
        "batch_size": int(args.batch_size),
        "device": args.device,
        "embeddings_path": str(emb_path),
        "sample_ids_path": str(sample_ids_path),
        "texts_path": str(texts_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Embedding] number of samples: {len(rows)}")
    print(f"[Embedding] embedding dim: {meta['embedding_dim']}")
    print(f"[Embedding] output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
