#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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
                raise SystemExit(f"invalid jsonl at {path}:{i}: {exc}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_texts(texts: list[str], encoder_name: str, device: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise SystemExit(f"sentence-transformers is required: {exc}")

    model = SentenceTransformer(encoder_name, device=device)
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    arr = np.asarray(emb, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise SystemExit("embeddings contain NaN/Inf")
    return arr


def stratified_split(items: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in items:
        by_label[row["labels"][0]].append(row)

    train: list[dict] = []
    val: list[dict] = []
    for label, rows in by_label.items():
        rng.shuffle(rows)
        n = len(rows)
        n_val = int(round(n * val_ratio))
        if n >= 2:
            n_val = min(max(1, n_val), n - 1)
        else:
            n_val = 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    return train, val


def count_label_vectors(rows: list[dict]) -> Counter[str]:
    cnt: Counter[str] = Counter()
    for r in rows:
        key = "+".join(sorted(r["labels"]))
        cnt[key] += 1
    return cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_jsonl", type=Path, default=Path("outputs/mmhead_debug/primitive_labels_v2/all_labeled_normalized.jsonl"))
    ap.add_argument("--train_manifest", type=Path, default=Path("outputs/mmhead_debug/motion_dataset_v1_ae_debug/train.jsonl"))
    ap.add_argument("--val_manifest", type=Path, default=Path("outputs/mmhead_debug/motion_dataset_v1_ae_debug/val.jsonl"))
    ap.add_argument("--text_encoder", type=str, default="/home/yuanyuhao/models/all-MiniLM-L6-v2")
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/mmhead_debug/prompt_augmented_primitive_dataset_v1"))
    ap.add_argument("--primitive_classes", type=str, default="turn_left,turn_right,nod,smile,mouth_open,neutral")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_real_per_label", type=int, default=450)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    primitive_classes = [x.strip() for x in args.primitive_classes.split(",") if x.strip()]
    allowed = set(primitive_classes)

    labels_rows = read_jsonl(args.label_jsonl)
    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)

    manifest_map: dict[str, dict] = {}
    for r in train_rows + val_rows:
        sid = str(r.get("sample_id", "")).strip()
        if sid:
            manifest_map[sid] = r

    real_by_label: dict[str, list[dict]] = defaultdict(list)
    missing_manifest = 0
    for r in labels_rows:
        sid = str(r.get("sample_id", "")).strip()
        label = str(r.get("label", "")).strip()
        if not sid or label not in allowed:
            continue
        m = manifest_map.get(sid)
        if m is None:
            missing_manifest += 1
            continue
        row = {
            "sample_id": f"real_{sid}",
            "text": str(m.get("searchable_text", "")),
            "labels": [label],
            "source": "real",
            "original_sample_id": sid,
        }
        if "npz_path" in m:
            row["npz_path"] = m.get("npz_path")
        if "motion_path" in m:
            row["motion_path"] = m.get("motion_path")
        real_by_label[label].append(row)

    rng = random.Random(args.seed)
    capped_real: list[dict] = []
    for label in primitive_classes:
        rows = real_by_label.get(label, [])
        rng.shuffle(rows)
        if args.max_real_per_label is not None and args.max_real_per_label > 0:
            rows = rows[: args.max_real_per_label]
        capped_real.extend(rows)

    train_real, val_real = stratified_split(capped_real, args.val_ratio, args.seed)

    single_templates = {
        "turn_left": ["turn left", "look left", "look to the left", "face left", "turn your head left"],
        "turn_right": ["turn right", "look right", "look to the right", "face right", "turn your head right"],
        "smile": ["smile", "smiling", "make a smile", "slight smile", "big smile", "happy smile", "grin"],
        "mouth_open": ["open mouth", "mouth open", "open your mouth", "jaw open"],
        "nod": ["nod", "nodding", "nod your head", "move head up and down"],
        "neutral": ["neutral", "stay still", "no motion", "keep a neutral face"],
    }
    multi_templates = {
        ("turn_left", "smile"): ["turn left and smile", "smile while turning left", "look left and smile"],
        ("turn_right", "smile"): ["turn right and smile", "smile while turning right", "look right and smile"],
        ("mouth_open", "smile"): ["smile with mouth open", "open your mouth and smile"],
        ("nod", "smile"): ["nod and smile", "smile while nodding"],
    }

    synthetic_rows: list[dict] = []
    sid_counter = 0
    for label, texts in single_templates.items():
        if label not in allowed:
            continue
        for t in texts:
            sid_counter += 1
            synthetic_rows.append(
                {
                    "sample_id": f"synthetic_{label}_{sid_counter:04d}",
                    "text": t,
                    "labels": [label],
                    "source": "synthetic",
                }
            )
    for labels, texts in multi_templates.items():
        if not set(labels).issubset(allowed):
            continue
        for t in texts:
            sid_counter += 1
            synthetic_rows.append(
                {
                    "sample_id": f"synthetic_{'_'.join(labels)}_{sid_counter:04d}",
                    "text": t,
                    "labels": list(labels),
                    "source": "synthetic",
                }
            )

    train_syn, val_syn = stratified_split(synthetic_rows, args.val_ratio, args.seed + 7)

    final_train = train_real + train_syn
    final_val = val_real + val_syn

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = args.output_dir / "train.jsonl"
    val_jsonl = args.output_dir / "val.jsonl"
    write_jsonl(train_jsonl, final_train)
    write_jsonl(val_jsonl, final_val)

    train_texts = [str(r.get("text", "")) for r in final_train]
    val_texts = [str(r.get("text", "")) for r in final_val]
    train_emb = encode_texts(train_texts, args.text_encoder, args.device)
    val_emb = encode_texts(val_texts, args.text_encoder, args.device)

    if train_emb.shape[0] != len(final_train) or val_emb.shape[0] != len(final_val):
        raise SystemExit("embedding row count mismatch with jsonl rows")

    train_ids = [r["sample_id"] for r in final_train]
    val_ids = [r["sample_id"] for r in final_val]

    np.save(args.output_dir / "train_embeddings.npy", train_emb)
    np.save(args.output_dir / "val_embeddings.npy", val_emb)
    (args.output_dir / "train_sample_ids.json").write_text(json.dumps(train_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "val_sample_ids.json").write_text(json.dumps(val_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    train_label_dist = count_label_vectors(final_train)
    val_label_dist = count_label_vectors(final_val)

    train_source = Counter(r["source"] for r in final_train)
    val_source = Counter(r["source"] for r in final_val)
    train_multi = sum(1 for r in final_train if len(r["labels"]) > 1)
    val_multi = sum(1 for r in final_val if len(r["labels"]) > 1)

    meta = {
        "primitive_classes": primitive_classes,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "max_real_per_label": args.max_real_per_label,
        "num_label_rows": len(labels_rows),
        "missing_manifest_count": missing_manifest,
        "train_count": len(final_train),
        "val_count": len(final_val),
        "train_source": dict(train_source),
        "val_source": dict(val_source),
        "train_multilabel_count": int(train_multi),
        "val_multilabel_count": int(val_multi),
        "train_label_distribution": dict(train_label_dist),
        "val_label_distribution": dict(val_label_dist),
        "text_encoder": args.text_encoder,
        "train_embeddings": str(args.output_dir / "train_embeddings.npy"),
        "val_embeddings": str(args.output_dir / "val_embeddings.npy"),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Build] train={len(final_train)} val={len(final_val)}")
    print(f"[Build] real/synthetic train={dict(train_source)} val={dict(val_source)}")
    print(f"[Build] multi-label train={train_multi} val={val_multi}")
    print(f"[Build] train label distribution={dict(train_label_dist)}")
    print(f"[Build] val label distribution={dict(val_label_dist)}")
    print(f"[Build] missing manifest rows from labels={missing_manifest}")
    print(f"[Build] output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
