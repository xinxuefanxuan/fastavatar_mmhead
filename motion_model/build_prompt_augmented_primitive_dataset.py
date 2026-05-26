#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SINGLE_TEMPLATES = {
    "turn_left": [
        "turn left",
        "look left",
        "look to the left",
        "look toward the left",
        "face left",
        "turn your head left",
        "rotate head left",
        "glance left",
        "slowly turn left",
        "slightly turn left",
        "make a left turn",
        "move your head to the left",
    ],
    "turn_right": [
        "turn right",
        "look right",
        "look to the right",
        "look toward the right",
        "face right",
        "turn your head right",
        "rotate head right",
        "glance right",
        "slowly turn right",
        "slightly turn right",
        "make a right turn",
        "move your head to the right",
    ],
    "smile": [
        "smile",
        "smiling",
        "make a smile",
        "slight smile",
        "big smile",
        "happy smile",
        "broad smile",
        "grin",
        "smile softly",
        "smile happily",
        "show a smile",
    ],
    "mouth_open": [
        "open mouth",
        "mouth open",
        "open your mouth",
        "jaw open",
        "slightly open mouth",
        "open the jaw",
        "part the lips",
        "lips parted",
        "mouth slightly open",
    ],
    "nod": [
        "nod",
        "nodding",
        "nod your head",
        "move head up and down",
        "slight nod",
        "give a nod",
        "gently nod",
    ],
    "neutral": [
        "neutral",
        "stay still",
        "no motion",
        "keep a neutral face",
        "remain neutral",
        "keep still",
    ],
}

COMPOSE_TEMPLATES = {
    ("turn_left", "smile"): [
        "turn left and smile",
        "smile while turning left",
        "look left and smile",
        "face left with a smile",
        "turn your head left and smile",
        "slowly turn left while smiling",
        "slightly turn left and smile",
    ],
    ("turn_right", "smile"): [
        "turn right and smile",
        "smile while turning right",
        "look right and smile",
        "face right with a smile",
        "turn your head right and smile",
        "slowly turn right while smiling",
        "slightly turn right and smile",
    ],
    ("mouth_open", "smile"): [
        "smile with mouth open",
        "open your mouth and smile",
        "smile while opening mouth",
        "happy open-mouth smile",
        "grin with mouth open",
    ],
    ("nod", "smile"): [
        "nod and smile",
        "smile while nodding",
        "nod your head and smile",
        "gently nod with a smile",
    ],
    ("turn_left", "mouth_open"): [
        "turn left and open mouth",
        "look left with mouth open",
        "open mouth while turning left",
    ],
    ("turn_right", "mouth_open"): [
        "turn right and open mouth",
        "look right with mouth open",
        "open mouth while turning right",
    ],
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_texts(texts: list[str], encoder_name: str, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder_name, device=device)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
    arr = np.asarray(emb, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise SystemExit("embeddings contain NaN/Inf")
    return arr


def stratified_split(items: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in items:
        key = row["labels"][0] if len(row["labels"]) == 1 else "+".join(sorted(row["labels"]))
        by_label[key].append(row)

    train, val = [], []
    for _, rows in by_label.items():
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
    out: Counter[str] = Counter()
    for r in rows:
        out["+".join(sorted(r["labels"]))] += 1
    return out


def make_variants(template: str, repeat: int) -> list[str]:
    if repeat <= 1:
        return [template]
    variants = [template, f"please {template}", f"make the avatar {template}", f"the person should {template}"]
    out = []
    for i in range(repeat):
        out.append(variants[i % len(variants)])
    return out


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
    ap.add_argument("--synthetic_repeat", type=int, default=20)
    ap.add_argument("--composition_repeat", type=int, default=30)
    ap.add_argument("--synthetic_weight_mode", type=str, default="duplicate")
    ap.add_argument("--include_direction_hard_negatives", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    primitive_classes = [x.strip() for x in args.primitive_classes.split(",") if x.strip()]
    allowed = set(primitive_classes)

    labels_rows = read_jsonl(args.label_jsonl)
    manifest_rows = read_jsonl(args.train_manifest) + read_jsonl(args.val_manifest)
    manifest_map = {str(r.get("sample_id", "")).strip(): r for r in manifest_rows if str(r.get("sample_id", "")).strip()}

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
    capped_real = []
    for label in primitive_classes:
        rows = real_by_label.get(label, [])
        rng.shuffle(rows)
        if args.max_real_per_label and args.max_real_per_label > 0:
            rows = rows[: args.max_real_per_label]
        capped_real.extend(rows)
    train_real, val_real = stratified_split(capped_real, args.val_ratio, args.seed)

    synthetic_rows = []
    sid_counter = 0

    for label, templates in SINGLE_TEMPLATES.items():
        if label not in allowed:
            continue
        for t in templates:
            for text in make_variants(t, args.synthetic_repeat):
                sid_counter += 1
                synthetic_rows.append({
                    "sample_id": f"synthetic_{label}_{sid_counter:06d}",
                    "text": text,
                    "labels": [label],
                    "source": "synthetic",
                })

    for labels, templates in COMPOSE_TEMPLATES.items():
        if not set(labels).issubset(allowed):
            continue
        for t in templates:
            for text in make_variants(t, args.composition_repeat):
                sid_counter += 1
                synthetic_rows.append({
                    "sample_id": f"synthetic_{'_'.join(labels)}_{sid_counter:06d}",
                    "text": text,
                    "labels": list(labels),
                    "source": "synthetic",
                })

    if args.include_direction_hard_negatives:
        pairs = [("turn left", "turn_left"), ("turn right", "turn_right"), ("look left", "turn_left"), ("look right", "turn_right"), ("face left", "turn_left"), ("face right", "turn_right")]
        for text, label in pairs:
            if label in allowed:
                sid_counter += 1
                synthetic_rows.append({
                    "sample_id": f"synthetic_hardneg_{sid_counter:06d}",
                    "text": text,
                    "labels": [label],
                    "source": "synthetic",
                })

    train_syn, val_syn = stratified_split(synthetic_rows, args.val_ratio, args.seed + 7)

    final_train = train_real + train_syn
    final_val = val_real + val_syn

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", final_train)
    write_jsonl(args.output_dir / "val.jsonl", final_val)

    train_emb = encode_texts([r["text"] for r in final_train], args.text_encoder, args.device)
    val_emb = encode_texts([r["text"] for r in final_val], args.text_encoder, args.device)
    if train_emb.shape[0] != len(final_train) or val_emb.shape[0] != len(final_val):
        raise SystemExit("embedding row count mismatch")

    np.save(args.output_dir / "train_embeddings.npy", train_emb)
    np.save(args.output_dir / "val_embeddings.npy", val_emb)
    (args.output_dir / "train_sample_ids.json").write_text(json.dumps([r["sample_id"] for r in final_train], ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "val_sample_ids.json").write_text(json.dumps([r["sample_id"] for r in final_val], ensure_ascii=False, indent=2), encoding="utf-8")

    train_label_dist = count_label_vectors(final_train)
    val_label_dist = count_label_vectors(final_val)
    train_source = Counter(r["source"] for r in final_train)
    val_source = Counter(r["source"] for r in final_val)
    train_multi = sum(1 for r in final_train if len(r["labels"]) > 1)
    val_multi = sum(1 for r in final_val if len(r["labels"]) > 1)
    syn_single = sum(1 for r in synthetic_rows if len(r["labels"]) == 1)
    syn_multi = sum(1 for r in synthetic_rows if len(r["labels"]) > 1)

    meta = {
        "primitive_classes": primitive_classes,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "max_real_per_label": args.max_real_per_label,
        "synthetic_repeat": args.synthetic_repeat,
        "composition_repeat": args.composition_repeat,
        "synthetic_weight_mode": args.synthetic_weight_mode,
        "include_direction_hard_negatives": bool(args.include_direction_hard_negatives),
        "num_label_rows": len(labels_rows),
        "missing_manifest_count": missing_manifest,
        "synthetic_single_label_examples": syn_single,
        "synthetic_multi_label_examples": syn_multi,
        "train_count": len(final_train),
        "val_count": len(final_val),
        "train_source": dict(train_source),
        "val_source": dict(val_source),
        "train_multilabel_count": train_multi,
        "val_multilabel_count": val_multi,
        "train_label_distribution": dict(train_label_dist),
        "val_label_distribution": dict(val_label_dist),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Build] train={len(final_train)} val={len(final_val)}")
    print(f"[Build] real/synthetic train={dict(train_source)} val={dict(val_source)}")
    print(f"[Build] synthetic(single)={syn_single} synthetic(multi)={syn_multi}")
    print(f"[Build] multi-label train={train_multi} val={val_multi}")
    print(f"[Build] train label distribution={dict(train_label_dist)}")
    print(f"[Build] val label distribution={dict(val_label_dist)}")
    print(f"[Build] missing manifest rows from labels={missing_manifest}")


if __name__ == "__main__":
    main()
