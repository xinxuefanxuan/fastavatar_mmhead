#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

SYNONYMS = {
    "left": ["left", "turn left", "look left", "head left"],
    "right": ["right", "turn right", "look right", "head right"],
    "up": ["up", "look up", "head up", "raise head"],
    "down": ["down", "look down", "head down", "lower head"],
    "smile": ["smile", "smiling", "happy", "grin"],
    "laugh": ["laugh", "laughing", "big smile"],
    "angry": ["angry", "anger", "mad"],
    "sad": ["sad", "sadness", "unhappy"],
    "surprise": ["surprise", "surprised", "shocked"],
    "mouth": ["mouth", "open mouth", "jaw", "speaking", "talk"],
}
PHRASE_BONUS_TABLE = {"turn head left": 2.5, "turn head right": 2.5, "look left": 2.0, "look right": 2.0, "look up": 2.0, "look down": 2.0, "open mouth": 2.0, "close eyes": 2.0, "smile": 1.2, "angry": 1.2, "surprised": 1.2, "sad": 1.2}


SMILE_TERMS = {"smile", "smiling", "happy", "grin"}
EXPR_NEGATIVE_TERMS = ["blink", "kiss", "suck cheeks", "hide lips", "eyes wide", "lower eyebrows"]
LEFT_POSITIVE_PHRASES = ["turn left", "turning left", "head left", "look left", "left turn"]
RIGHT_POSITIVE_PHRASES = ["turn right", "turning right", "head right", "look right", "right turn"]
MIXED_DIRECTION_PHRASES = ["left and right", "right and left", "look around"]


def contains_any_phrase(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    n = normalize_text(text)
    return n.split() if n else []


def expand_prompt(prompt: str) -> Tuple[set[str], str]:
    norm = normalize_text(prompt)
    tokens = set(tokenize(prompt))
    for canon, syns in SYNONYMS.items():
        if canon in tokens or any(normalize_text(s) in norm for s in syns):
            tokens.update(tokenize(" ".join(syns)))
    return tokens, norm


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_text(entry: dict, prompt: str) -> tuple[float, list[str], float, float]:
    expanded_tokens, norm_prompt = expand_prompt(prompt)
    searchable = normalize_text(entry.get("searchable_text", ""))
    searchable_tokens = set(tokenize(searchable))
    overlap = sorted(expanded_tokens.intersection(searchable_tokens))
    token_score = float(len(overlap)) / max(len(expanded_tokens), 1) * 10.0
    phrase_score = 0.0
    for phrase, bonus in PHRASE_BONUS_TABLE.items():
        if phrase in norm_prompt and phrase in searchable:
            phrase_score += bonus
    if norm_prompt and norm_prompt in searchable:
        phrase_score += 1.0
    return token_score + phrase_score, overlap, token_score, phrase_score


def score_entry(entry: dict, prompt: str, channel: str = "all") -> dict:
    text_score, overlap, token_score, phrase_score = score_text(entry, prompt)
    s = entry.get("motion_stats", {})
    head = float(s.get("head_delta_max_norm", 0.0) or 0.0)
    expr = float(s.get("expr_delta_max_norm", 0.0) or 0.0)
    jaw = float(s.get("jaw_delta_max_norm", 0.0) or 0.0)
    head_vel = float(s.get("head_velocity_max_norm", 0.0) or 0.0)
    intensity = float(s.get("motion_intensity_score", 0.0) or 0.0)
    head_purity = float(s.get("head_purity_score", 0.0) or 0.0)
    expr_purity = float(s.get("expr_purity_score", 0.0) or 0.0)
    jaw_purity = float(s.get("jaw_purity_score", 0.0) or 0.0)

    if channel == "head":
        searchable_norm = normalize_text(entry.get("searchable_text", ""))
        detail_head_pose = normalize_text(entry.get("annotations", {}).get("detail_head_pose", ""))
        dir_text = f"{searchable_norm} {detail_head_pose}".strip()
        prompt_norm = normalize_text(prompt)
        wants_left = "left" in tokenize(prompt_norm)
        wants_right = "right" in tokenize(prompt_norm)
        opposite_direction_penalty = 0.0
        mixed_direction_penalty = 0.0
        direction_bonus = 0.0
        if wants_left and not wants_right:
            if contains_any_phrase(dir_text, LEFT_POSITIVE_PHRASES):
                direction_bonus += 3.0
            if contains_any_phrase(dir_text, RIGHT_POSITIVE_PHRASES):
                opposite_direction_penalty += 2.5
            if contains_any_phrase(dir_text, MIXED_DIRECTION_PHRASES):
                mixed_direction_penalty += 2.0
        elif wants_right and not wants_left:
            if contains_any_phrase(dir_text, RIGHT_POSITIVE_PHRASES):
                direction_bonus += 3.0
            if contains_any_phrase(dir_text, LEFT_POSITIVE_PHRASES):
                opposite_direction_penalty += 2.5
            if contains_any_phrase(dir_text, MIXED_DIRECTION_PHRASES):
                mixed_direction_penalty += 2.0

        motion_reward = 2.0 * head
        purity_reward = 1.0 * head_purity
        cross_penalty = 0.15 * expr + 0.10 * jaw
        jitter_penalty = 1.0 * head_vel
        total = (
            text_score
            + direction_bonus
            + motion_reward
            + purity_reward
            - cross_penalty
            - jitter_penalty
            - opposite_direction_penalty
            - mixed_direction_penalty
        )
    elif channel == "expr":
        searchable_norm = normalize_text(entry.get("searchable_text", ""))
        prompt_norm = normalize_text(prompt)
        prompt_tokens = set(tokenize(prompt_norm))

        smile_boost = 0.0
        if prompt_tokens.intersection(SMILE_TERMS) and contains_any_phrase(searchable_norm, SMILE_TERMS):
            smile_boost = 4.0

        unrelated_penalty = 0.0
        for neg in EXPR_NEGATIVE_TERMS:
            if neg in searchable_norm and neg not in prompt_norm:
                unrelated_penalty += 1.0

        motion_reward = 1.0 * min(expr, 5.0)
        purity_reward = 1.0 * min(expr_purity, 5.0)
        cross_penalty = 0.6 * head + 0.1 * jaw
        jitter_penalty = 0.1 * head_vel + unrelated_penalty
        total = (2.0 * text_score) + smile_boost + motion_reward + purity_reward - cross_penalty - jitter_penalty
    elif channel == "jaw":
        motion_reward = 2.0 * jaw
        purity_reward = 1.0 * jaw_purity
        cross_penalty = 0.3 * head
        jitter_penalty = 0.5 * head_vel
        total = text_score + motion_reward + purity_reward - cross_penalty - jitter_penalty
    else:
        motion_reward = 0.05 * intensity + 0.2 * head + 0.2 * expr + 0.2 * jaw
        purity_reward = cross_penalty = jitter_penalty = 0.0
        total = text_score + motion_reward

    channel_terms = {
        "text_score": float(text_score), "motion_reward": float(motion_reward), "purity_reward": float(purity_reward), "jitter_penalty": float(jitter_penalty), "cross_channel_penalty": float(cross_penalty)
    }
    if channel == "head":
        channel_terms["opposite_direction_penalty"] = float(opposite_direction_penalty)
        channel_terms["mixed_direction_penalty"] = float(mixed_direction_penalty)
        channel_terms["direction_bonus"] = float(direction_bonus)
    return {
        "total_score": float(total), "token_score": float(token_score), "phrase_score": float(phrase_score), "motion_score": float(motion_reward),
        "matched_terms": overlap,
        "channel_score_terms": channel_terms
    }


def retrieve(prompt: str, entries: List[dict], top_k: int, channel: str = "all") -> List[dict]:
    scored = []
    for e in entries:
        sc = score_entry(e, prompt, channel)
        row = {"sample_id": e.get("sample_id", ""), "motion_path": e.get("motion_path", ""), "searchable_text": e.get("searchable_text", ""), "annotations": e.get("annotations", {}), "motion_stats": e.get("motion_stats", {}), "channel": channel, "retrieval_scores": {"total_score": sc["total_score"], "token_score": sc["token_score"], "phrase_score": sc["phrase_score"], "motion_score": sc["motion_score"]}, "matched_terms": sc["matched_terms"], "channel_score_terms": sc["channel_score_terms"]}
        scored.append(row)
    scored.sort(key=lambda x: (x["retrieval_scores"]["total_score"], x.get("sample_id", "")), reverse=True)
    top = scored[:max(0, top_k)]
    for i, r in enumerate(top, start=1):
        r["rank"] = i
    return top


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_results(rows: List[dict], prompt: str, channel: str) -> None:
    print(f"[Retrieval] prompt={prompt!r}, channel={channel}, top_k={len(rows)}")
    for r in rows:
        print(f"  #{r['rank']:02d} score={r['retrieval_scores']['total_score']:.4f} sample_id={r['sample_id']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--codebook_jsonl", required=True, type=Path)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--channel", choices=["all", "head", "expr", "jaw"], default="all")
    ap.add_argument("--output_jsonl", type=Path, default=None)
    args = ap.parse_args()
    entries = load_jsonl(args.codebook_jsonl)
    rows = retrieve(args.prompt, entries, args.top_k, args.channel)
    print_results(rows, args.prompt, args.channel)
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, rows)


if __name__ == "__main__":
    main()
