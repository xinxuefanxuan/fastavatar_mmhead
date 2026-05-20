#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Dict

HEAD_TERMS = ["head", "turn", "look", "left", "right", "up", "down", "nod", "shake"]
EXPR_TERMS = ["smile", "happy", "sad", "angry", "surprise", "surprised", "laugh", "blink", "expression", "face"]
JAW_TERMS = ["mouth", "open mouth", "jaw", "speak", "talk"]


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract(norm: str, terms: list[str]) -> str:
    hits = [t for t in terms if t in norm]
    return " ".join(hits)


def parse_channel_intents(prompt: str) -> Dict[str, object]:
    norm = _norm(prompt)
    head_prompt = _extract(norm, HEAD_TERMS)
    expr_prompt = _extract(norm, EXPR_TERMS)
    jaw_prompt = _extract(norm, JAW_TERMS)
    return {
        "head_prompt": head_prompt,
        "expr_prompt": expr_prompt,
        "jaw_prompt": jaw_prompt,
        "need_head": bool(head_prompt),
        "need_expr": bool(expr_prompt),
        "need_jaw": bool(jaw_prompt),
    }
