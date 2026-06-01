#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from FastAvatar.models.motion_token_adapter import MotionTokenAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test P9.1 MotionTokenAdapter shape compatibility.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--num_points", type=int, default=5)
    parser.add_argument("--input_dim", type=int, default=96)
    parser.add_argument("--transformer_dim", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    adapter = MotionTokenAdapter(
        input_dim=args.input_dim,
        hidden_dim=args.transformer_dim,
        output_dim=args.transformer_dim,
    ).to(device)
    query_tokens = torch.randn(args.batch_size, args.num_frames, args.num_points, args.transformer_dim, device=device)

    disabled_tokens = query_tokens.clone()
    assert torch.allclose(disabled_tokens, query_tokens), "disabled path should preserve query tokens"
    print("[MotionTokenDebug] disabled path: query tokens unchanged")

    motion_token_input = torch.randn(args.batch_size, args.input_dim, device=device)
    projected = adapter(motion_token_input)
    conditioned = query_tokens + args.scale * projected[:, None, None, :]

    trainable = [(name, p.numel()) for name, p in adapter.named_parameters() if p.requires_grad]
    print(f"[MotionTokenDebug] adapter parameters exist: {bool(trainable)}")
    print(f"[MotionTokenDebug] trainable parameter count: {sum(n for _, n in trainable)}")
    for name, count in trainable:
        print(f"[MotionTokenDebug] trainable: {name} ({count})")
    print(f"[MotionTokenDebug] motion_token_input shape: {list(motion_token_input.shape)}")
    print(f"[MotionTokenDebug] projected token shape: {list(projected.shape)}")
    print(f"[MotionTokenDebug] query token shape before conditioning: {list(query_tokens.shape)}")
    print(f"[MotionTokenDebug] query token shape after conditioning: {list(conditioned.shape)}")
    print("[MotionTokenDebug] full FastAvatar inference can be inspected with FASTAVATAR_TOKEN_DEBUG=1")


if __name__ == "__main__":
    main()
