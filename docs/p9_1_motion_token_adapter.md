# P9.1 MotionTokenAdapter

## Purpose

P9.1 adds model-side support for the P9.0 audit recommendation, Candidate B: project a compact motion/channel latent vector and add it to FastAvatar query / point embeddings before alternating cross-attention.

This stage does **not** change training losses, does **not** enable motion-zero training by default, and keeps normal inference behavior unchanged unless `model.use_motion_token=true`.

## Adapter and tensor shapes

`MotionTokenAdapter` lives in `FastAvatar/models/motion_token_adapter.py`.

Default shape contract:

- `motion_token_input`: `[B, 96]`
- 96 = `z_expr(64) + z_head(16) + z_jaw(16)`
- adapter output / `projected_motion_token`: `[B, transformer_dim]`
- FastAvatar query tokens before conditioning: `[B, N_frames, N_points, transformer_dim]`
- add-query conditioning:

```python
query_tokens = query_tokens + motion_token_scale * projected_motion_token[:, None, None, :]
```

## Config flags

Both train and inference configs include safe defaults:

```yaml
model:
  use_motion_token: false
  motion_token_input_dim: 96
  motion_token_hidden_dim: null
  motion_token_mode: "add_query"
  motion_token_scale: 1.0
  zero_flame_motion: false
  freeze_backbone_for_motion_token: false
```

When `use_motion_token=false`, no `motion_token_input` is required and the original query-token path is untouched.

When `use_motion_token=true`, `ModelFastAvatar.forward(...)` and `ModelFastAvatar.infer_images(...)` accept optional `motion_token_input`. If it is missing, the model uses zeros `[B, 96]` and prints a warning only when `FASTAVATAR_TOKEN_DEBUG=1`.

## zero_flame_motion plumbing

`zero_flame_motion=false` by default.

When enabled, P9.1 zeros the motion-like fields used in the query/motion-conditioning path:

- `expr`
- `jaw_pose`
- `neck_pose`
- `rotation`

It intentionally keeps `shape` / `betas`, cameras, source images, and `translation` unchanged. Translation is left unchanged because it can encode tracking/camera alignment in this codebase and should be revisited during the P9.2 motion-zero experiment.

## Freezing helper

When `freeze_backbone_for_motion_token=true`, `ModelFastAvatar.freeze_backbone_keep_motion_token_trainable()` freezes all existing model parameters and keeps only `motion_token_adapter` trainable. P9.1 only adds the helper; it does not change training scripts.

## Debugging

Set:

```bash
FASTAVATAR_TOKEN_DEBUG=1
```

The model prints:

- `use_motion_token`
- `zero_flame_motion`
- `motion_token_input` shape
- projected token shape
- query token shape before / after conditioning
- `motion_token_scale`
- zeroed FLAME fields if `zero_flame_motion=true`

## Smoke test

Module-level shape test:

```bash
python scripts/debug/debug_motion_token_forward.py
```

Syntax checks:

```bash
python -m py_compile FastAvatar/models/modeling_FastAvatar.py FastAvatar/models/alternating_cross_attn.py scripts/debug/debug_motion_token_forward.py
```

Full inference shape tracing can be run by adding `FASTAVATAR_TOKEN_DEBUG=1` to the normal inference command after enabling the config flags.

## Limitations and next step

P9.1 only supports Candidate B (`motion_token_mode="add_query"`). It does not implement concatenated transformer tokens, FiLM/AdaLN modulation, or context-memory token injection.

Next step: P9.2 should implement a small motion-zero + token overfit experiment using frozen or partially frozen FastAvatar weights and a known motion token source.
