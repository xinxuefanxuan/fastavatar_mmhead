# P9.0 FastAvatar token-injection audit

## Scope

This audit inspects the current FastAvatar data/model path to identify safe insertion points for a future motion/text token. No training behavior, model outputs, losses, or config defaults are changed in this step. The only code change is optional shape instrumentation gated by `FASTAVATAR_TOKEN_DEBUG=1`.

## Current FastAvatar data/model flow

### 1. FLAME parameter loading

**Training / dataset path**

- `FastAvatar/datasets/nersemble.py::NersembleDataset.inner_get_item` loads per-frame FLAME `.npz` files from `flame_param/{frame_idx:05d}.npz` and stacks keys into `all_flame_params`.
- The returned sample exposes both input and target FLAME tensors as `input_{key}` and `target_{key}`.
- Relevant shapes documented in the dataset class are:
  - `rotation`: `[N, 3]`
  - `neck_pose`: `[N, 3]`
  - `jaw_pose`: `[N, 3]`
  - `eyes_pose`: `[N, 6]`
  - `translation`: `[N, 3]`
  - `shape`: `[N, 300]`
  - `expr`: `[N, 100]`

**Inference / motion-sequence path**

- `FastAvatar/runners/infer/utils.py::load_flame_params` reads one frame `.npz` and returns tensors for `expr`, `rotation`, `neck_pose`, `jaw_pose`, `eyes_pose`, `translation`, and optional `teeth_bs`.
- `FastAvatar/runners/infer/utils.py::prepare_motion_seqs` reads `transforms.json`, loads all frame FLAME params, stacks them to `[N, ...]`, attaches `betas`, then unsqueezes to `[1, N, ...]`.
- `FastAvatar/runners/infer/fastavatar.py::FastAvatarInfer.run` calls `prepare_motion_seqs`, obtains `inf_flame_params = motion_seqs["flame_params"]`, moves them to device, and passes them to `self.model.infer_images(...)`.

### 2. Where expr / neck_pose / jaw_pose / rotation / translation are used

- `FastAvatar/models/modeling_FastAvatar.py::ModelFastAvatar.forward` uses `input_flame_params` only to create query points for latent Gaussian features, and uses `inf_flame_params` later for target-frame rendering.
- `FastAvatar/models/modeling_FastAvatar.py::ModelFastAvatar.infer_images` mirrors this inference path: `input_flame_params` drives query-point creation, while `inf_flame_params` drives animation/rendering.
- `FastAvatar/models/rendering/gs_renderer.py::GS3DRenderer.animate_gs_model` consumes the target FLAME fields:
  - `expr` enters blend-shape expression deformation,
  - `rotation`, `neck_pose`, `jaw_pose`, and `eyes_pose` are concatenated into FLAME pose in the FLAME model,
  - `translation` is applied to animated vertices/joints.

### 3. Motion condition encoding

There is no separate learned motion/text-condition encoder in the current FastAvatar internal model path. Motion enters through FLAME tensors:

1. **Input identity/query path:** `input_flame_params` -> `renderer.get_query_points(...)` -> canonical FLAME vertices/query points.
2. **Target animation path:** `inf_flame_params` -> `_render_multiple_frames(...)` -> `renderer(...)` -> `animate_gs_model(...)` -> FLAME animation.

A future token should therefore be introduced as an additional conditioning signal; it should not replace the existing FLAME render path unless the training objective is changed later.

### 4. Point/query token creation

- `ModelFastAvatar.forward_latent_points` calls `renderer.get_query_points(flame_for_query, device=image.device)`, which currently uses canonical shape/betas to get canonical FLAME vertices.
- `ModelFastAvatar.forward_transformer` embeds these query points through `self.pcl_embed(...)`, producing point/query tokens with shape `[B, N_input, N_points, C]`.
- If FramePack is active, a copied point-token frame is split as `compressed_x` for the compressed context branch.

### 5. Transformer inputs

- `ModelFastAvatar.forward_encode_image` encodes source images with DINOv2 into context/image tokens shaped `[B, N_input, H*W, C]`.
- Optional FramePack compression produces `compressed_cond` shaped `[B, 1, compressed_tokens, C]`.
- `ModelFastAvatar.forward_transformer` calls `AlternatingCrossAttn.forward(x, image_feats, compressed_x=..., compressed_cond=...)`.
- `AlternatingCrossAttn.forward` alternates:
  - frame attention over point tokens `x` and per-frame image context `cond`, and
  - global attention over flattened context/image tokens.

### 6. Context attention and frame attention inputs

- `AlternatingCrossAttn.forward_frame_attn` reshapes:
  - point/query tokens: `[B, N_frame, N_points, C] -> [B*N_frame, N_points, C]`
  - context/image tokens: `[B, N_frame, HW, C] -> [B*N_frame, HW, C]`
- `AlternatingCrossAttn.forward_global_attn` operates on flattened context tokens:
  - base-only: `[B, N_input * HW, C]`
  - FramePack: base + compressed context concatenated along token dimension.

## Candidate token injection points

| Candidate | File / class / function | Tensor shape | Pros | Cons | Expected code change size | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| A. Concatenate token to transformer point/query sequence | `FastAvatar/models/modeling_FastAvatar.py::ModelFastAvatar.forward_transformer` before `self.transformer(...)`; possibly `FastAvatar/models/alternating_cross_attn.py::AlternatingCrossAttn.forward_frame_attn` if token needs special handling | Existing `x`: `[B, N_input, N_points, C]`; token option: `[B, N_input, 1, C]` appended on point dimension | Most direct “token” formulation; token can attend with image context during frame attention; localized near query-token construction | Renderer expects latent points to match query points; appended token must be removed before rendering or paired with dummy query point; changes point count invariants | Medium (projection, concat, mask/drop before renderer) | Medium-high |
| B. Add token as conditioning vector to query embedding | `FastAvatar/models/modeling_FastAvatar.py::ModelFastAvatar.forward_transformer`, immediately after `x = self.pcl_embed(...)` | Existing `x`: `[B, N_input, N_points, C]`; projected condition can be `[B, N_input, 1, C]` broadcast/add to points | Smallest shape disruption; renderer point count unchanged; easy to gate/ablate; safest for first Motion-zero experiment | Less interpretable as an explicit token; global context sees only modified queries, not a standalone memory token | Small (MLP projection + addition) | Low |
| C. FiLM/AdaLN-style modulation | `FastAvatar/models/alternating_cross_attn.py::FrameAttn` / `AlternatingCrossAttn` blocks, or lower-level `SD3JointTransformerBlock` wrapper | Condition vector `[B, C]` or `[B, N_frame, C]` projected to scale/shift for hidden states | Strong conditioning control; does not change sequence length; can modulate both query and context streams | Requires modifying transformer block internals or wrapping layer norms; larger train/inference compatibility surface | Medium-large | Medium |
| D. Append token to frame/context attention memory | `FastAvatar/models/alternating_cross_attn.py::AlternatingCrossAttn.forward`, after context projection; append to `cond` as extra context token per frame or global memory | Base `cond`: `[B, N_input, HW, C]`; token option: `[B, N_input, 1, C]`; global flattened becomes `[B, N_input*(HW+1), C]` | Token participates as memory/context; renderer point count unchanged; suitable for text/motion summaries | Positional encoding currently assumes square `HW`; appending tokens requires careful positional handling and total-token bookkeeping | Medium | Medium |

## Recommendation

The safest first insertion point is **Candidate B: add a projected motion/text condition vector to query embeddings after point embedding and before alternating attention**.

Reasons:

1. It does not change the number of Gaussian points, so `latent_points.reshape(...)`, `query_points.reshape(...)`, and renderer point-count assumptions remain intact.
2. It is localized in `ModelFastAvatar.forward_transformer`, where query tokens already exist and are still pre-transformer.
3. It can be initialized as a zero-output projection, preserving current behavior at initialization.
4. It can later be upgraded into Candidate A or D once token training losses and masking rules are validated.

Candidate A is the most literal token injection strategy, but it requires explicitly removing the extra token before rendering or defining a dummy query point that never reaches `GS3DRenderer.forward_gs_attr`. That makes it riskier for the first Motion-zero/token-training experiment.

## Debug instrumentation added

Set:

```bash
FASTAVATAR_TOKEN_DEBUG=1
```

The model now prints shape summaries for:

- motion/FLAME condition dictionaries (`input_flame_params`, `inf_flame_params`),
- source image encoder inputs and context features,
- transformer query/point token tensors,
- compressed FramePack context tensors when present,
- renderer input/output tensors,
- final model forward/inference outputs.

The instrumentation is intentionally print-only and env-gated. It does not alter tensors, losses, or outputs.

## Suggested next step for P9.1

Implement a disabled-by-default `motion_token` path around `ModelFastAvatar.forward_transformer`:

1. Add optional `motion_token` argument with expected shape `[B, N_input, D_token]` or `[B, D_token]`.
2. Project it to transformer dim with a zero-initialized linear layer.
3. Add it to `x` after `pcl_embed` using broadcast over points.
4. Keep a config flag defaulting to disabled.
5. Add unit/smoke shape tests with `FASTAVATAR_TOKEN_DEBUG=1` before training.
