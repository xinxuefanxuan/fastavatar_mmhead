# P9.2 Motion-zero + GT Motion Token Overfit

## Goal

P9.2 is a minimal opt-in experiment to test whether FastAvatar can use an injected motion token after explicit FLAME motion is removed.

The comparison target is:

1. **A: normal motion, no token** — baseline FastAvatar behavior.
2. **B: zero motion, no token** — explicit FLAME motion removed, expected to lose expression/head/jaw motion.
3. **C: zero motion + GT token** — explicit FLAME motion removed, but a GT motion token built from the original FLAME motion is injected through the P9.1 `MotionTokenAdapter`.

Default FastAvatar behavior remains unchanged because all P9.2 options are disabled in the standard configs.

## Token construction

`motion_token_source: "frame_flame_gt"` builds `motion_token_input` before any zeroing:

1. Read target-frame FLAME tensors from `inf_flame_params` / target FLAME batch.
2. Extract:
   - `expr`: first `motion_token_expr_dim` dims, padded/truncated to 50 dims.
   - `neck_pose`: 3 dims.
   - `jaw_pose`: 3 dims.
3. Concatenate to `motion56 = [expr50, neck_pose3, jaw_pose3]`.
4. If `motion_token_norm_stats` is configured, normalize the 56-D vector using the dataset `mean/std` layout:
   - `0:50` expression,
   - `50:53` head/neck,
   - `53:56` jaw.
5. Pad to `motion_token_pad_to_dim=96` with zeros.

The first implementation frame-averages target-frame tensors to produce one token per batch item: `[B, 96]`.

## Zeroing order

When `use_motion_token=true`, `motion_token_source="frame_flame_gt"`, and `zero_flame_motion=true`:

1. Build `motion_token_input` from the original target FLAME tensors.
2. Zero explicit motion fields.
3. Run the query-conditioning path with the GT token.
4. Render with zeroed target FLAME motion.

Zeroed fields:

- `expr`
- `jaw_pose`
- `neck_pose`
- `rotation`

Fields intentionally kept unchanged:

- `shape` / `betas`
- source images / image encoder features
- camera intrinsics / extrinsics
- `translation`

`translation` is kept because it can contain tracking/camera alignment and should be ablated separately only after the initial overfit test is stable.

## Config flags

Standard configs keep these disabled/defaulted:

```yaml
model:
  use_motion_token: false
  motion_token_source: "none"
  motion_token_norm_stats: null
  motion_token_expr_dim: 50
  motion_token_pad_to_dim: 96
  zero_flame_motion: false
```

The overfit config enables:

```yaml
model:
  use_motion_token: true
  motion_token_mode: "add_query"
  motion_token_source: "frame_flame_gt"
  motion_token_norm_stats: "outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json"
  zero_flame_motion: true
  freeze_backbone_for_motion_token: true
```

## Debug batch

Run:

```bash
python scripts/debug/debug_motion_zero_token_batch.py
```

It prints:

- `motion_token_input` shape,
- token mean/std/norm,
- finite / NaN / Inf check,
- `expr` / `neck_pose` / `jaw_pose` norms before zeroing,
- same norms after zeroing,
- trainable `MotionTokenAdapter` parameter names/counts,
- projected token shape.

The debug script uses a synthetic batch by default so it can validate construction logic without requiring local datasets.

## Overfit run

Run:

```bash
bash scripts/debug/run_motion_zero_token_overfit.sh
```

The script launches the project training runner with:

```bash
python FastAvatar/launch.py train.fastavatar --config configs/train/fastavatar_motion_zero_token_overfit.yaml
```

You can append OmegaConf-style overrides after the script command, for example:

```bash
bash scripts/debug/run_motion_zero_token_overfit.sh train.debug_global_steps=5
```

## Expected results

For a successful P9.2 overfit:

- A should render normal motion.
- B should suppress expression / neck / jaw motion.
- C should recover motion better than B using only the injected token path.

## Failure diagnosis

If C does not improve over B:

1. Confirm `FASTAVATAR_TOKEN_DEBUG=1` shows a nonzero `motion_token_input` and projected token.
2. Confirm `zero_flame_motion=True` zeroes `expr`, `jaw_pose`, `neck_pose`, and `rotation`.
3. Check whether `freeze_backbone_for_motion_token=true` leaves only `motion_token_adapter` trainable.
4. Try disabling `freeze_backbone_for_motion_token` for a tiny overfit to test whether the adapter alone is too weak.
5. Verify `motion_token_norm_stats` matches the 56-D motion layout used by the P7/P8 channel motion pipeline.
