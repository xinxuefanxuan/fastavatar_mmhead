# P9.2 Tiny Smoke OOM Notes

## Why this exists

The full FastAvatar training configuration can exceed the memory available on a 24GB RTX 4090, especially when the P9.2 motion-zero + GT-token path is enabled with many input frames, target frames, high render resolution, perceptual/identity losses, and validation image saving.

The P9.2 tiny smoke configs are **not intended for final quality**. They are only meant to verify that:

1. the GT motion token is built from the original FLAME motion,
2. explicit FLAME motion can be zeroed,
3. the MotionTokenAdapter receives gradients,
4. a short forward/backward training loop can run without CUDA OOM.

## Tiny config

Use:

```bash
bash scripts/debug/run_motion_zero_token_overfit.sh
```

This defaults to:

```text
configs/train/fastavatar_motion_zero_token_overfit_tiny.yaml
```

The tiny config reduces memory by using:

- `dataset.input_frames: 32`
- `dataset.target_frames: 2`
- `dataset.render_image_res: 256`
- `dataset.source_image_res: 256`
- `model.source_image_res: 256`
- `model.rendering_chunk_size_train: 2`
- disabled perceptual, SSIM, identity, offset, tracking, and pruning losses
- disabled validation image saving and effectively disabled periodic checkpoints

It also sets:

```yaml
model.motion_token_train_adapter_only_strict: true
```

so smoke training fails if any non-`motion_token_adapter` parameter remains trainable when `freeze_backbone_for_motion_token: true`.

## Ultra-tiny fallback

If the tiny config still OOMs, use:

```bash
bash scripts/debug/run_motion_zero_token_overfit_ultra_tiny.sh
```

This uses:

```text
configs/train/fastavatar_motion_zero_token_overfit_ultra_tiny.yaml
```

and further reduces memory with:

- `dataset.input_frames: 8`
- `dataset.target_frames: 1`
- `dataset.render_image_res: 128`
- `dataset.source_image_res: 128`
- `model.source_image_res: 128`
- `model.rendering_chunk_size_train: 1`

This mode is only for checking whether forward/backward and token gradients work.

## Metadata generation

The runner regenerates project-local metadata before training:

```bash
python scripts/debug/create_p9_2_overfit_metadata.py \
  --config "${CONFIG_PATH}" \
  --min_pairs auto \
  --max_ids 6 \
  --max_items_per_id 4 \
  --prefer_ids 036 \
  --output datasets/p9_2_overfit_mixed_uids.json
```

`--min_pairs auto` uses the actual NersembleDataset requirement:

```text
required_pairs = dataset.input_frames + dataset.target_frames
```

If no frame groups satisfy this requirement, the metadata helper fails without writing output.

## If tiny succeeds

After the tiny smoke passes, scale up gradually in this order:

1. increase `target_frames`,
2. increase `input_frames`,
3. increase `render_image_res` / `source_image_res`,
4. re-enable losses one at a time,
5. re-enable validation image saving/checkpoint cadence.

Avoid changing multiple memory-heavy knobs at once.
