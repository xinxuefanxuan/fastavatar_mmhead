# P9.2 A100 local path setup

P9.2 smoke configs are now generic base configs. Do not edit them for a specific
machine. Instead, create a gitignored local override file or pass environment
variables to the runner.

## 1. Create local override file on the A100 machine

From the repo root (`/home/sjd/FastAvatar_mmhead` on the A100 machine):

```bash
mkdir -p configs/local
cat > configs/local/p9_2_local_paths.yaml <<'EOF'
nersemble_root_dir: "/ssd_data/sjd/data/nersemble_fastavatar_unified_full"
source_mixed_uids: "/home/sjd/FastAvatar_mmhead/datasets/mixed_uids.json"
generated_meta_path: "./datasets/p9_2_overfit_mixed_uids.json"
preferred_val_id: null
EOF
```

`configs/local/p9_2_local_paths.yaml` is ignored by git. The tracked template is
`configs/local/p9_2_local_paths.example.yaml`.

## 2. Run latent-only micro smoke

```bash
CONFIG_PATH=configs/train/fastavatar_motion_zero_token_overfit_micro.yaml \
CUDA_VISIBLE_DEVICES=0 \
FASTAVATAR_TOKEN_DEBUG=1 \
FASTAVATAR_DATASET_FAIL_FAST=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/debug/run_motion_zero_token_overfit.sh
```

The runner does **not** mutate the source config. It writes a resolved runtime
config under:

```text
outputs/mmhead_debug/p9_2_runtime_configs/<base_config_name>_resolved.yaml
```

That resolved config contains:

- `dataset.meta_path` set to the generated P9.2 metadata.
- `dataset.datasets.nersemble.root_dir` set to the local processed Nersemble root.
- `dataset.datasets.nersemble.val_id` set to a valid selected ID.

## 3. Override with environment variables

Environment variables have the highest priority:

```bash
P9_NERSEMBLE_ROOT=/ssd_data/sjd/data/nersemble_fastavatar_unified_full \
P9_SOURCE_META=/home/sjd/FastAvatar_mmhead/datasets/mixed_uids.json \
P9_GENERATED_META=./datasets/p9_2_overfit_mixed_uids.json \
P9_VAL_ID=083 \
bash scripts/debug/run_motion_zero_token_overfit.sh
```

Priority order is:

1. Environment variables (`P9_NERSEMBLE_ROOT`, `P9_SOURCE_META`,
   `P9_GENERATED_META`, `P9_VAL_ID`).
2. `configs/local/p9_2_local_paths.yaml`.
3. Generic base config defaults.

If `P9_VAL_ID` or `preferred_val_id` is invalid for the generated metadata, the
runner prints a warning and chooses a deterministic valid ID from the selected
IDs, preferring the last sorted ID. It never silently launches with an invalid
`val_id` or a zero-sized train split.

## 4. Renderer smoke

The default micro config is latent-only:

```yaml
model.debug_skip_renderer: true
model.debug_latent_smoke_loss: true
model.debug_max_query_points: null
```

For A100 full-render smoke, use:

```bash
CONFIG_PATH=configs/train/fastavatar_motion_zero_token_overfit_micro_render.yaml \
bash scripts/debug/run_motion_zero_token_overfit.sh
```

The renderer config keeps `debug_max_query_points: null` because flattened point
subsampling breaks FastAvatar renderer point-frame assumptions unless renderer
metadata is updated consistently.
