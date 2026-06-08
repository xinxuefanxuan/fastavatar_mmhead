#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="p9_2j_expanded_2k_counterfactual"
CONFIG_PATH="${CONFIG_PATH:-configs/train/fastavatar_motion_zero_token_adapter_expanded_2k_counterfactual.yaml}"
LOCAL_PATHS_CONFIG="${P9_LOCAL_PATHS_CONFIG:-configs/local/p9_2_local_paths.yaml}"
TRAIN_IDS="${P9_TRAIN_IDS:-017,018,024,030,031,032,033,035,036,037,038,040,042,043,055,056,057,059,061,063}"
HOLDOUT_IDS="${P9_HOLDOUT_IDS:-083,089,090}"
SACRIFICIAL_VAL_ID="${P9_SACRIFICIAL_VAL_ID:-017}"
TRAIN_VAL_ID="${P9_TRAIN_VAL_ID:-083}"
MAX_ITEMS_PER_ID="${P9_MAX_ITEMS_PER_ID:-8}"
DEBUG_STEPS="${P9_DEBUG_STEPS:-2000}"
METADATA_DIR="${P9_2J_METADATA_DIR:-outputs/mmhead_debug/p9_2j_counterfactual_metadata}"
METADATA_PATH="${P9_2J_METADATA_PATH:-${METADATA_DIR}/${EXPERIMENT_NAME}_mixed_uids.json}"
RUNTIME_CONFIG_DIR="${P9_2J_RUNTIME_CONFIG_DIR:-outputs/mmhead_debug/p9_2j_counterfactual_runtime_configs}"
RUNTIME_CONFIG_PATH="${RUNTIME_CONFIG_DIR}/fastavatar_motion_zero_token_adapter_expanded_2k_counterfactual_resolved.yaml"
EVAL_ROOT="${P9_2J_EVAL_ROOT:-outputs/mmhead_debug/p9_2j_counterfactual_eval}"
SENS_ROOT="${P9_2J_SENS_ROOT:-outputs/mmhead_debug/p9_2j_counterfactual_sensitivity}"
DIFF_ROOT="${P9_2J_DIFF_ROOT:-outputs/mmhead_debug/p9_2j_counterfactual_checkpoint_diff}"
CHECKPOINT_PATH="${P9_2J_CKPT:-exps/checkpoints/fastavatar/fastavatar_motion_zero_token_adapter_expanded_2k_counterfactual/002000/model.safetensors}"
PRETRAINED_CKPT="${P9_PRETRAINED_CKPT:-model_zoo/fastavatar/model.safetensors}"

export FASTAVATAR_TOKEN_DEBUG="${FASTAVATAR_TOKEN_DEBUG:-0}"
export FASTAVATAR_DATASET_FAIL_FAST="${FASTAVATAR_DATASET_FAIL_FAST:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${METADATA_DIR}" "${RUNTIME_CONFIG_DIR}" "${EVAL_ROOT}" "${SENS_ROOT}" "${DIFF_ROOT}"

# Resolve local root/source overrides without requiring PyYAML/OmegaConf in this wrapper.
eval "$(python - "${CONFIG_PATH}" "${LOCAL_PATHS_CONFIG}" "${METADATA_PATH}" <<'PY'
from pathlib import Path
import os
import shlex
import sys

repo = Path.cwd()
config_path = Path(sys.argv[1])
local_path = Path(sys.argv[2])
metadata_path = Path(sys.argv[3])

def resolve(value):
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo / p).resolve()

def simple_yaml_value(path, key):
    if not path.exists():
        return None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or ':' not in line:
            continue
        k, v = line.split(':', 1)
        if k.strip() == key:
            value = v.strip().split('#', 1)[0].strip()
            if value in ('', 'null', 'None'):
                return None
            return value.strip('"\'')
    return None

def config_root(path):
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if stripped.startswith('root_dir:') and not stripped.startswith('#'):
            return stripped.split(':', 1)[1].strip().split('#', 1)[0].strip().strip('"\'')
    return 'data/nersemble_fastavatar_unified_full'

root = os.environ.get('P9_NERSEMBLE_ROOT') or simple_yaml_value(local_path, 'nersemble_root_dir') or config_root(config_path)
source = os.environ.get('P9_SOURCE_META') or simple_yaml_value(local_path, 'source_mixed_uids') or 'datasets/mixed_uids.json'
print(f"P9_RESOLVED_ROOT={shlex.quote(str(resolve(root)))}")
print(f"P9_RESOLVED_SOURCE={shlex.quote(str(resolve(source)))}")
print(f"P9_RESOLVED_METADATA={shlex.quote(str(resolve(metadata_path)))}")
PY
)"

PREFER_IDS="${TRAIN_IDS}"
MAX_IDS=$(python - "${PREFER_IDS}" <<'PY'
import sys
ids=[x for x in sys.argv[1].split(',') if x]
print(len(dict.fromkeys(ids)))
PY
)

echo "[P9.2J] Generate expanded metadata: ${P9_RESOLVED_METADATA}"
echo "[P9.2J] train_val_id=${TRAIN_VAL_ID} holdout_sacrificial_val_id=${SACRIFICIAL_VAL_ID}"
python scripts/debug/create_p9_2_overfit_metadata.py \
  --config "${CONFIG_PATH}" \
  --src_meta "${P9_RESOLVED_SOURCE}" \
  --root_dir "${P9_RESOLVED_ROOT}" \
  --output "${P9_RESOLVED_METADATA}" \
  --min_pairs auto \
  --max_ids "${MAX_IDS}" \
  --max_items_per_id "${MAX_ITEMS_PER_ID}" \
  --prefer_ids "${PREFER_IDS}"

# Write a resolved runtime config with local paths and conservative epochs so debug_global_steps is reached.
eval "$(python - "${CONFIG_PATH}" "${RUNTIME_CONFIG_PATH}" "${P9_RESOLVED_ROOT}" "${P9_RESOLVED_METADATA}" "${TRAIN_VAL_ID}" "${DEBUG_STEPS}" <<'PY'
from pathlib import Path
import json
import math
import shlex
import sys

config_path, runtime_path, root_dir, meta_path, val_id, debug_steps = sys.argv[1:]
config_path = Path(config_path)
runtime_path = Path(runtime_path)
meta_path_obj = Path(meta_path)
debug_steps = int(debug_steps)
meta = json.loads(meta_path_obj.read_text())

def uid_from_key(key):
    if key.startswith('nersemble/'):
        key = key[len('nersemble/'):]
    return key.split('/', 1)[0]

train_count = sum(1 for key in meta if uid_from_key(key) != val_id)
steps_per_epoch = max(train_count, 1)
calculated_epochs = max(1, math.ceil(debug_steps / steps_per_epoch) + 1)
runtime_epochs = max(calculated_epochs, 1000)

out=[]
for raw in config_path.read_text().splitlines():
    stripped = raw.strip()
    indent = raw[:len(raw)-len(raw.lstrip())]
    if stripped.startswith('child:') and not stripped.startswith('#'):
        raw = f'{indent}child: fastavatar_motion_zero_token_adapter_expanded_2k_counterfactual'
    elif stripped.startswith('root_dir:') and not stripped.startswith('#'):
        raw = f'{indent}root_dir: "{root_dir}"'
    elif stripped.startswith('meta_path:') and not stripped.startswith('#'):
        raw = f'{indent}meta_path: "{meta_path}"'
    elif stripped.startswith('val_id:') and not stripped.startswith('#'):
        raw = f'{indent}val_id: ["{val_id}"] # Sacrificial validation ID excluded from adapter training'
    elif stripped.startswith('epochs:') and not stripped.startswith('#'):
        raw = f'{indent}epochs: {runtime_epochs}'
    elif stripped.startswith('debug_global_steps:') and not stripped.startswith('#'):
        raw = f'{indent}debug_global_steps: {debug_steps}'
    out.append(raw)
runtime_path.parent.mkdir(parents=True, exist_ok=True)
runtime_path.write_text('\n'.join(out) + '\n')
print(f'P9_TRAIN_COUNT={train_count}')
print(f'P9_STEPS_PER_EPOCH={steps_per_epoch}')
print(f'P9_CALCULATED_EPOCHS={calculated_epochs}')
print(f'P9_RUNTIME_EPOCHS={runtime_epochs}')
print(f'P9_RESOLVED_RUNTIME_CONFIG={shlex.quote(str(runtime_path.resolve()))}')
PY
)"

echo "[P9.2J] train_count=${P9_TRAIN_COUNT} steps_per_epoch=${P9_STEPS_PER_EPOCH} calculated_epochs=${P9_CALCULATED_EPOCHS} runtime_epochs=${P9_RUNTIME_EPOCHS}"
echo "[P9.2J] runtime_config=${P9_RESOLVED_RUNTIME_CONFIG}"

python FastAvatar/launch.py train.fastavatar --config "${P9_RESOLVED_RUNTIME_CONFIG}"

echo "[P9.2J] Run checkpoint diff"
python scripts/debug/inspect_p9_2_checkpoint_diff.py \
  --pretrained_checkpoint "${PRETRAINED_CKPT}" \
  --trained_checkpoint "${CHECKPOINT_PATH}" \
  --output_dir "${DIFF_ROOT}"

IFS=',' read -r -a HOLDOUT_ARRAY <<< "${HOLDOUT_IDS}"
for HOLDOUT_ID in "${HOLDOUT_ARRAY[@]}"; do
  echo "[P9.2J] Eval holdout ${HOLDOUT_ID}"
  P9_NERSEMBLE_ROOT="${P9_RESOLVED_ROOT}" \
  P9_SOURCE_META="${P9_RESOLVED_SOURCE}" \
  P9_GENERATED_META="${EVAL_ROOT}/holdout${HOLDOUT_ID}/runtime_configs/generated_mixed_uids.json" \
  python scripts/debug/eval_p9_2_abc_same_batch.py \
    --base_config "${CONFIG_PATH}" \
    --adapter_ckpt "${CHECKPOINT_PATH}" \
    --output_dir "${EVAL_ROOT}/holdout${HOLDOUT_ID}" \
    --split holdout \
    --train_ids "${TRAIN_IDS}" \
    --holdout_ids "${HOLDOUT_ID}" \
    --sacrificial_val_id "${SACRIFICIAL_VAL_ID}" \
    --metadata_max_items_per_id "${MAX_ITEMS_PER_ID}" \
    --num_batches "${P9_EVAL_NUM_BATCHES:-4}" \
    --max_save_frames "${P9_MAX_SAVE_FRAMES:-1}"
done

echo "[P9.2J] Adapter sensitivity on train split"
P9_NERSEMBLE_ROOT="${P9_RESOLVED_ROOT}" \
P9_SOURCE_META="${P9_RESOLVED_SOURCE}" \
P9_GENERATED_META="${SENS_ROOT}/train/runtime_configs/generated_mixed_uids.json" \
python scripts/debug/inspect_p9_2_adapter_token_sensitivity.py \
  --base_config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output_dir "${SENS_ROOT}/train" \
  --split train \
  --train_ids "${TRAIN_IDS}" \
  --metadata_prefer_ids "${PREFER_IDS}" \
  --metadata_max_ids "${MAX_IDS}" \
  --metadata_max_items_per_id "${MAX_ITEMS_PER_ID}" \
  --num_batches "${P9_SENS_NUM_BATCHES:-8}"

for HOLDOUT_ID in "${HOLDOUT_ARRAY[@]}"; do
  echo "[P9.2J] Adapter sensitivity holdout ${HOLDOUT_ID}"
  P9_NERSEMBLE_ROOT="${P9_RESOLVED_ROOT}" \
  P9_SOURCE_META="${P9_RESOLVED_SOURCE}" \
  P9_GENERATED_META="${SENS_ROOT}/holdout${HOLDOUT_ID}/runtime_configs/generated_mixed_uids.json" \
  python scripts/debug/inspect_p9_2_adapter_token_sensitivity.py \
    --base_config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output_dir "${SENS_ROOT}/holdout${HOLDOUT_ID}" \
    --split holdout \
    --train_ids "${TRAIN_IDS}" \
    --holdout_ids "${HOLDOUT_ID}" \
    --sacrificial_val_id "${SACRIFICIAL_VAL_ID}" \
    --metadata_max_items_per_id "${MAX_ITEMS_PER_ID}" \
    --num_batches "${P9_SENS_NUM_BATCHES:-4}"
done

echo "[P9.2J] Write summary report"
python - "${EVAL_ROOT}" "${SENS_ROOT}" "${DIFF_ROOT}" "${TRAIN_IDS}" "${HOLDOUT_IDS}" "${MAX_ITEMS_PER_ID}" "${P9_TRAIN_COUNT}" <<'PY'
from pathlib import Path
import json
import sys

eval_root, sens_root, diff_root = map(Path, sys.argv[1:4])
train_ids, holdout_ids, max_items, train_count = sys.argv[4:8]
holdouts = [x for x in holdout_ids.split(',') if x]
diff_summary_path = diff_root / 'checkpoint_diff_summary.json'
diff = json.loads(diff_summary_path.read_text()) if diff_summary_path.exists() else {}
lines = [
    '# P9.2j Counterfactual 2k Summary',
    '',
    f'* train IDs: `{train_ids}`',
    f'* holdout IDs: `{holdout_ids}`',
    f'* max_items_per_id: `{max_items}`',
    f'* train sample count: `{train_count}`',
    f'* checkpoint diff pass: `{diff.get("pass")}`',
    f'* changed non-motion_token_adapter parameters: `{diff.get("changed_non_motion_token_adapter")}`',
    '',
    '## Per-Holdout A/B/C/D/E/F Metrics',
]
for holdout in holdouts:
    metrics_path = eval_root / f'holdout{holdout}' / 'metrics.json'
    lines += ['', f'### Holdout {holdout}']
    if not metrics_path.exists():
        lines.append(f'* Missing metrics: `{metrics_path}`')
        continue
    metrics = json.loads(metrics_path.read_text())
    variants = metrics.get('aggregate', {}).get('variants', {})
    improvement = metrics.get('aggregate', {}).get('improvement', {})
    lines += [
        '| Variant | r_pixel mean | r_pixel std | full_image_l1 | center_face_crop_l1 | mouth_lower_face_crop_l1 | upper_face_crop_l1 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, stats in variants.items():
        pixel = stats.get('r_pixel', {})
        full_l1 = stats.get('full_image_l1', {})
        center_l1 = stats.get('center_face_crop_l1', {})
        mouth_l1 = stats.get('mouth_lower_face_crop_l1', {})
        upper_l1 = stats.get('upper_face_crop_l1', {})
        lines.append(
            f'| {name} | {pixel.get("mean")} | {pixel.get("std")} | '
            f'{full_l1.get("mean")} | {center_l1.get("mean")} | '
            f'{mouth_l1.get("mean")} | {upper_l1.get("mean")} |'
        )
    lines += [
        f'* C < B percentage: `{improvement.get("percent_batches_c_lt_b")}`',
        f'* C < D percentage: `{improvement.get("percent_batches_c_lt_d")}`',
        f'* C < E percentage: `{improvement.get("percent_batches_c_lt_e")}`',
        f'* C < F percentage: `{improvement.get("percent_batches_c_lt_f")}`',
        f'* image grids: `{metrics.get("image_grids")}`',
    ]
    c_e = improvement.get('percent_batches_c_lt_e')
    c_f = improvement.get('percent_batches_c_lt_f')
    if c_e is not None and c_f is not None and c_e > 0.5 and c_f > 0.5:
        lines.append('* token-content sanity conclusion: correct token beats wrong/zero controls on most batches.')
    else:
        lines.append('* token-content sanity conclusion: correct token does not consistently beat wrong/zero controls; inspect grids and sensitivity reports.')
lines += ['', '## Adapter Sensitivity Reports']
for name in ['train'] + [f'holdout{x}' for x in holdouts]:
    lines.append(f'* {name}: `{sens_root / name / "report.md"}`')
summary = eval_root / 'summary_report.md'
summary.write_text('\n'.join(lines) + '\n')
print(f'[P9.2J] wrote summary: {summary}')
PY

echo "[P9.2J] Done. Summary: ${EVAL_ROOT}/summary_report.md"
