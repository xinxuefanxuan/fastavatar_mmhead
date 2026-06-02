#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/train/fastavatar_motion_zero_token_overfit_micro.yaml}"

export FASTAVATAR_TOKEN_DEBUG="${FASTAVATAR_TOKEN_DEBUG:-1}"
export FASTAVATAR_DATASET_FAIL_FAST="${FASTAVATAR_DATASET_FAIL_FAST:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

META_PATH=$(python - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
p = Path(str(cfg.dataset.meta_path)).expanduser()
if not p.is_absolute():
    p = Path.cwd() / p
print(p)
PY
)

echo "[P9.2] Running motion-zero + GT motion-token overfit"
echo "[P9.2] cwd=$(pwd)"
echo "[P9.2] CONFIG_PATH=${CONFIG_PATH}"
echo "[P9.2] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[P9.2] FASTAVATAR_TOKEN_DEBUG=${FASTAVATAR_TOKEN_DEBUG}"
echo "[P9.2] FASTAVATAR_DATASET_FAIL_FAST=${FASTAVATAR_DATASET_FAIL_FAST}"
echo "[P9.2] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[P9.2] meta_path=${META_PATH}"

python scripts/debug/create_p9_2_overfit_metadata.py \
  --config "${CONFIG_PATH}" \
  --min_pairs auto \
  --max_ids 6 \
  --max_items_per_id 4 \
  --prefer_ids 036 \
  --output datasets/p9_2_overfit_mixed_uids.json

python scripts/debug/inspect_fastavatar_dataset_ids.py \
  --config "${CONFIG_PATH}" \
  --min_pairs auto \
  --require_train

python scripts/debug/inspect_p9_2_runtime_config.py \
  --config "${CONFIG_PATH}" \
  --require_micro

python FastAvatar/launch.py train.fastavatar --config "${CONFIG_PATH}" "$@"
