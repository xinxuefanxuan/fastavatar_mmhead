#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/train/fastavatar_motion_zero_token_overfit.yaml}"

export FASTAVATAR_TOKEN_DEBUG="${FASTAVATAR_TOKEN_DEBUG:-1}"

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
echo "[P9.2] Config: ${CONFIG_PATH}"
echo "[P9.2] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[P9.2] FASTAVATAR_TOKEN_DEBUG=${FASTAVATAR_TOKEN_DEBUG}"
echo "[P9.2] meta_path=${META_PATH}"

if [[ ! -f "${META_PATH}" ]]; then
  cat >&2 <<EOF
[P9.2][ERROR] Configured meta_path does not exist: ${META_PATH}
Create it without copying large data by running:
  python scripts/debug/create_p9_2_overfit_metadata.py
EOF
  exit 2
fi

python scripts/debug/inspect_fastavatar_dataset_ids.py --config "${CONFIG_PATH}" --require_train

python FastAvatar/launch.py train.fastavatar --config "${CONFIG_PATH}" "$@"
