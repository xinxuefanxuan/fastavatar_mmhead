#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/train/fastavatar_motion_zero_token_overfit_micro.yaml}"
LOCAL_PATHS_CONFIG="${P9_LOCAL_PATHS_CONFIG:-configs/local/p9_2_local_paths.yaml}"
RUNTIME_CONFIG_DIR="${P9_RUNTIME_CONFIG_DIR:-outputs/mmhead_debug/p9_2_runtime_configs}"

export FASTAVATAR_TOKEN_DEBUG="${FASTAVATAR_TOKEN_DEBUG:-1}"
export FASTAVATAR_DATASET_FAIL_FAST="${FASTAVATAR_DATASET_FAIL_FAST:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUNTIME_CONFIG_DIR}"

# Resolve local path overrides without mutating the source config.
eval "$(python - "${CONFIG_PATH}" "${LOCAL_PATHS_CONFIG}" "${RUNTIME_CONFIG_DIR}" <<'PY'
from pathlib import Path
import os
import shlex
import sys
from omegaconf import OmegaConf

repo = Path.cwd()
config_path = Path(sys.argv[1]).expanduser()
if not config_path.is_absolute():
    config_path = (repo / config_path).resolve()
local_path = Path(sys.argv[2]).expanduser()
if not local_path.is_absolute():
    local_path = (repo / local_path).resolve()
runtime_dir = Path(sys.argv[3]).expanduser()
if not runtime_dir.is_absolute():
    runtime_dir = (repo / runtime_dir).resolve()

cfg = OmegaConf.load(config_path)
local = OmegaConf.create({})
if local_path.exists():
    local = OmegaConf.load(local_path)

def cfg_get(obj, key, default=None):
    return getattr(obj, key) if hasattr(obj, key) else default

def resolve(value):
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo / p).resolve()

base_root = cfg.dataset.datasets.nersemble.root_dir
base_generated = cfg.dataset.meta_path
source_default = "datasets/mixed_uids.json"
root = os.environ.get("P9_NERSEMBLE_ROOT") or cfg_get(local, "nersemble_root_dir", None) or base_root
source = os.environ.get("P9_SOURCE_META") or cfg_get(local, "source_mixed_uids", None) or source_default
generated = os.environ.get("P9_GENERATED_META") or cfg_get(local, "generated_meta_path", None) or base_generated
preferred = os.environ.get("P9_VAL_ID") or cfg_get(local, "preferred_val_id", None) or ""

runtime_config = runtime_dir / f"{config_path.stem}_resolved.yaml"
values = {
    "P9_RESOLVED_CONFIG_PATH": str(runtime_config),
    "P9_RESOLVED_NERSEMBLE_ROOT": str(resolve(root)),
    "P9_RESOLVED_SOURCE_META": str(resolve(source)),
    "P9_RESOLVED_GENERATED_META": str(resolve(generated)),
    "P9_RESOLVED_PREFERRED_VAL_ID": str(preferred or ""),
    "P9_BASE_CONFIG_PATH": str(config_path),
    "P9_LOCAL_PATHS_CONFIG": str(local_path),
}
for key, value in values.items():
    print(f"export {key}={shlex.quote(value)}")
PY
)"

echo "[P9.2] Running motion-zero + GT motion-token overfit"
echo "[P9.2] cwd=$(pwd)"
echo "[P9.2] CONFIG_PATH=${CONFIG_PATH}"
echo "[P9.2] base_config=${P9_BASE_CONFIG_PATH}"
echo "[P9.2] local_paths_config=${P9_LOCAL_PATHS_CONFIG} exists=$(test -f "${P9_LOCAL_PATHS_CONFIG}" && echo true || echo false)"
echo "[P9.2] runtime_config=${P9_RESOLVED_CONFIG_PATH}"
echo "[P9.2] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[P9.2] FASTAVATAR_TOKEN_DEBUG=${FASTAVATAR_TOKEN_DEBUG}"
echo "[P9.2] FASTAVATAR_DATASET_FAIL_FAST=${FASTAVATAR_DATASET_FAIL_FAST}"
echo "[P9.2] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[P9.2] P9_NERSEMBLE_ROOT=${P9_RESOLVED_NERSEMBLE_ROOT}"
echo "[P9.2] P9_SOURCE_META=${P9_RESOLVED_SOURCE_META}"
echo "[P9.2] P9_GENERATED_META=${P9_RESOLVED_GENERATED_META}"
echo "[P9.2] preferred_val_id=${P9_RESOLVED_PREFERRED_VAL_ID:-<auto>}"

python scripts/debug/create_p9_2_overfit_metadata.py \
  --config "${P9_BASE_CONFIG_PATH}" \
  --src_meta "${P9_RESOLVED_SOURCE_META}" \
  --root_dir "${P9_RESOLVED_NERSEMBLE_ROOT}" \
  --output "${P9_RESOLVED_GENERATED_META}" \
  --min_pairs auto \
  --max_ids 6 \
  --max_items_per_id 4

# Pick a valid val_id from generated metadata and write the resolved runtime config.
eval "$(python - "${P9_BASE_CONFIG_PATH}" "${P9_RESOLVED_GENERATED_META}" "${P9_RESOLVED_CONFIG_PATH}" "${P9_RESOLVED_NERSEMBLE_ROOT}" "${P9_RESOLVED_PREFERRED_VAL_ID}" <<'PY'
from pathlib import Path
from collections import Counter
import json
import shlex
import sys
from omegaconf import OmegaConf

base_config = Path(sys.argv[1])
meta_path = Path(sys.argv[2])
runtime_config = Path(sys.argv[3])
root_dir = sys.argv[4]
preferred = sys.argv[5].strip()

with meta_path.open("r", encoding="utf-8") as f:
    meta = json.load(f)

def extract_uid(key: str) -> str:
    if key.startswith("nersemble/"):
        key = key[len("nersemble/"):]
    return key.split("/", 1)[0]

ids = sorted({extract_uid(k) for k in meta})
if not ids:
    raise RuntimeError(f"Generated metadata contains no selectable IDs: {meta_path}")
if preferred and preferred in ids:
    val_id = preferred
elif preferred:
    print(f"[P9.2][WARN] requested val_id={preferred} is not in selected IDs {ids}; choosing a valid ID instead", file=sys.stderr)
    val_id = ids[-1]
else:
    val_id = ids[-1]
train_ids = [uid for uid in ids if uid != val_id]
if not train_ids:
    raise RuntimeError(f"Chosen val_id={val_id} would leave zero train IDs from selected IDs={ids}")

cfg = OmegaConf.load(base_config)
cfg.dataset.meta_path = str(meta_path)
cfg.dataset.datasets.nersemble.root_dir = root_dir
cfg.dataset.datasets.nersemble.val_id = [val_id]
runtime_config.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, runtime_config)

print(f"[P9.2] selected_ids={ids}", file=sys.stderr)
print(f"[P9.2] selected_id_counts={dict(Counter(extract_uid(k) for k in meta))}", file=sys.stderr)
print(f"[P9.2] chosen_val_id={val_id}", file=sys.stderr)
print(f"[P9.2] train_ids={train_ids}", file=sys.stderr)
print(f"[P9.2] wrote_runtime_config={runtime_config}", file=sys.stderr)
print(f"export P9_CHOSEN_VAL_ID={shlex.quote(val_id)}")
PY
)"

echo "[P9.2] using chosen val_id=${P9_CHOSEN_VAL_ID}"

python scripts/debug/inspect_fastavatar_dataset_ids.py \
  --config "${P9_RESOLVED_CONFIG_PATH}" \
  --min_pairs auto \
  --require_train

python scripts/debug/inspect_p9_2_runtime_config.py \
  --config "${P9_RESOLVED_CONFIG_PATH}" \
  --require_micro

python FastAvatar/launch.py train.fastavatar --config "${P9_RESOLVED_CONFIG_PATH}" "$@"
