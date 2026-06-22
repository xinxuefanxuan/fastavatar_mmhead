#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/p7p8_stable_demo_check}"
GENERATOR="${GENERATOR:-motion_model/generate_from_text_temporal.py}"
SAFETY_CONFIG="${SAFETY_CONFIG:-configs/p7p8_motion_safety_stable.yaml}"
RENDER_WRAPPER="${RENDER_WRAPPER:-scripts/diagnostics/render_fastavatar_case.sh}"
SEQUENCE_NAME="${SEQUENCE_NAME:-nersemble_seq_214}"
NEUTRAL_TEMPLATE="${NEUTRAL_TEMPLATE:-assets/sample_motion/nersemble_seq_214_neutral}"
INFERENCE_N_FRAMES="${INFERENCE_N_FRAMES:-16}"
MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER:-1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-7}"
RENDER="${RENDER:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

MOTION_DIR="${OUTPUT_DIR}/motion_npz"
FRAME_DIR="${OUTPUT_DIR}/frames"
VIDEO_DIR="${OUTPUT_DIR}/videos"
REPORT="${OUTPUT_DIR}/summary_report.md"
GRID="${OUTPUT_DIR}/grid.png"
RENDER_STATUS="${OUTPUT_DIR}/render_status.tsv"

mkdir -p "${OUTPUT_DIR}" "${MOTION_DIR}" "${FRAME_DIR}" "${VIDEO_DIR}"

printf '[StableDemo] cwd=%s\n' "$(pwd)"
printf '[StableDemo] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[StableDemo] safety_config=%s\n' "${SAFETY_CONFIG}"
printf '[StableDemo] generator=%s\n' "${GENERATOR}"
printf '[StableDemo] render=%s skip_existing=%s\n' "${RENDER}" "${SKIP_EXISTING}"
printf '[StableDemo] inference_n_frames=%s max_single_frame_render=%s cuda=%s\n' "${INFERENCE_N_FRAMES}" "${MAX_SINGLE_FRAME_RENDER}" "${CUDA_VISIBLE_DEVICES_VALUE}"

if [[ ! -f "${GENERATOR}" ]]; then
  printf '[ERROR] generator not found: %s\n' "${GENERATOR}" >&2
  exit 1
fi
if [[ ! -f "${SAFETY_CONFIG}" ]]; then
  printf '[ERROR] safety config not found: %s\n' "${SAFETY_CONFIG}" >&2
  exit 1
fi
if [[ "${RENDER}" == "1" && ! -f "${RENDER_WRAPPER}" ]]; then
  printf '[ERROR] render wrapper not found: %s\n' "${RENDER_WRAPPER}" >&2
  exit 1
fi
if [[ "${RENDER}" == "1" && ! -d "${NEUTRAL_TEMPLATE}" ]]; then
  printf '[ERROR] neutral template not found: %s\n' "${NEUTRAL_TEMPLATE}" >&2
  exit 1
fi

CASES=(
  "turn_left|turn left|default|1.0"
  "turn_right|turn right|default|1.0"
  "open_mouth|open mouth|default|1.0"
  "subtle_smile|subtle smile|subtle|1.0"
  "turn_right_and_subtle_smile|turn right and subtle smile|subtle|1.0"
)

printf 'case\tstatus\tvideo\n' > "${RENDER_STATUS}"

for item in "${CASES[@]}"; do
  IFS='|' read -r case_name prompt smile_profile smile_gain <<< "${item}"
  case_dir="${OUTPUT_DIR}/${case_name}"
  mkdir -p "${case_dir}"
  post_npz="${MOTION_DIR}/${case_name}.npz"
  pre_npz="${MOTION_DIR}/${case_name}_pre_safety.npz"
  plan_json="${case_dir}/plan.json"
  safety_report="${case_dir}/safety_report.json"
  generate_log="${case_dir}/generate.log"
  wrapper_log="${case_dir}/render_wrapper.log"
  video_out="${VIDEO_DIR}/${case_name}.mp4"

  printf '[StableDemo] generating %s: %s\n' "${case_name}" "${prompt}"
  python "${GENERATOR}" \
    --prompt "${prompt}" \
    --output_npz "${post_npz}" \
    --output_plan_json "${plan_json}" \
    --motion_safety_config "${SAFETY_CONFIG}" \
    --apply_motion_safety \
    --save_pre_safety_motion "${pre_npz}" \
    --save_safety_report "${safety_report}" \
    --smile_profile "${smile_profile}" \
    --smile_gain "${smile_gain}" > "${generate_log}" 2>&1

  if [[ "${RENDER}" != "1" ]]; then
    printf '%s\tnot_run\t%s\n' "${case_name}" "${video_out}" >> "${RENDER_STATUS}"
    continue
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${video_out}" ]]; then
    printf '[StableDemo] skipping existing video %s\n' "${video_out}"
    printf '%s\tskipped_existing\t%s\n' "${case_name}" "${video_out}" >> "${RENDER_STATUS}"
    continue
  fi

  printf '[StableDemo] rendering %s\n' "${case_name}"
  MAX_SINGLE_FRAME_RENDER="${MAX_SINGLE_FRAME_RENDER}" \
  bash "${RENDER_WRAPPER}" \
    --motion_npz "${post_npz}" \
    --output_dir "${OUTPUT_DIR}" \
    --case_name "${case_name}" \
    --sequence_name "${SEQUENCE_NAME}" \
    --neutral_template "${NEUTRAL_TEMPLATE}" \
    --pack_root "${case_dir}/fastavatar_pack" \
    --head_target neck_pose \
    --inference_n_frames "${INFERENCE_N_FRAMES}" \
    --cuda_visible_devices "${CUDA_VISIBLE_DEVICES_VALUE}" > "${wrapper_log}" 2>&1

  cp -f "${OUTPUT_DIR}/fastavatar_video/${case_name}.mp4" "${video_out}"
  printf '%s\tok\t%s\n' "${case_name}" "${video_out}" >> "${RENDER_STATUS}"
done

if [[ "${RENDER}" == "1" ]]; then
  rm -f "${FRAME_DIR}"/*.png
  for item in "${CASES[@]}"; do
    IFS='|' read -r case_name _prompt _smile_profile _smile_gain <<< "${item}"
    video="${VIDEO_DIR}/${case_name}.mp4"
    if [[ ! -f "${video}" ]]; then
      printf '[ERROR] expected video missing: %s\n' "${video}" >&2
      exit 1
    fi
    ffmpeg -y -hide_banner -loglevel error -i "${video}" \
      -vf "select='eq(n,0)+eq(n,52)+eq(n,104)',scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2" \
      -vsync 0 "${FRAME_DIR}/${case_name}_%02d.png"
  done

  grid_inputs=()
  for item in "${CASES[@]}"; do
    IFS='|' read -r case_name _prompt _smile_profile _smile_gain <<< "${item}"
    grid_inputs+=(
      "${FRAME_DIR}/${case_name}_01.png"
      "${FRAME_DIR}/${case_name}_02.png"
      "${FRAME_DIR}/${case_name}_03.png"
    )
  done
  montage "${grid_inputs[@]}" -tile 3x -geometry 256x256+8+8 "${GRID}"
fi

python - "${OUTPUT_DIR}" "${SAFETY_CONFIG}" "${RENDER_STATUS}" "${REPORT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

output_dir = Path(sys.argv[1])
safety_config = Path(sys.argv[2])
render_status = Path(sys.argv[3])
report = Path(sys.argv[4])

cases = [
    ("turn_left", "turn left", "default"),
    ("turn_right", "turn right", "default"),
    ("open_mouth", "open mouth", "default"),
    ("subtle_smile", "subtle smile", "subtle"),
    ("turn_right_and_subtle_smile", "turn right and subtle smile", "subtle"),
]

status = {}
if render_status.exists():
    for line in render_status.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            status[parts[0]] = {"status": parts[1], "video": parts[2]}

def stats_for(case_name: str) -> dict:
    npz = output_dir / "motion_npz" / f"{case_name}.npz"
    if not npz.exists():
        return {}
    motion = np.asarray(np.load(npz)["motion"], dtype=np.float32)
    head = motion[:, 50:53]
    expr = motion[:, :50]
    jaw = motion[:, 53:56]
    head_step = np.linalg.norm(np.diff(head, axis=0), axis=1) if len(head) > 1 else np.zeros((0,), dtype=np.float32)
    return {
        "yaw_min": float(head[:, 1].min()),
        "yaw_max": float(head[:, 1].max()),
        "head_step_max": float(head_step.max()) if len(head_step) else 0.0,
        "expr_norm_max": float(np.linalg.norm(expr, axis=1).max()),
        "jaw_max": float(np.max(np.abs(jaw))) if jaw.size else 0.0,
    }

lines = [
    "# P7/P8 Stable Demo Check",
    "",
    f"Output directory: `{output_dir}`",
    f"Stable safety config: `{safety_config}`",
    "Grid: `grid.png`",
    "Videos: `videos/`",
    "",
    "## Stable Preset",
    "",
    "- `max_abs_yaw: 0.18`",
    "- `max_head_step: 0.018`",
    "- `head_smooth_window: 5`",
    "- `turn_left_scale: 1.0`",
    "- `turn_right_scale: 1.0`",
    "- smile is intentionally kept as `subtle_smile`; rank03 and random multi-dim vectors are not used.",
    "- current FastAvatar/MMHead smile basis does not support clean teeth smile robustly; keep gentle/open-teeth smile out of the default demo preset.",
    "- `open_mouth` is retained because jaw response was validated in FastAvatar renders.",
    "",
    "## Demo Cases",
    "",
    "| case | prompt | smile profile | yaw min | yaw max | max head step | expr max | jaw max | video | status |",
    "|---|---|---|---:|---:|---:|---:|---:|---|---|",
]
for case_name, prompt, smile_profile in cases:
    s = stats_for(case_name)
    st = status.get(case_name, {})
    video = st.get("video", str(output_dir / "videos" / f"{case_name}.mp4"))
    rel_video = Path(video)
    try:
        rel_video = rel_video.relative_to(output_dir)
    except ValueError:
        pass
    lines.append(
        f"| `{case_name}` | {prompt} | `{smile_profile}` | "
        f"{s.get('yaw_min', 0.0):.4f} | {s.get('yaw_max', 0.0):.4f} | "
        f"{s.get('head_step_max', 0.0):.4f} | {s.get('expr_norm_max', 0.0):.4f} | "
        f"{s.get('jaw_max', 0.0):.4f} | [video]({rel_video}) | `{st.get('status', 'missing')}` |"
    )

lines += [
    "",
    "## Diagnostic Conclusions",
    "",
    "- turn yaw uses the stable sweep recommendation: yaw cap `0.18`, smoother head window `5`, and max head step `0.018`.",
    "- turn left/right scales are symmetric at `1.0`; right turn is no longer reduced by the older `0.8` scale.",
    "- smile is deliberately conservative: `subtle_smile` uses the existing 50D MMHead/P7/P8 smile basis with the generator's subtle profile, not rank03, not gentle-teeth ablation, and not random 100D multi-dim search vectors.",
    "- open mouth remains a separate supported demo because jaw response is reliable in FastAvatar.",
    "- `turn_right_and_subtle_smile` is included as the stable composition case; treat it as a subtle expression demo rather than a clean-teeth smile.",
    "",
    "## Manual Review Checklist",
    "",
    "- Inspect `grid.png`: each row is a case, columns are start/mid/end frames.",
    "- Confirm turn right does not reintroduce the strong artifact seen in earlier aggressive yaw settings.",
    "- Confirm subtle smile is natural enough as a weak expression and does not become a forced grin.",
    "- Confirm open mouth remains visible.",
]

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[StableDemo] report={report}")
PY

printf '[StableDemo] report=%s\n' "${REPORT}"
if [[ "${RENDER}" == "1" ]]; then
  printf '[StableDemo] grid=%s\n' "${GRID}"
  printf '[StableDemo] videos=%s\n' "${VIDEO_DIR}"
fi
