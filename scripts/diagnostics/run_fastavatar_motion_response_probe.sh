#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/p7p8_response_probe}"
NEUTRAL_TEMPLATE="${NEUTRAL_TEMPLATE:-assets/sample_motion/nersemble_seq_214_neutral}"
SEQUENCE_NAME="${SEQUENCE_NAME:-nersemble_seq_214}"
SMILE_MOTION_NPZ="${SMILE_MOTION_NPZ:-outputs/mmhead_debug/p7p8_quality_diagnosis/smile/motion_post_safety.npz}"
OPEN_MOUTH_MOTION_NPZ="${OPEN_MOUTH_MOTION_NPZ:-outputs/mmhead_debug/p7p8_quality_diagnosis/open_mouth/motion_post_safety.npz}"
RENDER_WRAPPER="${RENDER_WRAPPER:-scripts/diagnostics/render_fastavatar_case.sh}"
INFERENCE_N_FRAMES="${INFERENCE_N_FRAMES:-32}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-7}"
SMILE_GAIN="${SMILE_GAIN:-3.0}"
OPEN_MOUTH_JAW="${OPEN_MOUTH_JAW:-0.35}"
NECK_YAW="${NECK_YAW:-0.25}"
ROTATION_YAW="${ROTATION_YAW:-0.25}"

NPZ_DIR="${OUTPUT_DIR}/probe_npz"
VIDEO_DIR="${OUTPUT_DIR}/fastavatar_video"
FRAME_DIR="${OUTPUT_DIR}/frames"
REPORT="${OUTPUT_DIR}/response_probe_report.md"
GRID="${OUTPUT_DIR}/response_probe_grid.png"

mkdir -p "${OUTPUT_DIR}" "${NPZ_DIR}" "${VIDEO_DIR}" "${FRAME_DIR}"

printf '[ResponseProbe] cwd=%s\n' "$(pwd)"
printf '[ResponseProbe] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[ResponseProbe] neutral_template=%s\n' "${NEUTRAL_TEMPLATE}"
printf '[ResponseProbe] sequence_name=%s\n' "${SEQUENCE_NAME}"
printf '[ResponseProbe] inference_n_frames=%s\n' "${INFERENCE_N_FRAMES}"
printf '[ResponseProbe] cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES_VALUE}"

if [[ ! -d "${NEUTRAL_TEMPLATE}" ]]; then
  printf '[ERROR] neutral template not found: %s\n' "${NEUTRAL_TEMPLATE}" >&2
  exit 1
fi
if [[ ! -f "${RENDER_WRAPPER}" ]]; then
  printf '[ERROR] render wrapper not found: %s\n' "${RENDER_WRAPPER}" >&2
  exit 1
fi
if [[ ! -f "${SMILE_MOTION_NPZ}" ]]; then
  printf '[ERROR] smile motion npz not found: %s\n' "${SMILE_MOTION_NPZ}" >&2
  exit 1
fi
if [[ ! -f "${OPEN_MOUTH_MOTION_NPZ}" ]]; then
  printf '[ERROR] open mouth motion npz not found: %s\n' "${OPEN_MOUTH_MOTION_NPZ}" >&2
  exit 1
fi

python - "${NPZ_DIR}" "${SMILE_MOTION_NPZ}" "${OPEN_MOUTH_MOTION_NPZ}" "${SMILE_GAIN}" "${OPEN_MOUTH_JAW}" "${NECK_YAW}" "${ROTATION_YAW}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

npz_dir = Path(sys.argv[1])
smile_npz = Path(sys.argv[2])
open_mouth_npz = Path(sys.argv[3])
smile_gain = float(sys.argv[4])
open_mouth_jaw = float(sys.argv[5])
neck_yaw = float(sys.argv[6])
rotation_yaw = float(sys.argv[7])

npz_dir.mkdir(parents=True, exist_ok=True)
template = np.load(smile_npz, allow_pickle=True)
base = np.zeros_like(np.asarray(template["motion"], dtype=np.float32))
if base.ndim != 2 or base.shape[1] < 56:
    raise SystemExit(f"unexpected motion shape: {base.shape}")

def ramped(target: np.ndarray) -> np.ndarray:
    out = np.asarray(target, dtype=np.float32).copy()
    ramp = np.linspace(0.0, 1.0, out.shape[0], dtype=np.float32)[:, None]
    return out * ramp

def save_case(name: str, motion: np.ndarray, head_target: str = "neck_pose") -> dict:
    motion = ramped(motion)
    payload = {
        "motion": motion.astype(np.float32),
        "motion_raw": motion.astype(np.float32),
        "motion_norm": motion.astype(np.float32),
        "expr_delta": motion[:, 0:50].astype(np.float32),
        "head_delta": motion[:, 50:53].astype(np.float32),
        "jaw_delta": motion[:, 53:56].astype(np.float32),
    }
    path = npz_dir / f"{name}.npz"
    np.savez(path, **payload)
    return {
        "name": name,
        "npz": str(path),
        "head_target": head_target,
        "expr_norm_max": float(np.linalg.norm(payload["expr_delta"], axis=1).max()),
        "head_delta_min": payload["head_delta"].min(axis=0).astype(float).tolist(),
        "head_delta_max": payload["head_delta"].max(axis=0).astype(float).tolist(),
        "jaw_delta_min": payload["jaw_delta"].min(axis=0).astype(float).tolist(),
        "jaw_delta_max": payload["jaw_delta"].max(axis=0).astype(float).tolist(),
    }

cases = []
cases.append(save_case("neutral_baseline", base))

motion = base.copy()
motion[:, 51] = -neck_yaw
cases.append(save_case("strong_neck_yaw_right", motion, "neck_pose"))

motion = base.copy()
motion[:, 51] = neck_yaw
cases.append(save_case("strong_neck_yaw_left", motion, "neck_pose"))

motion = base.copy()
motion[:, 51] = -rotation_yaw
cases.append(save_case("strong_rotation_yaw_right", motion, "rotation"))

motion = base.copy()
smile_motion = np.asarray(template["motion"], dtype=np.float32)
motion[:, 0:50] = smile_motion[:, 0:50] * smile_gain
cases.append(save_case(f"strong_smile_x{smile_gain:g}".replace(".", "p"), motion, "neck_pose"))

motion = base.copy()
open_data = np.load(open_mouth_npz, allow_pickle=True)
open_motion = np.asarray(open_data["motion"], dtype=np.float32)
motion[:, 0:50] = open_motion[:, 0:50]
motion[:, 53] = open_mouth_jaw
cases.append(save_case("strong_open_mouth", motion, "neck_pose"))

(npz_dir / "probe_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
for case in cases:
    print(f"{case['name']}|{case['npz']}|{case['head_target']}")
PY

mapfile -t CASES < <(python - "${NPZ_DIR}/probe_cases.json" <<'PY'
import json
import sys
for case in json.load(open(sys.argv[1], encoding="utf-8")):
    print(f"{case['name']}|{case['npz']}|{case['head_target']}")
PY
)

for item in "${CASES[@]}"; do
  IFS='|' read -r case_name motion_npz head_target <<< "${item}"
  printf '[ResponseProbe] rendering %s head_target=%s\n' "${case_name}" "${head_target}"
  mkdir -p "${OUTPUT_DIR}/${case_name}"
  bash "${RENDER_WRAPPER}" \
    --motion_npz "${motion_npz}" \
    --output_dir "${OUTPUT_DIR}" \
    --case_name "${case_name}" \
    --sequence_name "${SEQUENCE_NAME}" \
    --neutral_template "${NEUTRAL_TEMPLATE}" \
    --pack_root "${OUTPUT_DIR}/${case_name}/fastavatar_pack" \
    --head_target "${head_target}" \
    --inference_n_frames "${INFERENCE_N_FRAMES}" \
    --cuda_visible_devices "${CUDA_VISIBLE_DEVICES_VALUE}" > "${OUTPUT_DIR}/${case_name}/render_wrapper.log" 2>&1
done

for item in "${CASES[@]}"; do
  IFS='|' read -r case_name _motion_npz _head_target <<< "${item}"
  video="${VIDEO_DIR}/${case_name}.mp4"
  if [[ ! -f "${video}" ]]; then
    printf '[ERROR] expected video missing: %s\n' "${video}" >&2
    exit 1
  fi
  ffmpeg -y -hide_banner -loglevel error -i "${video}" \
    -vf "select='eq(n,0)+eq(n,52)+eq(n,104)'" -vsync 0 \
    "${FRAME_DIR}/${case_name}_%02d.png"
done

montage_inputs=()
for item in "${CASES[@]}"; do
  IFS='|' read -r case_name _motion_npz _head_target <<< "${item}"
  montage_inputs+=(
    "${FRAME_DIR}/${case_name}_01.png"
    "${FRAME_DIR}/${case_name}_02.png"
    "${FRAME_DIR}/${case_name}_03.png"
  )
done
montage "${montage_inputs[@]}" -tile 3x6 -geometry 256x256+8+8 "${GRID}"

python - "${OUTPUT_DIR}" "${NPZ_DIR}/probe_cases.json" "${REPORT}" <<'PY'
import json
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
cases = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
report = Path(sys.argv[3])

def grep_line(path: Path, pattern: str) -> str:
    if not path.exists():
        return "missing"
    rx = re.compile(pattern)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if rx.search(line):
            return line.strip()
    return "not found"

lines = [
    "# P7/P8 FastAvatar Motion Response Probe",
    "",
    f"Output directory: `{output_dir}`",
    f"Response grid: `response_probe_grid.png`",
    "",
    "## Probe Cases",
    "",
    "| case | head_target | npz | video | motion_seqs_dir in infer.log | pack target evidence |",
    "|---|---|---|---|---|---|",
]

for case in cases:
    name = case["name"]
    infer_log = output_dir / name / "fastavatar_render" / "infer.log"
    pack_log = output_dir / name / "fastavatar_render" / "pack.log"
    motion_line = grep_line(infer_log, r"^MOTION_SEQS_DIR:|Preparing motion sequences from:")
    pack_line = grep_line(pack_log, r"^\[FastAvatarPack\] root_flame_param=|^\[Write\] sequence_name=")
    lines.append(
        f"| {name} | {case['head_target']} | `{case['npz']}` | "
        f"[video](fastavatar_video/{name}.mp4) | `{motion_line}` | `{pack_line}` |"
    )

lines += [
    "",
    "## Motion Magnitudes",
    "",
    "| case | expr_norm_max | head_delta_min | head_delta_max | jaw_delta_min | jaw_delta_max |",
    "|---|---:|---|---|---|---|",
]
for case in cases:
    lines.append(
        f"| {case['name']} | {case['expr_norm_max']:.6f} | "
        f"`{case['head_delta_min']}` | `{case['head_delta_max']}` | "
        f"`{case['jaw_delta_min']}` | `{case['jaw_delta_max']}` |"
    )

lines += [
    "",
    "## Questions To Answer From The Grid",
    "",
    "- Confirm `infer.log` uses the corresponding probe pack by checking `MOTION_SEQS_DIR` or `Preparing motion sequences from` above.",
    "- If `strong_neck_yaw_right` visibly turns right, `neck_pose` edits reach the final render.",
    "- If `strong_rotation_yaw_right` is stronger than `strong_neck_yaw_right`, FastAvatar may respond more to global `rotation` than `neck_pose`.",
    "- If `strong_smile_x*` or `strong_open_mouth` is visible, expression/jaw fields are effective but normal P7/P8 amplitude or gain may be weak.",
    "- If even extreme probes are not visible, suspect infer data path, FastAvatar field reading, mask/source-view constraints, or renderer range.",
    "",
    "## Manual Visual Answers",
    "",
    "- infer data root: answered automatically in the table above; each row should point to its own probe pack.",
    "- strong_neck_yaw_right: inspect `response_probe_grid.png` row 2, columns start/mid/end.",
    "- strong_rotation_yaw_right vs neck_pose: compare row 4 against row 2.",
    "- strong_smile / strong_open_mouth: inspect rows 5 and 6.",
    "- Interpretation: fill in after visual inspection of the generated grid.",
]

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ResponseProbe] report={report}")
PY

printf '[ResponseProbe] grid=%s\n' "${GRID}"
printf '[ResponseProbe] report=%s\n' "${REPORT}"
