#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmhead_debug/p7p8_turn_yaw_sweep}"
NEUTRAL_TEMPLATE="${NEUTRAL_TEMPLATE:-assets/sample_motion/nersemble_seq_214_neutral}"
SEQUENCE_NAME="${SEQUENCE_NAME:-nersemble_seq_214}"
RENDER_WRAPPER="${RENDER_WRAPPER:-scripts/diagnostics/render_fastavatar_case.sh}"
INFERENCE_N_FRAMES="${INFERENCE_N_FRAMES:-32}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-7}"
YAW_VALUES="${YAW_VALUES:-0.10 0.12 0.15 0.18 0.22 0.25}"
SWEEP_PROFILES="${SWEEP_PROFILES:-default longer_smooth_step018 longer_smooth_step020}"
SWEEP_DIRECTIONS="${SWEEP_DIRECTIONS:-turn_right turn_left}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

NPZ_DIR="${OUTPUT_DIR}/motion_npz"
FRAME_DIR="${OUTPUT_DIR}/frames"
GRID_DIR="${OUTPUT_DIR}/grids"
REPORT="${OUTPUT_DIR}/sweep_report.md"
ALL_GRID="${OUTPUT_DIR}/turn_yaw_sweep_grid.png"

mkdir -p "${OUTPUT_DIR}" "${NPZ_DIR}" "${FRAME_DIR}" "${GRID_DIR}"

printf '[TurnYawSweep] cwd=%s\n' "$(pwd)"
printf '[TurnYawSweep] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[TurnYawSweep] yaw_values=%s\n' "${YAW_VALUES}"
printf '[TurnYawSweep] profiles=%s\n' "${SWEEP_PROFILES}"
printf '[TurnYawSweep] directions=%s\n' "${SWEEP_DIRECTIONS}"
printf '[TurnYawSweep] skip_existing=%s\n' "${SKIP_EXISTING}"
printf '[TurnYawSweep] inference_n_frames=%s\n' "${INFERENCE_N_FRAMES}"
printf '[TurnYawSweep] cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES_VALUE}"

if [[ ! -d "${NEUTRAL_TEMPLATE}" ]]; then
  printf '[ERROR] neutral template not found: %s\n' "${NEUTRAL_TEMPLATE}" >&2
  exit 1
fi
if [[ ! -f "${RENDER_WRAPPER}" ]]; then
  printf '[ERROR] render wrapper not found: %s\n' "${RENDER_WRAPPER}" >&2
  exit 1
fi

python - "${NPZ_DIR}" "${YAW_VALUES}" "${SWEEP_PROFILES}" "${SWEEP_DIRECTIONS}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

npz_dir = Path(sys.argv[1])
yaw_values = [float(x) for x in sys.argv[2].split()]
profiles = sys.argv[3].split()
directions = sys.argv[4].split()
npz_dir.mkdir(parents=True, exist_ok=True)

T = 32
D = 56

def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([xp[i : i + window].mean() for i in range(len(x))], dtype=np.float32)

def clamp_step(x: np.ndarray, max_step: float | None) -> np.ndarray:
    if max_step is None or max_step <= 0:
        return x
    out = x.astype(np.float32).copy()
    for i in range(1, len(out)):
        step = out[i] - out[i - 1]
        if abs(step) > max_step:
            out[i] = out[i - 1] + np.sign(step) * max_step
    return out

def yaw_curve(yaw_abs: float, direction: str, profile: str) -> tuple[np.ndarray, dict]:
    sign = -1.0 if direction == "turn_right" else 1.0
    target = sign * yaw_abs
    if profile == "default":
        # Mirrors the current P7/P8 turn shape: quick ease-in, hold, then
        # settle to 80% of peak. At yaw=0.12 this yields about 0.025 max step.
        ramp_frames = 8
        smooth_window = 3
        max_head_step = 0.025
        settle_ratio = 0.8
        y = np.zeros(T, dtype=np.float32)
        y[:ramp_frames] = target * smoothstep(np.linspace(1.0 / ramp_frames, 1.0, ramp_frames, dtype=np.float32))
        y[ramp_frames:16] = target
        settle = target * settle_ratio
        y[16:21] = np.linspace(target, settle, 5, dtype=np.float32)
        y[21:] = settle
    elif profile == "longer_smooth_step018":
        ramp_frames = 16
        smooth_window = 5
        max_head_step = 0.018
        settle_ratio = 0.9
        y = np.zeros(T, dtype=np.float32)
        y[:ramp_frames] = target * smoothstep(np.linspace(1.0 / ramp_frames, 1.0, ramp_frames, dtype=np.float32))
        y[ramp_frames:24] = target
        y[24:] = np.linspace(target, target * settle_ratio, T - 24, dtype=np.float32)
    elif profile == "longer_smooth_step020":
        ramp_frames = 16
        smooth_window = 5
        max_head_step = 0.020
        settle_ratio = 0.9
        y = np.zeros(T, dtype=np.float32)
        y[:ramp_frames] = target * smoothstep(np.linspace(1.0 / ramp_frames, 1.0, ramp_frames, dtype=np.float32))
        y[ramp_frames:24] = target
        y[24:] = np.linspace(target, target * settle_ratio, T - 24, dtype=np.float32)
    else:
        raise SystemExit(f"unknown profile: {profile}")
    y = moving_average(y, smooth_window)
    y = clamp_step(y, max_head_step)
    return y.astype(np.float32), {
        "ramp_frames": ramp_frames,
        "smooth_window": smooth_window,
        "max_head_step": max_head_step,
        "settle_ratio": settle_ratio,
    }

cases = []
for direction in directions:
    for profile in profiles:
        for yaw_abs in yaw_values:
            yaw, settings = yaw_curve(yaw_abs, direction, profile)
            motion = np.zeros((T, D), dtype=np.float32)
            motion[:, 51] = yaw
            yaw_tag = f"{yaw_abs:.2f}".replace(".", "p")
            name = f"{direction}_yaw{yaw_tag}_{profile}"
            path = npz_dir / f"{name}.npz"
            np.savez(
                path,
                motion=motion,
                motion_raw=motion,
                motion_norm=motion,
                expr_delta=motion[:, 0:50],
                head_delta=motion[:, 50:53],
                jaw_delta=motion[:, 53:56],
            )
            cases.append({
                "name": name,
                "direction": direction,
                "profile": profile,
                "yaw_abs": yaw_abs,
                "npz": str(path),
                "yaw_min": float(yaw.min()),
                "yaw_max": float(yaw.max()),
                "max_abs_yaw": float(np.abs(yaw).max()),
                "max_head_step_observed": float(np.abs(np.diff(yaw)).max()) if len(yaw) > 1 else 0.0,
                **settings,
            })

(npz_dir / "sweep_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
for case in cases:
    print(f"{case['name']}|{case['npz']}")
PY

mapfile -t CASES < <(python - "${NPZ_DIR}/sweep_cases.json" <<'PY'
import json
import sys
for case in json.load(open(sys.argv[1], encoding="utf-8")):
    print(f"{case['name']}|{case['npz']}")
PY
)

for item in "${CASES[@]}"; do
  IFS='|' read -r case_name motion_npz <<< "${item}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${OUTPUT_DIR}/fastavatar_video/${case_name}.mp4" ]]; then
    printf '[TurnYawSweep] skipping existing video %s\n' "${case_name}"
    continue
  fi
  printf '[TurnYawSweep] rendering %s\n' "${case_name}"
  mkdir -p "${OUTPUT_DIR}/${case_name}"
  bash "${RENDER_WRAPPER}" \
    --motion_npz "${motion_npz}" \
    --output_dir "${OUTPUT_DIR}" \
    --case_name "${case_name}" \
    --sequence_name "${SEQUENCE_NAME}" \
    --neutral_template "${NEUTRAL_TEMPLATE}" \
    --pack_root "${OUTPUT_DIR}/${case_name}/fastavatar_pack" \
    --head_target neck_pose \
    --inference_n_frames "${INFERENCE_N_FRAMES}" \
    --cuda_visible_devices "${CUDA_VISIBLE_DEVICES_VALUE}" > "${OUTPUT_DIR}/${case_name}/render_wrapper.log" 2>&1
done

for item in "${CASES[@]}"; do
  IFS='|' read -r case_name _motion_npz <<< "${item}"
  video="${OUTPUT_DIR}/fastavatar_video/${case_name}.mp4"
  if [[ ! -f "${video}" ]]; then
    printf '[ERROR] expected video missing: %s\n' "${video}" >&2
    exit 1
  fi
  ffmpeg -y -hide_banner -loglevel error -i "${video}" \
    -vf "select='eq(n,0)+eq(n,52)+eq(n,104)'" -vsync 0 \
    "${FRAME_DIR}/${case_name}_%02d.png"
done

all_inputs=()
for item in "${CASES[@]}"; do
  IFS='|' read -r case_name _motion_npz <<< "${item}"
  all_inputs+=(
    "${FRAME_DIR}/${case_name}_01.png"
    "${FRAME_DIR}/${case_name}_02.png"
    "${FRAME_DIR}/${case_name}_03.png"
  )
done
montage "${all_inputs[@]}" -tile 3x -geometry 192x192+6+6 "${ALL_GRID}"

for direction in ${SWEEP_DIRECTIONS}; do
  for profile in ${SWEEP_PROFILES}; do
    inputs=()
    for yaw_abs in ${YAW_VALUES}; do
      yaw_tag="$(printf '%.2f' "${yaw_abs}" | sed 's/\./p/')"
      case_name="${direction}_yaw${yaw_tag}_${profile}"
      inputs+=(
        "${FRAME_DIR}/${case_name}_01.png"
        "${FRAME_DIR}/${case_name}_02.png"
        "${FRAME_DIR}/${case_name}_03.png"
      )
    done
    montage "${inputs[@]}" -tile 3x -geometry 256x256+8+8 "${GRID_DIR}/${direction}_${profile}_grid.png"
  done
done

python - "${OUTPUT_DIR}" "${NPZ_DIR}/sweep_cases.json" "${REPORT}" <<'PY'
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
    "# P7/P8 Turn Yaw FastAvatar Sweep",
    "",
    f"Output directory: `{output_dir}`",
    f"All-case grid: `turn_yaw_sweep_grid.png`",
    "",
    "## Profiles",
    "",
    "| profile | ramp_frames | smooth_window | max_head_step | settle_ratio |",
    "|---|---:|---:|---:|---:|",
]
seen_profiles = {}
for case in cases:
    seen_profiles.setdefault(case["profile"], case)
for profile, case in seen_profiles.items():
    lines.append(
        f"| {profile} | {case['ramp_frames']} | {case['smooth_window']} | "
        f"{case['max_head_step']:.3f} | {case['settle_ratio']:.2f} |"
    )

lines += [
    "",
    "## Case Table",
    "",
    "| case | direction | profile | target yaw | observed max yaw | observed max step | video | infer motion_seqs_dir |",
    "|---|---|---|---:|---:|---:|---|---|",
]
for case in cases:
    name = case["name"]
    infer_log = output_dir / name / "fastavatar_render" / "infer.log"
    motion_line = grep_line(infer_log, r"^MOTION_SEQS_DIR:|Preparing motion sequences from:")
    lines.append(
        f"| {name} | {case['direction']} | {case['profile']} | {case['yaw_abs']:.2f} | "
        f"{case['max_abs_yaw']:.4f} | {case['max_head_step_observed']:.4f} | "
        f"[video](fastavatar_video/{name}.mp4) | `{motion_line}` |"
    )

lines += [
    "",
    "## Grids",
    "",
]
for profile in seen_profiles:
    for direction in sorted({case["direction"] for case in cases}):
        lines.append(f"- `{direction}` / `{profile}`: `grids/{direction}_{profile}_grid.png`")

lines += [
    "",
    "## Visual Review Checklist",
    "",
    "- For each grid, rows follow yaw values from low to high; columns are start/mid/end frames.",
    "- Mark the first yaw where turn direction is clearly visible.",
    "- Mark the largest yaw before visible artifacts become unacceptable.",
    "- Compare turn_right vs turn_left at the same yaw/profile for symmetry.",
    "",
    "## Manual Answers",
    "",
    "- turn_right minimum visible yaw: TODO after inspecting grids.",
    "- turn_right maximum yaw without obvious artifact: TODO after inspecting grids.",
    "- turn_left symmetry: TODO after inspecting grids.",
    "- recommended `max_abs_yaw`: TODO after inspecting grids.",
    "- recommended `max_head_step`: TODO after inspecting grids.",
    "- recommended `head_smooth_window`: TODO after inspecting grids.",
    "- recommended turn scale: TODO after inspecting grids.",
]
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[TurnYawSweep] report={report}")
PY

printf '[TurnYawSweep] all_grid=%s\n' "${ALL_GRID}"
printf '[TurnYawSweep] report=%s\n' "${REPORT}"
