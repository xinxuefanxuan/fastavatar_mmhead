# text_motion

独立文本到动作模块（不改 FastAvatar 核心模型），用于把受控文本提示映射到“可被现有 FastAvatar 推理脚本读取”的 motion 文件夹。

## 1) 先检查 motion 格式

```bash
python text_motion/inspect_motion_format.py \
  --motion_dir assets/sample_motion/nersemble_seq_214
```

输出包括：

- 目录树（深度 3）
- 首帧 npz 信息（key/shape/dtype/min/max/mean）
- transforms.json 顶层与 frame schema
- 帧数估计
- 与 camera/flame/head-pose 相关字段扫描

> 该步骤是必须的：避免假设字段名与语义。

## 2) 复制模板 motion（安全）

```bash
python text_motion/copy_motion_template.py \
  --template_motion assets/sample_motion/nersemble_seq_214 \
  --output_motion assets/sample_motion/text_turn_left
```

- 永不覆盖模板目录
- 若输出目录已存在则报错退出
- 支持 `--dry_run`

## 3) 文本生成动作

```bash
python text_motion/run_text_to_motion.py \
  --prompt "turn head left" \
  --template_motion assets/sample_motion/nersemble_seq_214 \
  --output_motion assets/sample_motion/text_turn_left \
  --num_frames 90
```

```bash
python text_motion/run_text_to_motion.py \
  --prompt "smile slightly" \
  --template_motion assets/sample_motion/nersemble_seq_214 \
  --output_motion assets/sample_motion/text_smile \
  --num_frames 90 \
  --primitive_config text_motion/primitives.yaml
```

支持 `--dry_run` 仅打印将要修改的字段。

## 4) 让 FastAvatar 使用生成结果

保持你现有推理命令不变，只需把 `motion_seqs_dir` 指向新输出目录中的一个 flame npz 文件路径（当前项目里 `prepare_motion_seqs` 通过其父目录读取 `transforms.json` / `flame_param/`）。

## 5) 限制与注意事项

- 不改 FastAvatar 模型/Transformer/训练流程。
- 不直接把文本 embedding 注入点云 prompt 或 3D prompt。
- FLAME expression 维度没有语义标签时，不应硬编码“某 index=smile”。
- 表情原语优先用 exemplar-delta：
  - `target_expr = neutral_expr + alpha * expression_delta`
- `blink once` 提供的是保底曲线注入策略，具体通道需按你数据校准。

## 6) 文件说明

- `inspect_motion_format.py`：格式侦测
- `copy_motion_template.py`：模板安全复制
- `motion_primitives.py`：时序曲线 + 原语实现
- `run_text_to_motion.py`：主入口
- `primitives.yaml`：受控 prompt 映射与参数

## 7) 检索式 Text-to-MMHead 代码本（第一版）

第一版目标：给定文本 prompt，从 MMHead 样本中做关键词检索，选出 top-k（默认取 top-1），再复用 `run_mmhead_to_motion.py` 完成 FastAvatar 可读 motion 序列生成。

### 数据假设

MMHead 根目录下可包含：

- `t2m_manifest.jsonl`
- `facial_motion/{sample_id}.pkl`（主格式）
- `facial_motion/{sample_id}.npz`（可选 fallback）
- `text_annotations/action/{sample_id}.txt`
- `text_annotations/detail_expression/{sample_id}.txt`
- `text_annotations/detail_head_pose/{sample_id}.txt`
- `text_annotations/emotion/{sample_id}.txt`
- `text_annotations/emotion_scenario/{sample_id}.txt`

### 7.1 构建 codebook

```bash
python text_motion/mmhead_codebook.py \
  --mmhead_root /path/to/MMHead \
  --manifest /path/to/MMHead/t2m_manifest.jsonl \
  --output_jsonl outputs/codebook.jsonl
```

可选调试：

```bash
python text_motion/mmhead_codebook.py \
  --mmhead_root /path/to/MMHead \
  --manifest /path/to/MMHead/t2m_manifest.jsonl \
  --output_jsonl outputs/codebook_debug.jsonl \
  --max_samples 20 \
  --verbose
```

### 7.2 检查 codebook

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/codebook.jsonl')
for i,l in enumerate(p.open('r',encoding='utf-8')):
    if i>=3: break
    print(json.loads(l))
PY
```

### 7.3 关键词检索 top-k

```bash
python text_motion/mmhead_retrieval.py \
  --prompt "turn head left and smile" \
  --codebook_jsonl outputs/codebook_debug.jsonl \
  --top_k 5 \
  --output_jsonl outputs/retrieval_debug.jsonl
```

### 7.4 文本到 FastAvatar motion（检索+转换）

```bash
python text_motion/run_text_to_mmhead_motion.py \
  --prompt "turn head left and smile" \
  --codebook_jsonl outputs/codebook_debug.jsonl \
  --template_motion assets/sample_motion/nersemble_seq_214 \
  --output_motion assets/sample_motion/text_retrieved_motion_debug \
  --top_k 5 \
  --rank_index 0 \
  --dry_run
```

支持保存检索结果与元数据：

```bash
python text_motion/run_text_to_mmhead_motion.py \
  --prompt "turn head left and smile" \
  --codebook_jsonl outputs/codebook.jsonl \
  --template_motion assets/sample_motion/nersemble_seq_214 \
  --output_motion assets/sample_motion/text_retrieved_motion \
  --top_k 10 \
  --rank_index 0 \
  --save_retrieval_jsonl assets/sample_motion/text_retrieved_motion/retrieval_topk.jsonl \
  --save_metadata_json assets/sample_motion/text_retrieved_motion/retrieval_meta.json
```

### 7.5 用 infer.sh 渲染

检索转换完成后，沿用你已有推理入口：

```bash
bash scripts/infer/infer.sh \
  configs/inference/infer.yaml \
  model_zoo/fastavatar/ \
  assets/sample_input/mono_video/nersemble_seq_214.mp4 \
  assets/sample_motion/text_retrieved_motion/
```

### 7.6 输出验证建议

- `codebook.jsonl` 每行一个样本，包含 `searchable_text`、`annotations`、`motion_stats`。
- `retrieval_topk.jsonl` 包含排序、匹配词、分数分解。
- `retrieval_meta.json` 记录 prompt、选中样本、命令行参数和时间戳。
- `output_motion` 目录结构应保持与模板兼容（含 `transforms.json` 与 `flame_param/*.npz`）。

### 7.7 已知限制（第一版）

- 仅关键词检索（无 embedding、无 FAISS）。
- 无聚类/码本压缩（每个样本即一个 codebook entry）。
- 运动方向统计是粗粒度（基于 delta norm）。
- 文本匹配不保证视觉上完全一致，仅提供可解释、可复现的第一版检索基线。

### 7.8 头部抖动稳定化建议（MMHead native posecodes）

如果检索到的样本在 `head_pose` 上出现抖动，可在转换时启用平滑与速度钳制：

```bash
python text_motion/run_text_to_mmhead_motion.py \
  --prompt "turn head left and smile" \
  --codebook_jsonl outputs/codebook.jsonl \
  --template_motion assets/sample_motion/nersemble_seq_214 \
  --output_motion assets/sample_motion/text_retrieved_motion_stable \
  --top_k 10 \
  --rank_index 0 \
  --head_smooth_window 9 \
  --head_max_step 0.03 \
  --head_scale 0.05
```

说明：
- `*_smooth_window > 1` 时，按时间维做中心滑动平均（边界用 edge padding）。
- `head_max_step > 0` 时，会对帧间头部步长做范数钳制并重建轨迹。
- 默认参数（window=1, max_step=0）保持旧行为不变。

## 8) 构建中性静态模板（neutral template）

当你希望“文本检索动作”不叠加原模板动态，而是从静态中性状态出发时，可先构建 neutral 模板：

```bash
python text_motion/make_neutral_template.py \
  --template_motion assets/sample_motion/nersemble_seq_214 \
  --output_motion assets/sample_motion/nersemble_seq_214_neutral \
  --reference_frame 0
```

或自动选择最中性帧：

```bash
python text_motion/make_neutral_template.py \
  --template_motion assets/sample_motion/nersemble_seq_214 \
  --output_motion assets/sample_motion/nersemble_seq_214_neutral \
  --auto_neutral
```

自动中性评分（使用可用键）：

- `score = ||expr||_2 + ||jaw_pose||_2 + 0.3 * ||rotation||_2`

脚本行为：
- 先完整复制模板目录到输出目录；
- 遍历 `flame_param/*.npz`；
- 将每帧 `expr/jaw_pose/rotation/translation/eyes_pose/shape`（若存在）替换为参考帧值；
- 保留 `canonical_flame_param.npz`、`transforms*.json`、`processed_data/` 以及其他文件不变。

随后可将 neutral 模板用于检索驱动：

```bash
python text_motion/run_text_to_mmhead_motion.py \
  --prompt "turn head left and smile" \
  --codebook_jsonl "$OUT_DIR/codebook_full.jsonl" \
  --template_motion assets/sample_motion/nersemble_seq_214_neutral \
  --output_motion assets/sample_motion/text_neutral_turn_left_smile \
  --top_k 10 \
  --rank_index 0 \
  --mode all \
  --num_frames 90 \
  --expr_scale 0.12 \
  --head_scale 0.05 \
  --jaw_scale 0.2 \
  --head_smooth_window 9 \
  --jaw_smooth_window 3 \
  --head_max_step 0.03
```

## 9) Channel-disentangled retrieval and composition

动机：holistic 检索会把 head/expression/jaw 耦合迁移，容易出现语义混合。该流程按通道检索并在 neutral 模板上组合，控制更干净。

### 9.1 刷新 codebook

```bash
python text_motion/mmhead_codebook.py \
  --mmhead_root /path/to/MMHead \
  --manifest /path/to/MMHead/t2m_manifest.jsonl \
  --output_jsonl "$OUT_DIR/codebook_full.jsonl"
```

### 9.2 先构建 neutral 模板

```bash
python text_motion/make_neutral_template.py \
  --template_motion assets/sample_motion/nersemble_seq_214 \
  --output_motion assets/sample_motion/nersemble_seq_214_neutral \
  --auto_neutral
```

### 9.3 通道组合生成动作

```bash
python text_motion/run_text_to_composed_motion.py \
  --prompt "turn head left and smile" \
  --codebook_jsonl "$OUT_DIR/codebook_full.jsonl" \
  --template_motion assets/sample_motion/nersemble_seq_214_neutral \
  --output_motion assets/sample_motion/text_composed_turn_left_smile \
  --top_k 10 \
  --head_scale 0.15 \
  --expr_scale 0.12 \
  --jaw_scale 0.2 \
  --head_smooth_window 9 \
  --head_max_step 0.03 \
  --head_target_field neck_pose
```

### 9.4 推理

```bash
bash scripts/infer/infer.sh \
  configs/inference/infer.yaml \
  model_zoo/fastavatar/ \
  assets/sample_input/mono_video/nersemble_seq_214.mp4 \
  assets/sample_motion/text_composed_turn_left_smile/ \
  16 \
  16 \
  Monocular \
  false
```

注意：传给 `infer.sh` 的 motion 目录建议带 trailing slash（`.../`）。

## Direction calibration and head-axis controls

When composed head motion direction looks inverted (e.g. prompt asks "left" but render looks "right"), use head axis controls in composed generation:

```bash
python text_motion/run_text_to_composed_motion.py \
  --prompt "turn head left and smile" \
  --codebook_jsonl "$OUT_DIR/codebook_full.jsonl" \
  --template_motion assets/sample_motion/nersemble_seq_214_neutral \
  --output_motion assets/sample_motion/text_composed_turn_left_smile \
  --top_k 10 \
  --head_target_field neck_pose \
  --head_axis_order 0,1,2 \
  --head_axis_signs 1,-1,1
```

`--head_axis_order` reorders MMHead head channels before writing to FastAvatar head target field, and `--head_axis_signs` applies per-axis sign flips.

To manually calibrate which `neck_pose` axis/sign corresponds to viewer-left/viewer-right, generate six calibration motions:

```bash
python text_motion/calibrate_neck_axes.py \
  --template_motion assets/sample_motion/nersemble_seq_214_neutral \
  --output_root assets/sample_motion/neck_axis_calib \
  --num_frames 16 \
  --amplitude 0.2
```

This creates:
- `axis0_pos`, `axis0_neg`
- `axis1_pos`, `axis1_neg`
- `axis2_pos`, `axis2_neg`

Render each folder and map axis/sign to your desired visual direction.

## P2.1: 构建训练用规范化运动数据集

用于下一阶段 motion AE/VAE 训练的数据导出（不包含模型训练）。

### 构建数据集

```bash
python text_motion/build_motion_dataset.py \
  --codebook_jsonl "$OUT_DIR/codebook_full.jsonl" \
  --output_root outputs/motion_dataset_v1 \
  --target_len 90 \
  --min_frames 32 \
  --ref_n 5 \
  --head_axis_signs 1,-1,1
```

脚本会：
- 读取 codebook；
- 按 native MMHead pkl 提取 `expcodes/posecodes`；
- 构建 `expr/head/jaw` 的 delta 序列；
- 对 `head` 应用 `--head_axis_signs`；
- 统一到 `target_len`；
- 输出 `motions/*.npz` + `manifest.jsonl/train.jsonl/val.jsonl`。

### 检查数据集

```bash
python text_motion/inspect_motion_dataset.py \
  --dataset_root outputs/motion_dataset_v1
```

输出包括：样本数、train/val 数、motion shape、expr/head/jaw 的 norm 统计与示例条目。

## P2.2: 将导出的 motion npz 回写为 FastAvatar 动作目录

将 `build_motion_dataset.py` 导出的单个样本（如 `[T,56]`）写回 neutral 模板，生成可直接用于 FastAvatar 推理的 motion 目录。

```bash
python text_motion/render_motion_npz.py \
  --motion_npz outputs/motion_dataset_v1/motions/EXAMPLE_ID.npz \
  --neutral_template assets/sample_motion/nersemble_seq_214_neutral \
  --output_motion_root outputs/mmhead_debug/render_npz_test_fixed \
  --sequence_name EXAMPLE_ID \
  --motion_key motion \
  --head_target neck_pose \
  --smooth \
  --head_velocity_clamp 0.03 \
  --overwrite
```

默认通道布局：
- `motion[:, 0:50] -> expr_delta`
- `motion[:, 50:53] -> head_delta`
- `motion[:, 53:56] -> jaw_delta`

并分别写入：
- `expr_delta -> expr`
- `head_delta -> neck_pose`（可切换到 `rotation`）
- `jaw_delta -> jaw_pose`

推荐用于 FastAvatar 的目录布局（`--output_motion_root`）：

```bash
python text_motion/render_motion_npz.py \
  --motion_npz outputs/mmhead_debug/motion_dataset_v1_debug/motions/CELEBVHQ_01ClRWyf9I4_0.npz \
  --neutral_template assets/sample_motion/nersemble_seq_214_neutral \
  --output_motion_root outputs/mmhead_debug/render_npz_test_fixed \
  --sequence_name CELEBVHQ_01ClRWyf9I4_0 \
  --motion_key motion \
  --head_target neck_pose \
  --overwrite
```

## P3.1 Motion Autoencoder（非VAE）

先确保 P2.1 数据集已包含：
- `motion_raw`（未归一化 56 维）
- `motion_norm`（按 `norm_stats.json` 逐维归一化）

### 训练 AE

```bash
python motion_model/train_motion_ae.py \
  --dataset_root outputs/motion_dataset_v1 \
  --output_dir outputs/motion_ae_v1 \
  --latent_dim 64 \
  --epochs 30 \
  --batch_size 64
```

损失为加权重建：
- expr: 1
- head: 10
- jaw: 10

### 重建单个样本

```bash
python motion_model/reconstruct_motion_ae.py \
  --input_npz outputs/motion_dataset_v1/motions/EXAMPLE_ID.npz \
  --checkpoint outputs/motion_ae_v1/best.pt \
  --norm_stats outputs/motion_dataset_v1/norm_stats.json \
  --output_npz outputs/motion_ae_v1/recon_EXAMPLE_ID.npz
```

输出 `recon_*.npz` 包含 `motion`/`motion_raw`/`motion_norm` 与 `expr_delta/head_delta/jaw_delta`，可直接配合 `text_motion/render_motion_npz.py` 渲染。

## P3.2 Motion VAE（最小增量）

在 P3.1 AE 基础上增加 `mu/logvar` 与重参数化，训练目标：

- `L = recon_loss + beta * KL`
- `beta` 默认 `1e-4`
- 默认启用 KL warmup（前 20 个 epoch 线性升温）
- recon_loss 仍使用通道加权：expr=1, head=10, jaw=10

### 训练 VAE

```bash
python motion_model/train_motion_vae.py \
  --dataset_root outputs/motion_dataset_v1 \
  --output_dir outputs/motion_vae_v1 \
  --latent_dim 64 \
  --epochs 30 \
  --beta 1e-4 \
  --kl_warmup_epochs 20
```

输出：`best.pt`、`last.pt`、`train_log.json`。

### 采样新动作（渲染兼容 npz）

```bash
python motion_model/sample_motion_vae.py \
  --checkpoint outputs/motion_vae_v1/best.pt \
  --norm_stats outputs/motion_dataset_v1/norm_stats.json \
  --output_npz outputs/motion_vae_v1/sample_000.npz \
  --num_frames 64
```

采样输出字段与 `render_motion_npz.py` 兼容：
- `motion`
- `motion_raw`
- `motion_norm`
- `expr_delta`
- `head_delta`
- `jaw_delta`

### VAE 重建单个样本（deterministic, use mu）

```bash
python motion_model/reconstruct_motion_vae.py \
  --input_npz outputs/motion_dataset_v1/motions/EXAMPLE_ID.npz \
  --checkpoint outputs/motion_vae_v1/best.pt \
  --norm_stats outputs/motion_dataset_v1/norm_stats.json \
  --output_npz outputs/motion_vae_v1/recon_EXAMPLE_ID_from_vae.npz \
  --motion_key motion_norm
```

输出同样与 `render_motion_npz.py` 兼容，并打印原始/重建的 expr/head/jaw 统计。
