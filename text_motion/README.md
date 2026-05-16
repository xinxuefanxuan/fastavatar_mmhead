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
