# P5 Text-Driven FastAvatar MVP Report

## 1. 项目目标

本项目当前 MVP 的核心目标是建立一条可复现、可解释、可落地的视频生成链路：

**text prompt -> primitive parsing -> latent motion composition -> FastAvatar render**。

具体来说：
- 用户输入自然语言动作描述（如“turn left and smile”）；
- 系统将文本解析为 primitive 语义控制信号；
- 在已训练 VAE 的 latent 空间中完成 primitive 方向组合；
- 解码回 motion 并转换为 FastAvatar 可直接推理的动作目录；
- 最终输出渲染视频。

---

## 2. Pipeline 总览

当前 P5 MVP 的全链路如下：

1. **Rule-based text parser**
   - 使用规则词典将文本映射到 primitive：`turn_left / turn_right / nod / smile / mouth_open / neutral`。

2. **Primitive weights + intensity multipliers**
   - 使用基础权重（primitive 默认权重）与强度词倍率（如 `slight`/`very`/`strong`）共同决定 latent 位移幅度。
   - 支持通过 `motion_model/primitive_presets.json` 做可配置标定。

3. **VAE latent prototype composition**
   - 使用 prototype 均值 latent：
     \[
     z = z_{neutral} + \sum_i \alpha_i (z_{primitive_i} - z_{neutral})
     \]
   - 支持噪声注入与整体缩放。

4. **Temporal hold mode**
   - 使用 ramp/hold/release 的时序整形，把“动作意图”从 latent 解码后的原序列重排为更可控的展示曲线。

5. **`render_motion_npz.py` FastAvatar pack**
   - 将 `motion npz` 写回 FastAvatar 所需目录结构，支持 pack 模式与 root 兼容路径。

6. **一键 Demo: `scripts/demo/text_rule_to_video.sh`**
   - 串联：text-rule 生成 -> pack -> FastAvatar 推理 -> 视频拷贝归档。

7. **结果汇总: `scripts/demo/summarize_text_demo_runs.py`**
   - 自动统计每个 run 的 yaw/head/expr/jaw 指标并输出 markdown 报告。

---

## 3. 当前里程碑

- **P0 / P1：Retrieval baseline**
  - 完成 MMHead codebook、关键词检索、通道拆分检索与 FastAvatar retarget baseline。

- **P2：`motion_dataset_v1`**
  - 完成标准化数据导出、`norm_stats`、可渲染回写桥接。

- **P3：Motion AE / VAE**
  - 完成时序卷积 AE/VAE 训练、重建、采样、渲染可用性闭环。

- **P4：Primitive prototype composition**
  - 完成 primitive 标签、prototype 聚合、prototype 直接生成与组合生成。

- **P5：Text-driven MVP**
  - 完成 rule-based text->primitive->latent composition->video 的端到端一键链路。

- **P5.4：Calibrated preset**
  - 完成 primitive 权重/强度词倍率的 JSON 外部配置化。

- **P5.5：Demo summary**
  - 完成批量 run 统计与 markdown 汇总工具。

---

## 4. 当前 Demo Prompts

已覆盖/建议演示 prompt：

- `turn left`
- `turn right`
- `smile`
- `open mouth`
- `turn left and smile`
- `slightly turn left and smile`
- `strongly turn left and smile`

---

## 5. 当前结果总结

现阶段观测结果（MVP 阶段）：

- `turn_right`：yaw 已校准到约 **-0.10** 量级（负向右转）。
- `turn_left_smile`：yaw 已达约 **+0.09** 量级（正向左转）。
- `smile`：可见但强度中等，尚有增强空间。

---

## 6. 当前限制

当前 MVP 的主要限制：

1. **Rule-based parser**
   - 文本解析依赖规则和关键词，泛化能力有限。

2. **Primitive vocabulary 有限**
   - 当前仅覆盖少量 primitive，语义覆盖不足。

3. **Smile prototype 强度仍偏弱**
   - 视觉笑容可见但稳定性/强度尚未最优。

4. **表达维度映射简化**
   - 50D MMHead expression 仅映射到 FastAvatar expression 前 50 维。

5. **尚无 learned text-to-motion 模型**
   - 当前仍非真正端到端文本生成模型（无 text encoder + latent predictor 训练闭环）。

---

## 7. 下一步建议

1. **Demo packaging**
   - 整理一键运行、日志、可视化与结果目录规范，形成可发布 demo 包。

2. **Refined smile prototype**
   - 基于筛选样本与统计再标定，增强 smile 的显著性与稳定性。

3. **Parser expansion**
   - 扩展 primitive 词表与同义词、动作词、情绪词覆盖。

4. **Temporal language support**
   - 支持“先…再…/ gradually / then / quickly”等时序语义解析。

5. **Ablation experiments**
   - 系统化对比：权重配置、hold 参数、prototype 统计策略、噪声注入对可控性与质量的影响。

---

## 8. 结论

P5 阶段已实现可运行、可解释的 text-driven FastAvatar MVP：
- 从文本到视频的工程链路打通；
- primitive 级控制与时序整形成果稳定；
- 具备继续向 learned text-to-motion（而非规则驱动）演进的基础。
