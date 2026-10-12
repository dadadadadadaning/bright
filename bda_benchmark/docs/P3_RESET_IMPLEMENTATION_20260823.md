# P3 Reset：受控 RGB–SAR 实验链

## 1. 研究问题

本实现不再向 P3 同时叠加多个模块，而把问题严格拆成：

1. **A0：模态信息审计**：RGB、P3 pseudo-SAR、RGB+SAR 中谁包含可读的建筑严重度信息？
2. **B0–B2：SAR 表征审计**：P3 pseudo-SAR、H6 NativeSAR、MM-OVSeg SAR-DINO 谁更好？
3. **C0–C2：融合审计**：原 P3 concat、安全差异残差、三专家 MoE-lite 谁更好？
4. **D0–D2：损毁头审计**：四类竞争、建筑条件三类、层级严重度谁更好？

所有训练和选择只使用 train/val；这些入口没有 test 模式。

## 2. A0/B0–B2 的公平协议

- 编码器全部冻结；
- 所有候选只训练 `LayerNorm(no affine) + Linear(C,3)`；
- 建筑区域只由二值 GT building support 生成；
- I/D/X 标签只在区域生成后按多数类赋值，纯度低于 0.75 的区域被排除；
- 不按损毁类别生成连通域，避免把答案泄露到区域边界；
- 统一使用 Macro-F1、Damaged Recall、Event Macro-F1 选择权重。

MM-OVSeg 适配器按本地官方源码执行：ViT-Base、checkpoint 推断 patch size（官方当前实现通常为 patch8）、384 输入、第 4/8/12 block。第一轮 B2 只读取 Block12。第三方源码保持不变。

## 3. C/D 的安全协议

- P3 RGB 定位和 P3 主路径冻结；
- 新残差输出层为全零，因此 C1/C2 初始预测严格等于 P3；
- 小的正 gate 让零输出层在首步仍可获得梯度；
- C1 使用 `[R,S,R-S,|R-S|]`，差异只是输入，不被直接当成 damage score；
- C1S 是 C1 失败后的定向对照：固定 P3 的 Intact–Affected 总概率，只允许 NativeSAR 修改 Damaged–Destroyed 条件比例；
- C2 只有 RGB、SAR、common 三个小专家，不使用 top-k/O2SP；
- D1 显式建模 `P(B) × P(I/D/X|B)`；
- D2 显式建模 `P(B) × P(Affected|B) × P(Destroyed|Affected)`；
- D1/D2 必须加载并冻结已通过门槛的 C1/C2，避免同时改变融合和分类头。

## 4. 执行顺序

服务器同步代码后：

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

export SAR_DINO_CKPT=/实际路径/B16_checkpoint.pth
bash run/run_P3_reset_5090.sh preflight
```

第一阶段只运行表征审计：

```bash
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/P3_RESET
mkdir -p "$LOG_ROOT"

nohup env SAR_DINO_CKPT="$SAR_DINO_CKPT" \
  bash run/run_P3_reset_sequence.sh features \
  > "$LOG_ROOT/feature_audit.log" 2>&1 &

tail -F "$LOG_ROOT/feature_audit.log"
```

结果文件：

```text
/root/shared-nvme/BRIGHT-checkpoints/P3_RESET/
  FEATURE_AUDIT/feature_source_decision.json
```

只有其中 `passed=true` 才运行融合：

```bash
nohup env SAR_DINO_CKPT="$SAR_DINO_CKPT" \
  bash run/run_P3_reset_sequence.sh fusion \
  > "$LOG_ROOT/fusion_audit.log" 2>&1 &
```

融合序列会先运行 C1；C1 门槛失败时脚本会停止，不运行 C2。

C1 若表现为严重度改善但 occurrence 退化，运行 C1S：

```bash
nohup env OPTIMIZER_STEPS=1500 \
  bash run/run_P3_reset_sequence.sh severity \
  > "$LOG_ROOT/c1s_severity_only.log" 2>&1 &
```

只有锁定融合候选后才运行 D1。例如：

```bash
export SAR_SOURCE=b2
export INTERACTION=c1
export FUSION_CKPT=/root/shared-nvme/BRIGHT-checkpoints/P3_RESET/MODEL/C1_B2_3000_seed3407/best_event_worst.pth

nohup bash run/run_P3_reset_sequence.sh heads \
  > "$LOG_ROOT/head_audit.log" 2>&1 &
```

D1 门槛失败时不会运行 D2。

## 5. 停止条件

- A0-RS 不同时优于/不伤害 A0-R 的 Macro-F1、Damaged Recall、Event Macro-F1：停止复杂融合。
- B1/B2 未相对 B0 带来可解释提升：停止外部 SAR 表征替换。
- C1 未通过正式分割门槛：停止 C2。
- D1 不能在定位不变的情况下提升 Damaged/Event：停止 D2。
- 任一阶段不得根据 test 结果选择结构、权重或超参数。

## 6. 文件对应关系

```text
model/p3_reset/
  p3_feature_audit.py
  sar_sources/{p3_pseudo_sar,native_sar,mmov_sar_dino}.py
  fusion/{p3_concat,residual_diff,moe_lite}.py
  heads/{p3_4class,conditional_severity,hierarchical_severity}.py
  P3_Reset_Model.py

script/standard_ML/
  train_P3_reset_feature_audit.py
  summarize_P3_reset_feature_audit.py
  train_P3_reset_model.py

run/
  run_P3_reset_5090.sh
  run_P3_reset_sequence.sh
```
