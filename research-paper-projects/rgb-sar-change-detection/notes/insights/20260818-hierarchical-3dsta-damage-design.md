# 层级式 3D-STA 建筑损毁评估设计

- 用户研究方向：灾前 RGB + 灾后单通道 SAR 建筑损毁分级
- 研究问题：让 HeteCD 3D-STA 负责 Intact/Affected，再在 Affected 条件下判断 Damaged/Destroyed
- 分析日期：2026-08-18

## 核心设计

不采用硬级联，而采用可微的层级概率分解：

```text
RGB → P3建筑定位 → p_building

RGB/SAR stage4 → Lite-3D-STA → p_affected

A2损毁特征 → 条件严重度头 → p_destroyed_given_affected
```

最终四类概率：

```text
P(background) = 1 - p_building
P(intact) = p_building × (1-p_affected)
P(damaged) = p_building × p_affected × (1-p_destroyed_given_affected)
P(destroyed) = p_building × p_affected × p_destroyed_given_affected
```

## 与 A2 的精确安全初始化

设 A2 原有三类建筑损毁 logits 为 `z_i,z_d,z_x`：

```text
z_aff_base = logsumexp(z_d,z_x) - z_i
z_sev_base = z_x - z_d
```

则：

```text
sigmoid(z_aff_base) = P_A2(damaged)+P_A2(destroyed)
sigmoid(z_sev_base) = P_A2(destroyed | affected)
```

因此新模型可以写成：

```text
z_aff = z_aff_base + gamma_aff × delta_3d
z_sev = z_sev_base + gamma_sev × delta_severity
```

并令 `gamma_aff=gamma_sev=0`，从而 step0 与 A2 数学上完全等价，新增模块仍能接收梯度。

## 模块职责

### Lite-3D-STA

- 只用于 stage4；
- 输入 RGB/SAR 投影后的低通道特征；
- 只输出一个 affected residual；
- 建筑掩膜外强制为零；
- 不直接控制 Damaged/Destroyed 边界。

### 严重度条件头

- 仅学习 Damaged vs Destroyed；
- 初始值来自 A2 原始 `z_x-z_d`；
- 后续 residual 主要读取 SAR stage4 结构特征和 A2 common/unique 特征；
- 不再要求其同时学习 Intact 分离。

## 分阶段训练

1. Phase A：冻结 P3/A2，只训练 `delta_3d` 和 `gamma_aff`，目标为 Affected vs Intact。
2. Phase B：冻结发生层，只训练 severity residual，目标为 GT affected 中的 Damaged vs Destroyed。
3. Phase C：可选低学习率联合微调，只使用最终层级负对数似然，不增加 FCA、对比或原型损失。

## 受控消融

| 实验 | 变化 | 目的 |
|---|---|---|
| H0 | 原始 A2 四分类 | 锁定基线 |
| H1 | 只做层级重参数化，residual=0 | 验证概率和指标精确等价 |
| H2 | affected 使用普通 3×3 差异残差 | 控制参数增加影响 |
| H3 | affected 使用 Lite-3D-STA | 验证 3D-STA 是否优于普通差异 |
| H4 | H3 + severity residual | 验证条件严重度学习是否改善 Damaged/Destroyed |

## 必须统计的诊断

- Affected F1/IoU；
- GT Damaged 被 occurrence gate 漏掉的比例；
- 在 GT affected 内 Damaged/Destroyed 条件 F1；
- Damaged→Intact、Damaged→Destroyed；
- Destroyed→Damaged、Destroyed→Intact；
- Final mIoU、Event mean/worst、逐事件正收益数量。

## 风险

- 如果 affected gate 主要学习强 SAR 破坏证据，仍可能漏掉轻中度 Damaged；必须单独报告 GT Damaged gate recall。
- 3D-STA 原论文使用三极化 SAR 和二值建设/拆除数据，单通道灾害迁移收益未被原文证实。
- HeteCD 四尺度完整模块计算大且容易过拟合；这里只允许 stage4 轻量版本进入首轮实验。
- 只有 H3 明显优于参数匹配的 H2，才能把 3D 时空差异作为有效方法贡献。

## 建议题目

1. **Hierarchical Change-Conditioned RGB-SAR Network for Building Damage Assessment**
2. **Safe 3D Cross-Modal Damage Occurrence Modeling with Conditional Severity Estimation**
3. **From Change Presence to Damage Severity: Hierarchical Optical-SAR Building Assessment**

## 代码落地（2026-08-18）

设计已实现为 `H10-H3D`：

- 模型：`bda_benchmark/model/DamageFormer_RGBSAR_H10_H3D.py`
- 单一层级 NLL 与错误分解：`bda_benchmark/script/standard_ML/h10_h3d_runtime.py`
- 训练：`bda_benchmark/script/standard_ML/train_H10_H3D.py`
- 独立验证/测试：`bda_benchmark/script/standard_ML/eval_H10_H3D.py`
- 运行入口：`bda_benchmark/run/run_H10_H3D_5090.sh`
- H2/H3 受控序列：`bda_benchmark/run/run_H10_H3D_sequence.sh`
- 测试：`bda_benchmark/tests/test_h10_h3d.py`
- 中文实验说明：`bda_benchmark/docs/H10_H3D_层级3DSTA实验说明.md`

工程实现额外加入了 step-0 安全权重参与验证选择的机制。它在共享 H4
训练器中默认关闭，只由 H10 开启，因此不会改变历史 H4/H5/H6 实验行为。
