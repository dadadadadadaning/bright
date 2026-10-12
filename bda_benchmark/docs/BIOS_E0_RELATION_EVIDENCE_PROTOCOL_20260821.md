# BIOS E0：建筑内部关系证据审计协议

## 1. 本阶段只回答一个问题

在不训练、不修改 P3/H6-A2、也不访问测试集的条件下，检查：

> 建筑内部 RGB 与 SAR 的结构关系异常，能否识别 P3 漏判为 Background/Intact 的真实 Damaged 建筑？

E0 通过后才允许实现 E1 的可学习 readout。E0 不直接提高 mIoU，也不包含 E1–E4 的训练模块。

## 2. 为什么不用 RGB 与 SAR 特征直接相减

RGB 编码反射光，SAR 编码微波后向散射。两个非共享编码器同一通道不具有严格物理对应关系。E0 分别在模态内部计算自相似矩阵：

\[
A_R(i,j)=\cos(r_i,r_j),\qquad A_S(i,j)=\cos(s_i,s_j)
\]

再计算无标签关系异常：

\[
E(i,j)=|A_R(i,j)-A_S(i,j)|
\]

这比较的是“建筑内部组织是否变化”，而不是强迫 RGB 与 SAR 的数值相等。

## 3. 固定输入与 ROI

- P3：冻结，产生建筑定位和基准损毁预测。
- H6-A2：冻结，只读取 RGB/SAR Stage3 与 Stage4 特征。
- Feature Backbone Lock：沿用 Q0-B0 所登记的 H6-A2 来源、SAR 输入协议和 checkpoint 指纹，但绝不读取 Q0 分类概率。
- GT ROI：验证标签的建筑连通区域，只判断信号上限。这里不再使用“Oracle”一词，避免与 P3+A2 correctness Oracle 混淆。
- P3 ROI：冻结 P3 定位的连通区域，代表实际推理条件。
- 每个 ROI 使用 `ROIAlign` 转成 `7×7` token；ROI 外 token 不参与关系统计。

GT ROI 不能用于正式推理。当前 P3 ROI 仍是语义连通区域，不等同官方单栋实例。P3 概率只用于生成 ROI 和定义 gate-miss 子集，绝不进入结构分数；Q0 概率完全不进入 E0。

## 4. 预注册主假设

主假设在运行前固定，验证标签不能选择信号：

```text
ROI       = 7×7
Feature   = Stage3
Relation  = 局部3×3模态内cosine关系差
Statistic = Mean
Metric    = P3 gate-miss AUROC
```

Stage3-Std 是预注册异质性诊断。Stage4、Stage3+4、Global、Local+Global、Similarity+Dissimilarity、Q90、面积和连通集中度均属于次要诊断，不能替代主信号决定是否进入 E1。

机器可读预注册文件为 `configs/bios_e0_preregistered.json`。运行时会验证配置内容，并把文件 SHA-256 写入结果，防止实验后更换主信号或门槛。

直接跨模态余弦差作为匹配基线，不参与关系表示定义。

## 5. 任务与压力测试

每个信号报告：

- Intact vs Damaged AUROC/AUPRC；
- Damaged vs Destroyed AUROC/AUPRC；
- Affected vs Intact AUROC/AUPRC；
- P3 预测为 Background/Intact 时，识别真实 Damaged 的 AUROC/AUPRC；
- 按事件 AUROC、事件宏平均、最差事件、方向一致事件数；
- 分别统计事件 AUROC `>0.50`、`>0.55`、`>0.60` 的数量；方向正确严格定义为 `AUROC>0.50`；
- Intact→Damaged→Destroyed 的 Spearman 有序相关。

压力测试只对上述预注册主信号执行：

- SAR token 平移 ±1、±2、±3 格，采用零填充而不是循环移位；
- SAR token 空间打乱；
- 同时比较关系异常与直接特征差的 AUROC 下降。

## 6. E1 准入门槛

GT ROI 与 P3 ROI 都必须通过各自门槛：

- P3 非受损漏检 AUROC ≥ 0.60；
- 事件宏 AUROC ≥ 0.58；
- 至少 10/14 个可比事件 AUROC ≥ 0.50；
- 未位移、±1 和 ±2 三种设置中，关系异常 AUROC 都高于直接特征差；
- ±1 和 ±2 token 偏移时，关系异常 AUROC 降幅均小于直接特征差。
- 未打乱关系 AUROC 高于空间打乱对照；

同时要求：

```text
|GT gate-miss AUROC - P3 gate-miss AUROC| ≤ 0.10
```

决策：

- GT/P3 ROI 均通过且差距不过大：`PROCEED_TO_E1_CONTROLLED_READOUT`；
- 仅 GT ROI 通过，或 GT/P3 断崖明显：`STOP_PRACTICAL_BIOS_OR_REDESIGN_PROPOSALS`；
- 两者均失败：`STOP_BIOS_RELATION_ROUTE`；
- Smoke：只检查代码，不作科学决策。

## 7. 服务器执行

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_BIOS_E0_5090.sh preflight
bash run/run_BIOS_E0_5090.sh smoke

LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/BIOS
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/BIOS_E0_relation_val.log"

nohup bash run/run_BIOS_E0_5090.sh val > "$LOG" 2>&1 &
tail -F "$LOG"
```

默认输出：

```text
/root/shared-nvme/BRIGHT-eval-results/BIOS/
BIOS_E0_relation_evidence_val_seed3407.json
```

脚本没有 `eval_test` 能力；测试集不能用于本阶段选择。

## 8. E0如何决定未来建模形式

E0只给条件性建议，不自动授权E1：

- 若主分数中位数严格满足 `Intact < Damaged < Destroyed`，记录为有序强度候选；
- 若 Stage3-Std 中 Damaged 高于 Intact 和 Destroyed，记录为“强度+异质性”二维候选；
- 若只有 gate-miss 子集有效而整体有序性弱，未来只能定位为“漏检中度损毁补充证据”；
- 所有准入门槛未同时通过时，无论次要统计多漂亮，都停止 BIOS。
