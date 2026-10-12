# S2 Evidence Ladder V1：八阶段隔离实验

## 总原则

这不是一个从 S1 一路累加到 S8 的大模型。S4–S7 都从已经锁定的 S3 基础 Utility
Gate 独立分叉，只有各分支通过内部校准后，S8 才组合其中至少两个。

```text
S1 有害修改审计 → S2 方向规则 → S3 基础Utility Gate
                                      ├─ S4 Memory verifier
                                      ├─ S5a IM4CD局部匹配
                                      ├─ S5b 模态内关系差
                                      ├─ S6 Selective RGID
                                      └─ S7 RGB软超像素 vs Grid
                                                ↓
                                至少两个独立通过 → S8联合
```

所有阶段使用同一个 Train2918 事件分层划分。模块选择只看291张内部校准；Val450只在
方法和阈值冻结后访问一次。Test与官方实例标签始终禁止。

## S1–S3：当前正在运行的基础实验

基础实验完成后，先把产物登记到ladder：

```bash
cd /root/data/BRIGHT-master/bda_benchmark

LADDER=/root/data/BRIGHT-eval-results/SPI_S2_EVIDENCE_LADDER_V1
UTILITY=/root/data/BRIGHT-eval-results/SPI_S2_UTILITY_GATE_V1
POLICY=/root/data/BRIGHT-checkpoints/SPI_S2_UTILITY_GATE_V1/direction_policy_seed3407.json
P3=/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth
mkdir -p "$LADDER/summaries"

python -m bda_benchmark.experiments.spi_s2_evidence_ladder_v1.summarize_foundation \
  --input "$POLICY" --kind audit --output "$LADDER/summaries/s1.json"
python -m bda_benchmark.experiments.spi_s2_evidence_ladder_v1.record_result \
  --stage-id s1_s2_audit --summary-json "$LADDER/summaries/s1.json" \
  --output "$LADDER/s1_s2_audit.json" --source-artifact "$POLICY" \
  --checkpoint "$P3" --split internal_calibration

python -m bda_benchmark.experiments.spi_s2_evidence_ladder_v1.summarize_foundation \
  --input "$POLICY" --kind direction --output "$LADDER/summaries/s2.json"
python -m bda_benchmark.experiments.spi_s2_evidence_ladder_v1.record_result \
  --stage-id s2_direction_policy --summary-json "$LADDER/summaries/s2.json" \
  --output "$LADDER/s2_direction_policy.json" \
  --parent-result "$LADDER/s1_s2_audit.json" --source-artifact "$POLICY" \
  --checkpoint "$P3" --split internal_calibration

python -m bda_benchmark.experiments.spi_s2_evidence_ladder_v1.summarize_utility_result \
  --input "$UTILITY/internal_calibration_step500_seed3407.json" \
  --output "$LADDER/summaries/s3.json"
python -m bda_benchmark.experiments.spi_s2_evidence_ladder_v1.record_result \
  --stage-id s3_base_utility_gate --summary-json "$LADDER/summaries/s3.json" \
  --output "$LADDER/s3_base_utility_gate.json" \
  --parent-result "$LADDER/s2_direction_policy.json" \
  --source-artifact "$UTILITY/internal_calibration_step500_seed3407.json" \
  --checkpoint /root/data/BRIGHT-checkpoints/SPI_S2_UTILITY_GATE_V1/gate_step500_seed3407/gate_last.pth \
  --split internal_calibration
```

查询允许开展的下一阶段：

```bash
bash experiments/spi_s2_evidence_ladder_v1/run.sh status
```

如果 S3 被拒绝，S4–S8 全部停止；不能用更复杂模块掩盖基础门控没有信号。

## S4：Memory verifier

输入不是Memory类别预测，而是每个S2修改区域的检索可靠性：三类Top-1、三类Top-k均值、
三类可用性和检索margin，共10通道。实现位于 `cross_event_memory_evidence`。

约束：Memory只从gate_train建立；查询时强制排除同事件；三类分别Top-k；检索失败输出
availability=0并回退基础门控。训练时加载锁定S3权重，只训练新的Evidence Gate。

对照：`S3`、`S3+随机Memory分数`、`S3+同事件Memory(仅诊断)`、`S4跨事件Memory`。

## S5：IM4CD与关系差是竞争分支

`im4cd_reliability_evidence`输出8个局部匹配通道；
`relation_difference_evidence`输出4个模态内关系变化通道。两者必须分别训练：

```text
S5a = 锁定S3 + IM4CD证据
S5b = 锁定S3 + Relation证据
```

两者都必须做±1/±2 token偏移测试。不能先拼接两者再报告结果；若都通过，联合属于S8。

## S6：Selective RGID

RGID只替换候选区域：`LocCC parent → 有风险时拆成RGID children → 固定S2`。复用
`spi_rgid_semantic_v1.method.selective_partition`，必须验证拆分前后建筑支持逐像素相同。

对照：LocCC-S2、全部RGID拆分、Selective RGID、等数量随机父区域拆分。只有Selective
RGID超过等数量随机拆分，才能证明收益来自选择策略。

## S7：可微软超像素

复用H13的`RGBGuidedSoftSuperpixels`。只用灾前RGB生成一次分区，RGB/SAR共享。Grid与
RGB软超像素必须使用相同cell size和节点预算。先比较证据，不修改P3 backbone。

对照：`S3+Grid evidence`与`S3+RGB superpixel evidence`。只有后者稳定更好，才允许训练
分区参数；否则停止可微超像素路线。

## S8：多模块联合

`IsolatedEvidenceGate(..., allow_combination=True)`只接受已经登记为ACCEPT的模块名。
联合后必须进行leave-one-module-out：移除任一模块都应使mIoU下降，否则该模块不属于最终模型。

推荐最多组合两个证据组：优先`Memory + (IM4CD或Relation)`。RGID/超像素改变候选分区，
只有它们单独明显超过LocCC后才进入联合。

## 代码接口

- `protocol.json`：八阶段父子关系、必要输出和统一阈值。
- `contracts.py`：指纹、父结果、泄漏与准入验证。
- `evidence.py`：Memory、IM4CD、关系差、Superpixel/Grid证据。
- `memory_bank.py`：只允许Train构建、带P3与split指纹的冻结Memory。
- `partitions.py`：Selective RGID与等数量随机拆分，强制建筑支持守恒。
- `router.py`：单证据Gate与最终多证据Gate；默认禁止堆叠。
- `record_result.py`：把每次内部校准登记为ACCEPT或REJECT。
- `advance.py`：只显示当前协议允许执行的阶段。
