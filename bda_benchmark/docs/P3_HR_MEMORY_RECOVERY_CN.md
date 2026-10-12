# P3层级残差与Memory失败救援运行说明

## 1. 两条路线必须隔离

P3-HR只验证模型结构：

```text
冻结P3 -> Occurrence/Severity残差头 -> 四类概率重构
```

Memory-M1R只验证失败Memory是否仍有可信子集：

```text
参数分类器为默认 -> 严格跨事件检索 -> 高置信分歧时有限介入
```

两者不在同一次训练中组合，也不访问Test。

## 2. P3-HR单命令安全训练

同步以下文件：

```text
bda_benchmark/model/DamageFormer_P3_HierarchicalResidual.py
bda_benchmark/script/standard_ML/p3_hierarchical_residual_runtime.py
bda_benchmark/script/standard_ML/train_P3_HierarchicalResidual.py
bda_benchmark/run/run_P3_HierarchicalResidual.sh
bda_benchmark/tests/test_p3_hierarchical_residual.py
```

服务器运行：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

nohup env \
SEED=7341 \
P3_CHECKPOINT=/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth \
OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/P3_HIERARCHICAL_RESIDUAL \
NUM_WORKERS=4 \
bash run/run_P3_HierarchicalResidual.sh auto_h1 \
> /root/data/BRIGHT-train-logs/P3_HR0_HR1_seed7341.log 2>&1 &
```

该入口严格按顺序执行：

```text
Preflight -> H0真实数据整网P3守卫 -> 守卫通过 -> H1 500步
```

查看日志：

```bash
tail -f /root/data/BRIGHT-train-logs/P3_HR0_HR1_seed7341.log
```

H0要求真实batch输出逐比特等于P3，并且完整Val指标在P3预期值0.10pp内。
H1只训练深层Occurrence和Severity小头，P3全部冻结；不包含新SAR编码器、
Cross-Attention、DeCUR或Memory。

## 3. Memory M1失败后的唯一允许实验

当前M1结果已经是 `STOP_CROSS_EVENT_METRIC_M1`。运行：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

env \
SEED=7341 \
D1_CACHE_ROOT=/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341 \
OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/SPI_CROSS_EVENT_METRIC_MEMORY_V2/seed7341 \
RESULT_ROOT=/root/data/BRIGHT-eval-results/SPI_CROSS_EVENT_METRIC_MEMORY_V2 \
MEMORY_TOPK=4 \
RETRIEVAL_TEMPERATURE=0.10 \
MINIMUM_MEMORY_PURITY=0.0 \
bash experiments/spi_cross_event_metric_memory_v2/run.sh rescue
```

M1R在meta_train跨事件OOF预测上冻结置信优势和Memory margin，然后只在
meta_calibration评估一次。不能根据calibration结果再次改变阈值。

## 4. 结果解释

```text
STOP_MEMORY_NO_IDENTIFIABLE_SAFE_SUBSET
```

训练集跨事件OOF中都找不到稳定可信子集，停止Memory。

```text
STOP_MEMORY_AFTER_SELECTIVE_RESCUE
```

训练集存在候选规则，但不能泛化到calibration，停止Memory。

```text
ADMIT_SELECTIVE_MEMORY_AS_AUXILIARY_E1
```

Memory只能作为小范围辅助证据进入下一轮LOEO，不能覆盖参数分类器。

无论M1R结果如何，都不能把它直接接到H1。本轮目的就是分别确认：层级模型结构
是否有效，以及失败Memory是否还存在可靠的局部使用空间。

## 5. 夜间连续队列

HR1失败后允许运行的正交消融为：

```text
HR1-O  深层融合特征，只训练Occurrence
HR1-S  深层融合特征，只训练Severity
HR2    Stage2+3关系Occurrence + 深层Severity
HR3-O  Stage2+3关系Occurrence + 区域保护损失，不训练Severity
M1R    失败Memory的选择性高置信救援
```

它们不构成学习率搜索，每一项只改变一个结构或监督因素。使用：

```bash
nohup env \
SEED=7341 \
P3_CHECKPOINT=/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth \
OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/P3_HIERARCHICAL_RESIDUAL \
MEMORY_OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/SPI_CROSS_EVENT_METRIC_MEMORY_V2/seed7341 \
MEMORY_RESULT_ROOT=/root/data/BRIGHT-eval-results/SPI_CROSS_EVENT_METRIC_MEMORY_V2 \
D1_CACHE_ROOT=/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341 \
LOG_ROOT=/root/data/BRIGHT-train-logs/P3_HR_NIGHT_seed7341 \
NUM_WORKERS=4 \
bash run/run_P3_HR_Memory_night_queue.sh \
> /root/data/BRIGHT-train-logs/P3_HR_NIGHT_MASTER_seed7341.log 2>&1 &
```

队列先执行Preflight和H0。H0失败会终止全部模型实验；其他独立分支即使失败也会
记录退出码并继续下一项。已存在的完整结果会跳过，避免重复训练。最终查看：

```text
/root/data/BRIGHT-train-logs/P3_HR_NIGHT_seed7341/manifest.tsv
/root/data/BRIGHT-train-logs/P3_HR_NIGHT_seed7341/summary.txt
/root/data/BRIGHT-train-logs/P3_HR_NIGHT_seed7341/summary.json
```
