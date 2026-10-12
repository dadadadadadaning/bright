# Quality-aware balanced region Memory V2

这是隔离实验目录；不修改任何旧实验。它不是SRAC官方复现，也不承诺Memory有效。

## 研究问题

冻结现有 `source_hard` 的front、Q、建筑概率、区域Embedding和原区域分类器，只研究Train支持库：

1. 旧库把任意 `labelled_mass>0` 的软尾槽当作支持，是否包含大量没有任何硬归属位置的ghost槽？
2. 删除ghost槽后，性能变化来自支持质量，还是来自类别候选分配？

为避免“先cap再过滤”混杂，本实验一次建库产生两个独立reservoir：

- `bank_all.pt`：每个 event×dominant-class 从全部 `labelled_mass>0` 候选保留最多64个；
- `bank_hard.pt`：直接从 `labelled_mass>0 AND hard_count>0` 候选保留最多64个；
- 每个候选只采样一次priority，两库共享priority规则，但各自独立cap。

`hard_count`无阈值：

```text
winner = q.argmax(candidate_axis)
slot = ids.gather(candidate_axis, winner)
hard_count.scatter_add(slot, 1)
```

它只表示区域至少赢得一个native-grid位置，不表示高纯度、真实建筑实例或正确损毁区域。库同时保存 `labelled_mass`、`purity=max(label distribution)`、`entropy`，本轮只审计，不额外过滤。

## 四个配对残差头

四个头均复用 `damageformer_region_memory_v2.ResidualSupportHead`，同参数量、bit-identical初始化、独立存储、同一批次与损失：

| 阶段/名称 | 支持来源 | 要回答的问题 |
|---|---|---|
| M0 `nomemory` | 4个可学习label slots | 增加同构残差容量本身有什么作用？ |
| M1a `global_all` | all库，全局余弦Top-8 | 严格延续旧全局检索基线 |
| M1b `global_hard` | hard库，全局余弦Top-8 | 非ghost支持是否优于全部软尾支持？ |
| M2 `balanced_hard` | hard库，每个dominant class取2个 | 类别均衡返回是否优于同一hard候选池的全局Top-8？ |

主要差值：

```text
M1质量：global_hard - global_all
M2均衡：balanced_hard - global_hard
Memory：global_all - nomemory（以及balanced_hard - nomemory）
```

所有头输出：

```text
final_region_logits = frozen_source_hard_logits + residual_head(...)
```

最后一层零初始化，因此step0四个头逐元素严格等于 `source_hard`，不只是mIoU相同。零初始化不保证训练后不伤害基线。`off`是训练后头只看query fallback，不等于 `source_hard`。

## 防泄漏与固定条件

- 支持只来自Train；Val仅作查询和事后评价；Test不访问。
- 同图永远排除，默认同事件排除。
- balanced只读取Train支持的dominant class；查询GT从不进入检索、排序或分类器。
- hard/hard汇总和回投保持不变，不同时改Q、主干、损失或Memory头容量。
- full-image1024支持与Train随机512查询仍有上下文尺度差异，manifest明确记录。

## 执行顺序

先上传整个目录，然后：

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
export SOURCE_RUN=/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/formal5000_seed3407_20261005_133348

bash bda_benchmark/experiments/damageformer_roi_memory_balanced_v2/run.sh preflight
```

### 1. 冒烟建库

```bash
export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_BALANCED_V2/smoke_bank_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_roi_memory_balanced_v2/run.sh smoke_bank
export BANK_ROOT="$OUTPUT_ROOT"
```

### 2. 先做只读支持筛查

```bash
export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_BALANCED_V2/smoke_audit_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_roi_memory_balanced_v2/run.sh smoke_audit
```

正式结论必须重建完整Train2918库并审计Val450：

```bash
export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_BALANCED_V2/bank_seed3407
bash bda_benchmark/experiments/damageformer_roi_memory_balanced_v2/run.sh bank
export BANK_ROOT="$OUTPUT_ROOT"

export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_BALANCED_V2/audit_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_roi_memory_balanced_v2/run.sh audit
```

审计报告：

```bash
bash bda_benchmark/experiments/damageformer_roi_memory_balanced_v2/run.sh report
```

若normal的label-only kNN证据没有优于random和shuffle_labels，停止复杂Memory训练。主筛查只统计 `labelled_mass>0 AND query hard_count>0` 的非ghost查询，同时报告全部positive查询数量、`supported / hard-labelled`覆盖率以及labelled-mass加权的accuracy/soft-CE；若覆盖率低，不能只引用幸存查询的kNN分数。该gate不是最终模型选择，也不能把区域当独立样本做显著性推断。

### 3. 通过筛查后再训练

```bash
export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_BALANCED_V2/smoke_train_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_roi_memory_balanced_v2/run.sh smoke_train

export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_BALANCED_V2/formal5000_seed3407_$(date +%Y%m%d_%H%M%S)
nohup bash bda_benchmark/experiments/damageformer_roi_memory_balanced_v2/run.sh train \
  > /root/shared-nvme/BRIGHT-train-logs/ROI_MEMORY_BALANCED_V2/formal5000_seed3407.log 2>&1 &
```

正式协议固定：5000 optimizer steps、每500步Val450、batch2×accum8=16、AdamW lr1e-4/wd.005、warmup100+cosine、pixel NLL+Lovasz。最终同step结果为主，best仅作次要记录。

## 结果解释与停止条件

- `global_hard <= global_all`：hard occupancy过滤没有提供增量，停止把“soft tail”作为主因。
- `balanced_hard <= global_hard`：类别均衡返回无增量，停止增加复杂类条件检索。
- Memory头不优于`nomemory`，或normal不优于random/shuffle：没有证明历史支持提供独立信息。
- 只提高Damaged Recall但明显损害Precision、Background或Intact，不应称为长尾改善。
- 单种子Val450差异很小则先重复种子；不能把像素当独立样本，也不使用Test选方案。
