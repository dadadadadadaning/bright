# Source Audit of uploaded `experiments.zip`

对上传压缩包做了完整目录扫描：共 1164 个 archive entries。针对 V1 soft-partition / building-context / region-audit 这条链，进一步对全部 `.py/.sh/.md` 搜索了：

```text
damageformer_a1_cross_modal_superpixel_v1
association_probabilities
predicted_building_instances
```

共命中 25 个相关源码/脚本/说明文件，集中在以下 5 条历史实验链：

1. `damageformer_a1_cross_modal_superpixel_v1`
2. `damageformer_a1_joint_soft_partition_v1`
3. `damageformer_roi_soft_partition_v1`
4. `damageformer_building_context_v2`
5. `damageformer_region_audit_frozen_v1`

## 核对到的关键事实

### V1 association

`damageformer_a1_cross_modal_superpixel_v1/association.py`

- `candidate_map()` 是整幅 feature map 的 fixed-stride global grid。
- candidate IDs 对 batch 共享，不包含 building ownership。
- `soft_pool_unpool()` 只有 weighted mean。

### V1 model

`damageformer_a1_cross_modal_superpixel_v1/model.py`

- `relation = relation_encoder(...) * relation_gate`。
- aligned SAR 进入 token 时再次乘 `relation_gate`。
- final feedback 使用 `building * confidence * update`。
- learnable association 是 `RGB structure + relation + edge + p_building (+ hint)` 直接 concat 后输出 9-way logits。
- building probability 是 feature/gate，不是 hard candidate constraint。

### V1 losses

`damageformer_a1_cross_modal_superpixel_v1/losses.py`

- region losses = spatial + building reconstruction + detached relation-evidence reconstruction + global mass balance (+ optional boundary)。
- evidence loss 又使用 `building * confidence` 权重。
- balance 对所有 positive region mass 使用一个 global mean。

### Joint experiment

`damageformer_a1_joint_soft_partition_v1`

- 直接继承 V1 C5。
- `enable_joint_training()` 让 A1 和 cmsp 同时训练，正是历史不稳定对照。

### ROI soft partition

`damageformer_roi_soft_partition_v1`

- 只是把 V1 C5 固定成单方法，不改变 V1 partition 机制，因此不能解决上述结构问题。

### Building Context V2

`damageformer_building_context_v2/building_ops.py` / `model.py`

- 已经有 `predicted_building_instances()`，但逻辑是先生成 global V1 Region，再 `Region -> Building` 分组。
- 没有做到 `Building -> Region`。
- final building feedback 仍使用 confidence gate。

### Region audit frozen V1

`damageformer_region_audit_frozen_v1`

- 能审计 V1 Region 数量/跨楼/质量。
- 其中 damage entropy / damaged evidence coverage 依赖 semantic damage label；这些指标更适合作为 class consistency，而不能当作真实 sub-building damage localization GT。

## V2 重写策略

因此没有直接覆盖上面五个历史目录，而是新建：

```text
damageformer_building_internal_change_partition_v2
```

这样可以保留：

```text
V1 global joint partition
Frozen-V1
Joint-V1
Building-context-after-region
```

作为后续论文消融与失败机制对照，同时避免修改历史 checkpoint 对应源码。

## V2 仍依赖的外部 BRIGHT 代码

新包只保留以下必要外部依赖：

```text
bda_benchmark.model.DamageFormer_S8_MICB_DSMIL
bda_benchmark.model.DamageFormer_ROI_LCA.ROILocalCorrelationResidual
bda_benchmark.dataset.make_data_loader.MultimodalDamageAssessmentDatset
bda_benchmark.script.standard_ML.train_S5_S8_GPU
bda_benchmark.experiments.p3_damageformer_ablation_v1.core
```

不再 import V1 的 `model.py / association.py / losses.py`，避免把旧逻辑悄悄带回 V2。
