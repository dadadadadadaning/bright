# V2 Change Manifest

本包不覆盖历史 V1，而是新建独立实验目录，原因是必须保留 V1/Frozen-V1 作为因果对照。

## 重写模块

- `building_ops.py`：predicted Building-first + adaptive per-building candidates + NULL background。
- `association.py`：B-specific candidate IDs、mean/std、soft-peak、soft unpool、entropy。
- `model.py`：RGB structural base + bounded zero-init RGB-SAR residual refinement；confidence/entropy 仅作 feature；三阶段 feedback。
- `losses.py`：building-local balance、weak building bag task loss、stage losses；移除伪 local damage reconstruction。
- `train.py`：Stage-0/1/2；A1 始终 frozen；plugin-only checkpoint，降低磁盘压力。
- `exact_replay.py`：Stage-0/1 identity replay guard。
- `audit.py`：V2 geometry/evidence audit，不再把 building label consistency 当作 local damage GT。
- `variants.py`：P-S/P-C/P-SC 与 peak/bag/uncertainty/shuffled-SAR controls。
- `run.sh` / `install.sh`：服务器 preflight、smoke、三阶段训练、audit。
- `test_contract.py`：same-building candidate、NULL background、mass conservation、soft-peak、weak bag gradient。

## 有意不修改的历史目录

- `damageformer_a1_cross_modal_superpixel_v1`
- `damageformer_a1_joint_soft_partition_v1`
- `damageformer_roi_soft_partition_v1`
- `damageformer_building_context_v2`
- `damageformer_region_audit_frozen_v1`

这些是 V1/失败机制/历史对照，覆盖它们会破坏可重复性。
