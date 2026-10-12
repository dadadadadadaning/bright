# Source / Reuse Audit

本包是在以下已存在实验基础上新增的“隔离验证层”，不覆盖它们：

- `damageformer_building_internal_change_partition_v2`：Region formation / change refinement / feedback 的主来源。
- `damageformer_region_audit_frozen_v1`：历史 Region audit 思路参考。
- `spi_building_prototype_memory_v1`：Train-only prototype memory、same-event exclusion 的历史经验参考。
- `spi_superpixel_exemplar_memory_v2`：superpixel exemplar、wrong semantic controls、先裸检索后回写的历史经验参考。
- `spi_selective_memory_oracle_v1`：先 Oracle 再训练 verifier 的历史经验参考。
- `spi_multi_prototype_context_v1`：旧多原型上下文属于不同路线，本包不复用其模型权重。

本包重新定义了统一 cache 与 Gate A-E，目的是让 Superpixel -> Tail -> Expert -> Prototype -> Memory 之间可以做可重复的因果比较，而不是继续叠加旧模块。
