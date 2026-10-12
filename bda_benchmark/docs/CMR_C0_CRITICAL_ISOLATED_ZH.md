# CMR-C0：冻结损毁关键区域病例卡与反事实检索

## 1. 本阶段回答什么问题

本阶段只验证一个问题：

> 与整栋建筑的全部局部 Token 相比，只保留 P3 认为最像“Damaged 而非 Intact”的空间位置，能否更安全地识别 P3 的 D→I 漏检？

只有 C0 通过准入门槛，才允许继续灾害类型组织、多原型或融合。C0 不修改 P3，不输出新分割图，也不访问 Test。

## 2. 冻结项与唯一变量

冻结项：

- P3 权重及预测；
- V1 已冻结的建筑 connected component；
- P0 RawMean 表示；
- M3 RawMean Top-20 候选及顺序；
- Train/Validation 划分；
- 查询条件：P3 预测 Intact，且 Top-20 中至少有一个 Damaged 病例。

唯一变量是每张病例卡选哪些 `fused_roi` 空间 Token：

| 变体 | 含义 |
| --- | --- |
| `all` | C0-A：建筑内全部 `fused_roi` Token |
| `di_top10` | C0-B：按 `logit_D-logit_I` 取 Top 10% |
| `di_top20` | C0-C：取 Top 20% |
| `di_top30` | C0-D：取 Top 30% |
| `di_top20_nms` | C0-E：Top 20% 加空间 NMS，多保留分散位置 |

关键 Token 数为 `ceil(r*N)`，默认最少 4、最多 32；若整栋建筑不足 4 个则全部保留。Intact 与 Damaged 病例执行同一选择规则，GT 标签不参与 Token 选择。

## 3. 一张病例卡保存什么

每栋建筑保存：

- `features`：`fused_roi` 的 256-D、逐 Token L2 归一化特征；
- `xy`、`normalized_xy`：空间位置；
- `damage_logits`：同一位置的 `[Intact, Damaged, Destroyed]` logits；
- `di_margin`：`Damaged logit - Intact logit`；
- `building_confidence`、`component_weight`；
- `is_boundary`：仅供可靠性诊断；
- 五套 `critical_indices`，不重复保存五份 256-D 特征；
- 事件、灾害类型、bbox、区域指纹及上游文件来源。

灾害类型在 C0 中只保存为元数据，绝不用于路由或计分。

## 4. 检索如何计算

对冻结 RawMean Top-20 中每个 I/D 病例：

1. 每个查询关键 Token 在该病例的关键 Token 中找余弦相似度最高的 3 个；
2. 先对这 3 个相似度求均值，再对所有查询 Token 求均值；
3. 分别汇总 Top-20 中 Intact 病例得分 `S_I` 与 Damaged 病例得分 `S_D`；
4. 核心反事实证据为 `Delta=S_D-S_I`。

这是严格的 Query→Memory 非对称匹配。查询端的每个损毁关键位置都必须在历史病例中找到证据，但不会要求病例中的每个位置反向匹配查询。

## 5. 运行顺序

先确认 V1 的三个 Region Cache 已存在，然后：

```bash
cd ~/shared-nvme/BRIGHT-master/bda_benchmark

bash run/run_CMR_C0_critical_isolated.sh preflight
bash run/run_CMR_C0_critical_isolated.sh build
bash run/run_CMR_C0_critical_isolated.sh audit
```

也可一次执行：

```bash
bash run/run_CMR_C0_critical_isolated.sh all
```

如路径不同，复制 `configs/cmr_c0/paths.example.env`，修改后：

```bash
export CMR_C0_ENV=/absolute/path/to/cmr_c0.env
bash run/run_CMR_C0_critical_isolated.sh all
```

`build` 依次生成 support、train_query、validation_query 病例卡。已存在的完整缓存会跳过；只有明确设置 `FORCE_CACHE=1` 才重建。

## 6. 输出与准入标准

默认输出：

```text
/root/shared-nvme/BRIGHT-eval-results/CMR_C0_CRITICAL_ISOLATED/
├── CMR_C0_train_support_cases_seed3407.pth
├── CMR_C0_train_query_cases_seed3407.pth
├── CMR_C0_val_query_cases_seed3407.pth
└── CMR_C0_critical_audit_seed3407.json
```

关键比例只由 Train 的 leave-one-incident-out 分数选择，再锁定到 Validation。继续下一阶段必须同时满足：

- 至少 100 个 Validation proposal；
- Train 选中的不是 `all`；
- critical combined AUROC 比 `all` 至少高 0.05；
- critical set-only AUROC 比 `all` 至少高 0.05；
- event-macro AUROC 不下降；
- Train 锁定阈值在 Validation 上 precision≥0.55、净纠正>0、rescued≥10、至少 10 个事件净纠正非负。

结果还单独报告：D→I rescue exposure、Intact harm exposure，以及 RawMean Top-1 不是 D 但 Top-20 含 D 的 unused-D headroom。

若输出：

```text
STOP_AT_C0_CRITICAL_REGION_REPRESENTATION_INSUFFICIENT_DO_NOT_FUSE
```

含义不是代码失败，而是当前损毁关键区表示没有提供足够增益；此时不应继续灾害类型聚类、多原型或 P3 融合。

## 7. 代码位置

- 病例卡、关键区选择、非对称检索：`model/cmr_c0/case_card.py`
- 病例卡构建：`script/standard_ML/cmr_c0/build_case_cache.py`
- Train 选择和 Validation 审计：`script/standard_ML/cmr_c0/audit_critical_verifier.py`
- 独立入口：`run/run_CMR_C0_critical_isolated.sh`
- 单测：`tests/cmr_c0/test_case_card.py`、`tests/cmr_c0/test_retrieval.py`、`tests/cmr_c0/test_audit_pipeline.py`

## 8. 隔离保证

CMR-C0 只读旧 P3/P0/M3/V1 Region Cache，只写 `CMR_C0_CRITICAL_ISOLATED`。它没有导入或调用 CMR-V2 的事件专家，不修改 V0/V1/V2/P3 文件，也没有 Test 参数或测试集入口。
