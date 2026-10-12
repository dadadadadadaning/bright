# CMR-C0 冻结关键区域可视化诊断

## 目的与边界

本工具解释 CMR-C0 为什么没有通过准入条件。它只读取已经冻结的 C0 病例缓存、V1 建筑区域缓存、C0 审计 JSON 和原始 RGB/SAR 图像。

- 不重新训练或修改 P3；
- 不改变 RawMean Top-20 候选；
- 不重新选择 Top 10%/20%/30% 或 NMS 方案；
- 关键方案仍使用 Train 已选择的 `selected_variant`；
- 真实标签仅用于 Validation 离线分组；
- 明确拒绝任何带有 Test 来源标记的输入。

## 运行

在 `bda_benchmark` 目录执行：

```bash
bash run/run_CMR_C0_visual_diagnostics.sh preflight
bash run/run_CMR_C0_visual_diagnostics.sh visualize
```

若默认路径不适用，复制并修改：

```bash
cp configs/cmr_c0/visual_paths.example.env /tmp/cmr_c0_visual.env
export CMR_C0_VIS_ENV=/tmp/cmr_c0_visual.env
bash run/run_CMR_C0_visual_diagnostics.sh visualize
```

首次执行需要重新计算 Validation 上 `all` 与 Train 选定关键方案的逐建筑局部分数，因此会使用 GPU；后续默认复用小型 JSON 分数缓存，不再重复匹配。它不会载入巨大的 Train-query 病例缓存。

分数生成或载入后，程序会重新计算两个方案的 Validation AUROC，并要求与原冻结 C0 审计精确复现；保留样本数或 AUROC 不一致会立即停止，避免在错误候选、错误校准器或错误缓存上继续画图。

默认只保存 PNG，避免在当前空间较紧的共享盘上产生大量 PDF。快速试画可进一步减少每组数量：

```bash
VIS_PER_EVENT=1 VIS_MAXIMUM_PER_GROUP=4 SAVE_VIS_PDF=0 \
bash run/run_CMR_C0_visual_diagnostics.sh visualize
```

确认 PNG 正常后，如确实需要矢量容器版本，再设置 `SAVE_VIS_PDF=1`。原始 RGB/SAR 是栅格数据，因此 PDF 仍可能较大。

需要确认分数缓存可复现时才使用：

```bash
FORCE_VIS_SCORES=1 bash run/run_CMR_C0_visual_diagnostics.sh visualize
```

## 四个固定样本组

1. `rescue_target`：真实 Damaged，关键表示的直接 `S_D-S_I` 为正；
2. `harm`：真实 Intact，关键表示的直接 `S_D-S_I` 为正；
3. `critical_degradation`：`all` 的直接判断正确，Train 选定关键表示判断错误；
4. `maximum_disagreement`：`all` 与关键表示的冻结组合概率差异最大。

每个事件最多选 `VIS_PER_EVENT` 个，然后按组内诊断强度排序并截断到 `VIS_MAXIMUM_PER_GROUP`。这用于避免单一大事件主导病例图，不是一次新的模型选择。

## 病例图布局

每张图固定为三行八列：

- 第一行：查询灾前 RGB、灾后 SAR、冻结 component、RGB/SAR 建筑轮廓和身份信息；
- 第二行：原始 D-I 热图、RGB/SAR 叠加、Top 10/20/30、Top20+NMS、core/boundary；
- 第三行：最相似 Intact/Damaged 病例、两类局部相似度矩阵和 `S_I`、`S_D`、`S_D-S_I` 等数值。

热图同时保留两个版本：

- `Local-scale` 仅帮助观察单栋建筑内部的相对位置；
- `Global-scale` 使用整个 Validation 的 D-I 绝对值第 98 百分位作为统一对称色标，支持跨样本比较。

红色表示更支持 Damaged，蓝色表示更支持 Intact；黄色轮廓表示冻结建筑；绿色圆点表示 core token；橙色叉号表示 boundary token。

## 输出

默认目录：

```text
CMR_C0_CRITICAL_ISOLATED/visual_diagnostics_seed3407/
├── CMR_C0_visual_scores_seed3407.json
├── CMR_C0_VIS_manifest.json
├── CMR_C0_VIS_manual_observations.csv
├── CMR_C0_VIS_event_selection_summary.csv
└── case_sheets/
    ├── rescue_target/
    ├── harm/
    ├── critical_degradation/
    └── maximum_disagreement/
```

`CMR_C0_VIS_manual_observations.csv` 已预留人工记录列：热区位置、主要模态特征、空间形态、Top 比例稳定性、NMS 影响、Intact/Damaged 病例匹配质量、`all` 更优原因和最终可信度。病例图生成完成后，应逐行填写这些列，而不是仅凭少数图片下结论。

填写完成后运行：

```bash
bash run/run_CMR_C0_visual_diagnostics.sh summarize
```

这会生成 `CMR_C0_VIS_manual_summary.json`，分别按组、按“组×事件”统计每个观察字段的数量和最常见类别。汇总器不会自动给出继续/停止决定，避免把主观标注伪装成定量性能指标。建议在同一字段中始终使用一致的短词，例如 `building_interior`、`boundary`、`outside_building`、`sar_bright_scatter`、`dispersed`、`not_credible`，不要对同一含义反复换写法。

## 结果解释规则

- 若 Rescue 与 Harm 都集中在边界、背景或孤立 SAR 亮斑：停止使用 P3 D-I 作为关键度；
- 若 Rescue 可解释而 Harm 有稳定噪声模式：只进入可靠性过滤研究，不直接融合；
- 若正确证据分散在整栋建筑：停止硬 Top-K，优先保留整体上下文；
- 若规律只在少数事件成立：视为事件/成像域特征，不能称为跨事件损毁证据；
- 若 Rescue 与 Harm 没有稳定差异：停止当前关键区域路线，不增加匹配器、聚类或多原型复杂度。

由于数据没有像素级损毁标注，本工具不会输出“热区命中真实损毁位置”的定量指标。最终结论必须被描述为受控定性诊断。
