# P3 漏判空间定位审计（只读）

本实验不训练、不改动 P3，也不使用 Memory。它仅解释真实 Affected 像素被 P3 预测为 Background/Intact 时，是否集中在建筑核心、边界环、小损毁区域、RGB–SAR 边缘位移代理区域、SAR 局部高 MAD 区域，或 fixed S2 提出的预测修改簇中。

阈值只在 `meta_train` 冻结：小损毁取真实 Affected 连通区域面积的 25% 分位；高位移代理与高 SAR 噪声分别取建筑像素的 75% 分位。随后仅在 `meta_calibration` 报告漏判率与相对风险。

`alignment_displacement_proxy` 不是配准真值：它只描述“RGB 与 SAR 边缘在小范围平移后是否更一致”。它用于提出局部对齐模块的证据，不能单独证明真实配准误差。

修改簇完全由 P3 的定位预测和 fixed S2 的输出差异生成，不使用 GT。GT 仅用于统计：真实 Affected 漏判中有多少位于 S2 修改簇，以及其中有多少被 fixed S2 恢复。

```bash
bash experiments/spi_p3_gate_miss_spatial_audit_v1/run.sh preflight
bash experiments/spi_p3_gate_miss_spatial_audit_v1/run.sh smoke
bash experiments/spi_p3_gate_miss_spatial_audit_v1/run.sh auto
```

不访问 `meta_confirm`、Val450 或 Test。
