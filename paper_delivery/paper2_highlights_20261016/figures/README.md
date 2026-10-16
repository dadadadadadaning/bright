# Verified data charts accompanying 论文2

All three charts are drawn with standard Matplotlib from existing reported values. No existing manuscript image was changed. No missing ablation score, confidence interval, or error bar was invented.

## Main-text charts

- `Fig9_class_changes.{png,pdf,svg}`: the reported foreground-class IoUs of DamageFormer, ChangeMamba, MPF and Ours; the right panel derives Ours-minus-reference class differences from the printed source table. Source: `论文2.docx`, first table (zero-based index 0). Ours uses **76.25** for Intact and the main-table reported **69.64** four-class mIoU, rather than silently inheriting the different 76.24/69.62 values from the historical progressive table.
- `Fig10_event_deltas.{png,pdf,svg}`: Ours minus DamageFormer and ChangeMamba over 14 events, with arithmetic equal-event means. The differences are **+1.7771428571 pp** and **+2.6471428571 pp**; positive counts are **11/14** and **10/14**. Source: `论文2.docx`, second table (zero-based index 1). The event table does not clearly establish four-class versus foreground averaging, so the graph deliberately retains the generic label **Event mIoU**. This scope should be specified in the final article. No confidence interval is inferred.

The figures are 17 cm wide, with vector PDF and editable SVG versions. The PNG files are 360 dpi. Original data, comparisons, source SHA256, and verification status are saved in `verified_chart_data.json` and the CSV files.

### Suggested Chinese captions

**图9　完整模型的类别性能及其与参照配置的差异。** (a) DamageFormer、ChangeMamba、MPF和完整Ours的完好、受损、毁坏IoU；(b)完整Ours相对DamageFormer与MPF的四类IoU差值。分数来自本文总体比较表，差值由打印值计算，单位为百分点。该图展示收益集中于受损与毁坏类别，并不单独分离区域分区、结构适配与检索模块的贡献。

**图10　完整模型在14个事件上的配对性能差异。** (a)相对DamageFormer；(b)相对ChangeMamba。青色和红色分别表示正向及负向变化，虚线表示14事件等权差值均值，分别为+1.78和+2.65个百分点。数值来自逐事件报告，采用其事件指标口径；最终归档需明确该口径的类别范围与缺类处理。该图不提供统计显著性结论，也不将跨事件差值解释为单一模块的因果作用。

## Supplementary historical diagnostic

`FigS1_historical_reader_curve.{png,pdf,svg}` plots the previously reported 0–5000-step Val curves. This is the **historical Top-64 cosine parent retrieval / two-slot Reader** experiment. It is **not** the current Top-8 supervised-projection model and must not fill its missing ablation rows. Its source is recorded in the JSON metadata and `historical_top64_twoslot_val_sequence.csv`.

**图S1　历史Top64双槽读取器实验的验证曲线。** A/B分别使用固定空间/建筑soft双槽与图内参考，C/D在对应分槽下读取跨事件Train案例。虚线为冻结基底grid_probq。四组在500步短暂提高，最终均低于基底；该图用于说明既有训练版本的优化与泛化问题，不能作为本文Top8监督投影及结构Adapter的消融结果。

## Rebuild

```bash
python build_verified_charts.py --docx /path/to/论文2.docx --historical-source /path/to/detailed_results_analysis.md
```

The script checks that source event averages match the 14 reported rows to two decimals and that the expected chart comparisons match. The graph provenance is **source-reported**, not independent prediction re-evaluation.
