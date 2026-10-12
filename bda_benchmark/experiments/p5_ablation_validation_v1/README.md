# P5 模块消融与机制验证（独立实验）

这套代码不改动 P3/P5 主实验，只调用已有训练器并锁定相同训练条件。

## 实验含义

| 编号 | 模型 | 唯一新增内容 | 对照 |
|---|---|---|---|
| E0 | 冻结 P3 | 无 | 已锁定基线 |
| E1 | P5-C0-flat | 原生单通道 SAR 编码器 | E0 |
| E2 | P5-C1-flat | 固定 raw/low/high/gradient 滤波 | E1 |
| E3A | P5-C2Raw-flat | 原生 SAR + 8×8 局部双向 Cross-Attention | E1 |
| E3 | P5-C2-flat | 8×8 窗口双向局部 Cross-Attention | E2 |
| E4 | P5-C2-hierarchical | Occurrence/Severity 残差解码 | E3 |

E1–E4（包括 E3A）使用同一随机种子、有效批量 16、步数、学习率、验证频率和训练/验证列表。
E3A 与 E3 的 Cross-Attention 结构完全相同，唯一差别是 E3A 重复原生 SAR
形成四通道输入，E3 使用固定 raw/low/high/gradient 四视图。E3A 直接从 P3
初始化，不读取 E1、E2 或 E3 的检查点。
E4 与 E3 都从同一个 P3 初始化并训练相同步数；区别只在层级解码器。
长训练中断后，再执行同一个命令会自动读取实验目录中的 `latest_resume.pth`；若
`final.pth` 已存在则自动跳过。只有明确设置 `FORCE_RERUN=1` 才重新训练已完成实验。

检查当前进程、GPU占用、检查点步数和最近验证结果：

```bash
bash experiments/p5_ablation_validation_v1/run.sh status
```

## 推荐执行顺序

下面第一组命令假设当前目录是 `BRIGHT-master`。如果终端提示符已经位于
`BRIGHT-master/bda_benchmark`，请去掉命令开头的 `bda_benchmark/`，例如：

```bash
bash experiments/p5_ablation_validation_v1/run.sh preflight
bash experiments/p5_ablation_validation_v1/run.sh stats
```

```bash
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh preflight
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh stats

for experiment in e1 e2 e3a e3 e4; do
  bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh smoke "$experiment"
done

# 用同一套审计器复现冻结 P3，供逐图配对比较
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh audit e0

for experiment in e1 e2 e3a e3 e4; do
  bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh pilot "$experiment"
  bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh audit "$experiment"
done

bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c1_c0
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c2raw_c0
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c2raw_c0_x8
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c2raw_c0_y8
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c2raw_c0_final
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c2_c1
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c2_c1_x8
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c2_c1_y8
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c2_c2raw
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c2_c0_final
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare hier_flat
bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare final
```

路径通过 `DATASET_ROOT`、`BASE_CKPT`、`SAR_STATS_JSON`、`OUTPUT_ROOT` 和
`AUDIT_ROOT` 环境变量覆盖。默认只访问 validation，专项审计明确拒绝 test。

## 输出说明

每个模型会得到：

- `none.json`：普通精度、逐图混淆矩阵、逐事件指标和内部特征统计；E4 还包含
  同一模型在层级修正前的 `internal_flat`，可直接观察错误流向被怎样改变；
- `shift_x4.json` / `shift_y4.json` / `shift_x8.json` / `shift_y8.json`：
  SAR 人为错位结果；已有 JSON 默认复用，设置 `FORCE_AUDIT=1` 才重新计算；
- `pair_shuffle.json`：错误 RGB-SAR 配对结果；
- `sar_frequency_views.png`：SAR 编码器实际收到的四个通道频谱；C0 是四份 raw，
  C1/C2 是 raw、low、high、gradient，C0/C2Raw 是四份 raw；
- `*_admission.json/.md`：增益、Damaged F1、事件胜负、成对 bootstrap CI 和准入结论。

Cross-Attention 的机制条件是：E3 不仅正常精度高于 E2，而且同样平移 4 像素后，
精度下降量不能比 E2 更大。层级解码条件是：E4 相对 E3 至少提高 0.30pp，且
Damaged F1 不下降。最终条件另查至少 10/14 事件不下降及 bootstrap CI 下界大于 0。

`gradient_norms` 是一次只读验证批次的诊断反向传播结果，不更新权重；它用于检查
新增分支是否真的收到梯度，不作为模型选择指标。
