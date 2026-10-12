# SPI T1：组件均衡伪实例监督的最小因果验证

## 1. 已知证据与本阶段问题

Stage-0 在 697 张公开 Validation 上得到：

| 方法 | valid-mIoU | Damaged F1 | Building Macro-F1 |
|---|---:|---:|---:|
| E0 Raw P3 | 0.5408 | 0.3834 | 0.7117 |
| E2 Damage-class CC 后处理 | 0.5402 | 0.3822 | 0.7095 |
| E4 GT instance pooling | 0.6274 | 0.4246 | 0.7582 |

这证明“正确建筑归属”存在明显价值，但没有证明简单 CC 后处理有效，也没有证明伪组件监督能够改善网络。T1 只回答一个更小、可证伪的问题：

> 在不使用真实实例训练、不修改网络结构、不做实例后处理的条件下，让每个语义前景组件获得相等训练权重，能否改善 P3 的原始逐像素输出？

## 2. 伪组件定义

训练语义标签为 `y(x) in {0,1,2,3,255}`。对每个前景类别分别进行四连通分解：

`R(c,k) = CC4({x | y(x)=c}), c in {1,2,3}`。

面积小于 32 像素的组件不进入组件级损失。不同损毁类别即使相邻也会分开；同类相邻建筑仍会粘连。因此本文将它称为“语义组件”或“伪区域”，不能写成真实建筑实例。

## 3. T0 与 T1 的唯一差异

P3 原目标保留为：

`L_P3 = L_loc + L_damage + L_MICB-DSMIL`。

其中定位与损毁分割均沿用 CE + Lovasz：

`L_seg = L_CE + L_Lovasz`。

对组件 `R_j`，组件平均负对数似然为：

`ell_j = (1/|R_j|) * sum_{x in R_j} -log p_{y_j}(x)`。

组件均衡损失为：

`L_cb = (1/N_R) * sum_j ell_j`。

于是：

- T0：`L_T0 = L_P3`；
- T1：`L_T1 = L_P3 + lambda_cb * L_cb`，冻结 `lambda_cb=0.10`。

普通逐像素 CE 中，一个 1000 像素的大组件约是一个 20 像素小组件的 50 倍权重；`L_cb` 先在组件内部求平均，再对组件求平均，使二者各贡献一个组件单位。原始损失仍然保留，所以 T1 是附加小组件保护，而不是用不可靠伪实例完全替代像素监督。

## 4. 成对控制协议

每个 Seed 都从同一个锁定 P3 checkpoint 独立启动两个训练：

| 固定项 | T0 | T1 |
|---|---|---|
| 初始权重 SHA256 | 相同 | 相同 |
| 数据与训练列表 | 相同 | 相同 |
| 数据顺序与增强随机种子 | 相同 | 相同 |
| Batch、梯度累积、有效 Batch | 相同 | 相同 |
| 优化器、学习率、权重衰减 | 相同 | 相同 |
| 训练步数 | 相同固定步数 | 相同固定步数 |
| CE、Lovasz、MICB/DSMIL | 相同 | 相同 |
| `lambda_cb` | 0 | 0.10 |

训练阶段不读取 Validation 实例，不保存“最佳 Validation”模型，只保存相同固定步数的 `last.pth`。这避免 T0 与 T1 因为选择了不同检查点而产生归因混杂。

`last.pth` 不保存 AdamW 优化器状态，因为本协议不允许续训后再挑选步数；这也避免每个 checkpoint 因一阶、二阶矩而膨胀约两倍。保存过程使用临时文件原子替换，磁盘仍需至少预留约两个模型文件的瞬时空间。

## 5. Raw 验证

验证只使用模型原始 `argmax` 语义图：

`y_hat(x) = argmax_c z_c(x)`。

禁止 CC 池化、GT pooling、Watershed、Gate、Rollback、Memory 或 Oracle。原始
`target/*_building_damage.tif` 是 valid-mIoU、Damaged F1 和 Intact IoU 的主评价真值；
官方实例只用于把 raw 预测汇总成建筑级指标，不作为输入。由实例 JSON 栅格化得到的
语义结果保存在 `instance_raster_semantic` 中，仅用于解释标注域差异，不能替代论文主指标。

每个 Seed 同时报告锁定 Source P3、T0 与 T1；因果主比较是 T1-T0：

- `Delta mIoU = 100 * (mIoU_T1 - mIoU_T0)`；
- `Delta D-F1 = 100 * (D-F1_T1 - D-F1_T0)`；
- `Delta BMacro = 100 * (BMacro_T1 - BMacro_T0)`；
- `Delta Intact-IoU = 100 * (IntactIoU_T1 - IntactIoU_T0)`。

此外要求 T1 相对锁定 Source P3 的 mIoU、Damaged F1 与 Building Macro-F1 均不下降。该条件防止把“T0退化更多、T1退化较少”误报为方法提升。

单 Seed 探索门槛：

- `Delta mIoU >= 0.5 pp`；
- `Delta D-F1 >= 1.0 pp`；
- `Delta BMacro >= 1.0 pp`；
- 至少 10 个事件的 Building Macro-F1 不下降；
- `Delta Intact-IoU >= -0.5 pp`。

## 6. 三 Seed 最终确认

使用 `3407/4407/5407`。最终进入 T2 必须同时满足：

1. 三 Seed 平均的三项主要增益达到上述门槛；
2. 多数 Seed 的三项主要指标同时为正；
3. 每个 Seed 的 Intact IoU 均不跌破安全界限；
4. 对每个事件先跨 Seed 求平均，再要求至少 10 个事件不下降；
5. 全程 `test_accessed=false`。

通过输出 `PROCEED_TO_T2_PSEUDO_GEOMETRY_LEARNING`；失败输出 `STOP_AT_T1_COMPONENT_BALANCING_INSUFFICIENT`。失败只否定“组件等权这一最小机制”，不否定真实实例归属的 Stage-0 上限。

## 7. 运行顺序

服务器数据盘为 `/root/data` 时：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash run/run_SPI_T1_isolated.sh preflight
bash run/run_SPI_T1_isolated.sh smoke

SEED=3407 NUM_WORKERS=4 bash run/run_SPI_T1_isolated.sh full
SEED=4407 NUM_WORKERS=4 bash run/run_SPI_T1_isolated.sh full
SEED=5407 NUM_WORKERS=4 bash run/run_SPI_T1_isolated.sh full

bash run/run_SPI_T1_isolated.sh summarize
```

若 3407 已经训练完成，可同步最新版代码后，在 tmux 中使用门控式续跑脚本。它会先用
双 GT 协议重新审计已有 3407 checkpoint；只有 3407 通过预注册门槛才会继续训练
4407/5407，最后自动汇总：

```bash
bash run/run_SPI_T1_after_seed3407.sh
```

如果希望每一步结束后人工检查、不自动衔接，使用独立模式：

```bash
SEED=4407 bash run/run_SPI_T1_isolated.sh train_t0
SEED=4407 bash run/run_SPI_T1_isolated.sh train_t1
SEED=4407 bash run/run_SPI_T1_isolated.sh audit
```

`train_t0`、`train_t1` 和 `audit` 是三个互相独立的进程；上一条返回提示符并通过
checkpoint 检查后，再人工启动下一条。

若预检报告 `No module named 'cv2'`，说明当前 Conda 环境缺少 OpenCV，而不是数据路径或 T1 代码错误。在激活的环境中安装无图形界面的版本后重新预检：

```bash
python -m pip install opencv-python-headless matplotlib
python -c "import cv2, matplotlib; from dataset.make_data_loader import MultimodalDamageAssessmentDatset; print(cv2.__version__, matplotlib.__version__)"
bash run/run_SPI_T1_isolated.sh preflight
```

若环境变量曾保存旧 `/root/shared-nvme` 路径，应在命令中显式覆盖 `DATASET_ROOT`、`P3_CKPT`、`TRAIN_LIST`、`VAL_LIST`、`SAR_STATS_JSON`、`OUTPUT_ROOT` 和 `AUDIT_ROOT`。

## 8. 结果解释

- T1 Raw 全面提高：伪组件监督确实反馈到语义网络，可进入 T2 学习更好的伪几何。
- Building Macro-F1 提高但像素指标不提高：组件一致性有价值，但边界/背景受到损害，暂不进入 T2。
- 仅 mIoU 提高而 Damaged F1 不提高：可能只是类别分布或 Intact 改善，不支持损毁实例主张。
- T1 与 T0 接近：简单组件等权不足，不能把 E4 的上限归因给它。
- T1 更差：伪组件噪声大于均衡收益，应停止该机制，不得用后处理掩盖 raw 失败。
