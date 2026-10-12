# SPI Region Feedback：置信门控与核心回写实验

## 1. 研究问题

本实验只回答一个问题：在不使用真实实例监督、不重新训练 P3 的条件下，P3 自身预测出的建筑区域能否提供可靠的区域一致性，使 Standard-ML Val450 上的原始语义预测超过锁定基线 60.5020%。

真实语义 TIFF 只在所有预测完成后用于计算指标和错误审计。区域构造、区域类别、门控决策均不读取 GT。官方实例标注和原始 Standard-ML Test 均不得访问。

## 2. 实验变量

| 变体 | 区域来源 | 回写规则 | 作用 |
|---|---|---|---|
| A0 | 无 | P3 原始逐像素预测 | 锁定基线 |
| A1 | P3 定位 CC | 前景多数投票，整区域回写 | 已验证的区域一致性基线 |
| A2 | P3 定位 CC | 前景平均概率，整区域回写 | 非学习池化对照 |
| A3 | P3 定位 CC | MICB Bag 类别，整区域回写 | 学习池化对照 |
| A4 | P3 定位 CC | 置信门控 Majority，整区域回写 | 检验是否能拒绝危险区域 |
| A5 | P3 定位 CC | 置信门控 Majority，只回写核心 | 检验是否能保护边界 |

A4、A5 与 A1 使用相同定位 CC 和相同 Majority 类别。A4−A1 只测门控价值；A5−A4 只测边界保护价值。

## 3. 数学定义

P3 损毁概率为：

\[
p_c(x)=\frac{\exp z_c(x)}{\sum_{r=0}^{3}\exp z_r(x)}.
\]

P3 定位头预测建筑掩膜：

\[
M^{loc}(x)=\mathbf 1[\arg\max z^{loc}(x)=Building].
\]

对该掩膜做四连通分量并过滤过小区域：

\[
\mathcal R=CC_4(M^{loc}),\qquad |R_j|\ge A_{min}.
\]

区域多数类别只在三个前景类别中选择：

\[
\hat y_j=\arg\max_{c\in\{I,D,X\}}
\sum_{x\in R_j}\mathbf 1[\hat y(x)=c].
\]

纯度的分母使用整个定位区域，因此区域中的背景预测会降低门控置信度：

\[
\rho_j=\frac{
\sum_{x\in R_j}\mathbf 1[\hat y(x)=\hat y_j]
}{|R_j|}.
\]

区域平均概率为：

\[
q_j(c)=\frac{1}{|R_j|}\sum_{x\in R_j}p_c(x).
\]

门控概率间隔要求 Majority 类别同时获得概率支持：

\[
m_j=q_j(\hat y_j)-\max_{c\ne \hat y_j,\ c\in\{I,D,X\}}q_j(c).
\]

因此若多数投票与平均概率的首选类别冲突，则 \(m_j\) 可能为负，区域会被拒绝。

门控为：

\[
g_j=\mathbf 1[
\rho_j\ge\tau_\rho\land
m_j\ge\tau_m\land
|R_j|\le A_{max}
].
\]

当前预注册的首次设置为：

\[
A_{min}=32,\quad \tau_\rho=0.70,\quad
\tau_m=0.10,\quad A_{max}=65536.
\]

A4 在 \(g_j=1\) 时对整个区域回写。A5 先对区域做半径为 2 像素的形态学腐蚀：

\[
R_j^{core}=Erode(R_j,r=2),
\]

只在非空核心回写，边界与区域外部保留 P3 原预测。

## 4. 错误审计

每种方法相对 A0 统计：

\[
Corrected=\#\{x:\hat y_0(x)\ne y(x),\hat y_v(x)=y(x)\},
\]

\[
Harmed=\#\{x:\hat y_0(x)=y(x),\hat y_v(x)\ne y(x)\},
\]

\[
NetGain=Corrected-Harmed.
\]

同时按区域面积、纯度和概率间隔对 A1 的 NetGain 分箱，以判断错误主要来自大粘连区域、低纯度区域还是类别置信冲突。该 GT 信息只用于解释结果，不能回流到同一次推理。

## 5. 统计检验和准入条件

在每个灾害事件内部进行配对图像 Bootstrap，重复 10000 次。对候选方法要求同时满足：

1. P3 精确复现 60.5020%；
2. 相对 A0 的 mIoU 提升不少于 0.5 pp；
3. Damaged F1 提升不少于 1.0 pp；
4. Intact 和 Destroyed IoU 各自下降不超过 0.5 pp；
5. Val450 实际包含 14 个事件；按原 7/11 的比例折算，至少 9/14 个事件 mIoU 不下降；
6. mIoU 配对 Bootstrap 95% CI 下界大于 0。

此外直接输出 A4−A1、A5−A1 和 A5−A4 的置信区间，分别隔离门控及核心回写的贡献。

## 6. 运行顺序

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash run/run_SPI_REGION_FEEDBACK_VAL450.sh preflight

SMOKE_IMAGES=2 NUM_WORKERS=4 \
bash run/run_SPI_REGION_FEEDBACK_VAL450.sh smoke

NUM_WORKERS=4 BOOTSTRAP_ITERATIONS=10000 \
GATE_MINIMUM_PURITY=0.70 \
GATE_MINIMUM_MARGIN=0.10 \
GATE_MAXIMUM_AREA=65536 \
CORE_RADIUS=2 \
bash run/run_SPI_REGION_FEEDBACK_VAL450.sh audit 2>&1 | \
tee /root/data/BRIGHT-train-logs/SPI_REGION_FEEDBACK_GATED_VAL450_seed3407.log
```

完整 JSON、紧凑 summary JSON 和控制台分箱报告保存在：

```text
/root/data/BRIGHT-eval-results/SPI_REGION_FEEDBACK_VAL450/
```

## 7. 后续决策

- 若 A4 > A1 且 A4 的置信区间更稳定：门控有效，下一步将该门控固定后比较 LocCC 与 RGID。
- 若 A5 > A4：边界保护有效，可以形成模型中的空间门控模块。
- 若 A4/A5 不超过 A1：保留简单 Majority，不继续堆叠后处理，直接检验 RGID 几何。
- RGID 与 LocCC 必须使用同一个 Majority/门控器；否则不能把差异归因于实例拆分。
- 只有 `RGID + same pooling` 显著超过 `LocCC + same pooling`，才能声称 learned pseudo-instance geometry 改善了语义分割。
