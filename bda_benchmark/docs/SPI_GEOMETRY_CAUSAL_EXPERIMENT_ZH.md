# SPI 下一阶段：锁定 A6 的几何因果实验

## 1. 当前证据与本阶段问题

Standard-ML Val450 上，冻结 P3 的复现结果为 `60.5020%`。定位连通组件加多数投票 A1 为
`61.3790%`，锁定门控 A6（纯度 0.70、概率间隔 0.10、无面积上限、无核心腐蚀）为
`61.5399%`。这证明了区域反馈有用，但尚未证明收益一定来自建筑结构，也未证明 RGID 的实例拆分
优于普通定位连通组件。

本阶段只回答两个可证伪问题：

1. 完整建筑连通区域是否优于同等约束下的普通局部平滑？
2. 在分类与回写规则完全相同时，RGID-G3 几何是否优于定位 CC 几何？

## 2. 四个严格配对的变体

| 变体 | 几何来源 | 分类与回写 | 作用 |
|---|---|---|---|
| C0 | 无 | P3 原始逐像素预测 | 锁定基线 |
| C1 | 5×5 局部窗口 | A6 纯度/间隔门控后的局部多数类 | 排除普通空间平滑解释 |
| C2 | P3 定位 CC | 完全相同的 A6 门控与多数投票 | 测试建筑区域上下文 |
| C3 | Learned RGID-G3 | 完全相同的 A6 门控与多数投票 | 测试实例拆分的额外价值 |

C1 只在 C2 所使用的有效定位像素范围内工作，因此 C1 与 C2 使用同一 P3、同一建筑支持范围、
同一最小区域面积、同一纯度阈值和同一概率间隔阈值。两者唯一主要差异是局部窗口还是完整区域。

## 3. 锁定公式

P3 像素类别为

\[
\hat y(x)=\arg\max_{c\in\{0,I,D,X\}}p_c(x).
\]

对一个候选区域 \(R_j\)，前景多数类别为

\[
\hat c_j=\arg\max_{c\in\{I,D,X\}}
\sum_{x\in R_j}\mathbf 1[\hat y(x)=c].
\]

区域纯度为

\[
\rho_j=\frac{\max_c\sum_{x\in R_j}\mathbf 1[\hat y(x)=c]}{|R_j|}.
\]

区域平均前景概率与概率间隔为

\[
q_j(c)=\frac1{|R_j|}\sum_{x\in R_j}p_c(x),\qquad
m_j=q_j(\hat c_j)-\max_{c\ne\hat c_j}q_j(c).
\]

A6 门控固定为

\[
g_j=\mathbf 1[\rho_j\ge0.70\ \land\ m_j\ge0.10].
\]

若 \(g_j=1\)，将 \(R_j\) 回写为 \(\hat c_j\)；否则保持 P3 原预测。面积上限和边界腐蚀
已经由前序实验判定有害，本阶段固定关闭，不允许在 Val450 上继续调参。

C1 将区域 \(R_j\) 替换为像素 \(x\) 周围的 5×5 邻域 \(N_5(x)\)，并且只统计 C2
有效定位掩膜内的像素。其余纯度、间隔和回写逻辑完全相同。

## 4. 两个核心因果对比

结构特异性：

\[
\Delta_{structure}=mIoU(C2)-mIoU(C1).
\]

只有当 C2−C1 的事件分层配对 Bootstrap 95% CI 下界大于 0，才说明完整建筑区域带来的收益
不能被普通 5×5 平滑充分解释。

RGID 几何增益：

\[
\Delta_{RGID}=mIoU(C3)-mIoU(C2).
\]

由于 C2 与 C3 使用相同的 Majority、A6 和 P3，差异才可以归因于 RGID 的 Center、Boundary
与 Watershed/Gate/Rollback 几何，而不是换了分类器。

实现中还强制 C2 与 C3 覆盖完全相同的 P3 有效建筑像素。RGID 拆出的小子区域不会因为面积小于
32而被删除；若 C3 的区域并集与 C2 的有效建筑掩膜不一致，程序直接报错。这样
\(\Delta_{RGID}\) 不会混入“覆盖范围变多或变少”的影响。

C3 的预注册准入条件为：

- C3−C2 至少 `+0.2 pp`；
- C3−C2 的事件分层配对 Bootstrap 95% CI 下界大于 0；
- C3−P3 至少 `+0.5 pp`；
- Damaged F1 相对 P3 至少 `+1.0 pp`；
- 至少 9/14 个事件 mIoU 不下降。

此外报告等事件权重 Bootstrap，防止大事件凭样本数量主导结论；它是稳定性诊断，不替代主指标。

## 5. 数据隔离

- 主评估：Standard-ML Val450 的语义 TIFF。
- 官方真实实例：不读取。
- 原 Standard Test：不读取。
- P3：严格锁定 SHA256 和 `60.5020%` 复现。
- A6 和 5×5 窗口：在运行前锁定，不在 Val450 上搜索。
- C3 只接受伪实例训练、仅训练集校准且从未读取 Val450/测试集的 RGID checkpoint。

## 6. 服务器运行顺序

先运行无需 RGID checkpoint 的结构控制实验：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash run/run_SPI_GEOMETRY_CAUSAL_VAL450.sh preflight
SMOKE_IMAGES=2 NUM_WORKERS=4 bash run/run_SPI_GEOMETRY_CAUSAL_VAL450.sh smoke
NUM_WORKERS=4 BOOTSTRAP_ITERATIONS=10000 \
  bash run/run_SPI_GEOMETRY_CAUSAL_VAL450.sh audit 2>&1 | \
  tee /root/data/BRIGHT-train-logs/SPI_GEOMETRY_CAUSAL_controls_seed3407.log
```

若 C2 显著优于 C1，再训练/校准伪监督 RGID，并运行：

```bash
RGID_CKPT=/root/data/BRIGHT-checkpoints/RGID_PSEUDO_ONLY/seed3407/best.pth \
RGID_PROVENANCE_JSON=/root/data/BRIGHT-checkpoints/RGID_PSEUDO_ONLY/seed3407/provenance.json \
NUM_WORKERS=4 BOOTSTRAP_ITERATIONS=10000 \
  bash run/run_SPI_GEOMETRY_CAUSAL_VAL450.sh audit 2>&1 | \
  tee /root/data/BRIGHT-train-logs/SPI_GEOMETRY_CAUSAL_rgid_seed3407.log
```

`provenance.json` 必须真实记录：

```json
{
  "protocol": "RGID_PSEUDO_ONLY_TRAIN_v1",
  "official_instances_accessed": false,
  "standard_ml_val_accessed": false,
  "original_standard_test_accessed": false,
  "calibration_split": "train_only",
  "p3_sha256": "与锁定P3完全一致的SHA256"
}
```

不要为了通过校验而虚构该文件；只有训练过程真实满足这些条件时，C3 才有论文证据效力。

## 7. 决策顺序

1. 若 C2 不显著优于 C1：只能声称“自适应空间去噪有效”，暂停 RGID 语义反馈主张。
2. 若 C2 显著优于 C1：建筑结构特异性成立，才值得训练在线/紧凑 RGID。
3. 若 C3 不优于 C2：保留 A6 作为语义后处理，RGID 只作为实例分割支线。
4. 若 C3 满足全部门槛：再做 3 seeds、消融和最终 Test；此前不访问 Test。
