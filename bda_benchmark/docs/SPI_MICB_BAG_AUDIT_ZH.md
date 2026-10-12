# MICB Bag—像素信息流诊断

该实验只读取 CVPRW26 公开 697 张 Validation，不训练模型、不访问 Test。

## 三种区域协议

1. `semantic_cc`：使用 Validation 语义 TIFF 按类别生成连通组件。它复现 MICB
   的训练 Bag 定义，但使用了验证标签确定区域，因此只能诊断，不能部署。
2. `official_instance`：使用官方真实建筑掩膜。它检验伪组件训练得到的 MICB
   是否泛化到真实建筑，同样属于 Oracle 几何诊断。
3. `predicted_loc_cc`：只从 P3 建筑定位预测生成连通组件。它是唯一具有可部署
   几何条件的协议；语义标签仅在统计阶段用于确定区域真值和纯度。

## 核心指标

设区域真值为 `y_j`，MICB Bag 预测为 `b_j`，区域内原始像素多数投票为 `v_j`：

```text
BagAcc = mean[b_j == y_j]
PixelRegionAcc = mean[v_j == y_j]
BagCorrectGivenPixelWrong = count(b_j=y_j and v_j!=y_j) / count(v_j!=y_j)
```

区域级 D→I 可恢复上限：

```text
count(y_j=Damaged, v_j=Intact, b_j=Damaged)
------------------------------------------------
count(y_j=Damaged, v_j=Intact)
```

X→I 同理。像素级版本统计每个区域内部真正发生 D→I/X→I 的像素。

`bag_broadcast_semantic` 把每个 Bag 的预测反事实广播回区域，然后重新计算完整
语义 TIFF 指标。它用于判断“Bag 证据若直接进入像素头会怎样”，不是正式模型。
`oracle_correct_bag_only_semantic` 只广播真值判断正确的 Bag，是不可部署的上限。

此外报告把层级头拆成两部分：

```text
existence_accuracy               Intact / Affected 是否判断正确
severity_accuracy_given_affected Damaged / Destroyed 是否判断正确
factorization_reversal_given_affected
                                 存在头判为Affected但联合三分类又回到Intact
```

最后一个指标用于检测层级概率分解的 argmax 反转。例如
`P(Affected)=0.6`、`P(D|Affected)=P(X|Affected)=0.5` 时，联合概率为
`P(I)=0.4, P(D)=0.3, P(X)=0.3`，最终三分类仍会输出 Intact。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash run/run_SPI_MICB_BAG_AUDIT.sh preflight
SMOKE_IMAGES=2 NUM_WORKERS=4 bash run/run_SPI_MICB_BAG_AUDIT.sh smoke
NUM_WORKERS=4 bash run/run_SPI_MICB_BAG_AUDIT.sh audit
```

默认结果：

```text
/root/data/BRIGHT-eval-results/SPI_MICB_BAG_AUDIT/
SPI_MICB_BAG_p3_locked_validation_full_seed3407.json
```

若要诊断 T1 的 Bag Head，必须显式给出检查点并关闭默认 P3 哈希约束：

```bash
CHECKPOINT=/root/data/BRIGHT-checkpoints/SPI_T1_ISOLATED/seed3407/t1_component_balanced/last.pth \
MODEL_TAG=t1_component_balanced \
EXPECTED_CHECKPOINT_SHA256= \
NUM_WORKERS=4 \
bash run/run_SPI_MICB_BAG_AUDIT.sh audit
```

必须优先查看 `predicted_loc_cc`。如果它的 Bag 正确率低或
`bag_harm_given_pixel_correct` 高，不能直接打开 `mil_residual_scale`；需要先学习
可靠性门控。如果 `semantic_cc` 明显高而 `official_instance`/`predicted_loc_cc` 低，
问题主要是伪组件定义和几何域差异，而不是像素解码器。
