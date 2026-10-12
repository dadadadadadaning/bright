# SRAC机制隔离实验 V1

## 1. 这个实验不是SRAC复现

当前没有公开论文正文和官方代码。本实验只验证公开方法图启发出的通用机制，并保留与SRAC不同的核心问题：在强P3上进行可拒绝、metric-aligned的安全写回。禁止把结果表述为SRAC复现。

## 2. 四层实验

### 层A：区域池化

固定B2基础变化特征和同一批SLICO区域：

- `p0_mean`：区域均值；
- `p1_stats`：mean/std/max；
- `p2_attn_skip`：学习注意力+最大值+零初始化统计跳连；
- `p3_attn_tail`：再加入最异常25%局部证据。

预期：`p2>p1>p0`说明学习区域证据有价值；`p3>p2`说明小面积异常没有被平均淹没。

### 层B：B2证据来源

固定最优池化：

- `c0_base`：原Change Decoder；
- `c1_diff`：显式DiffFusion；
- `c2_enhanced`：B2实际分类前增强特征；
- `c3_dual`：base+diff；
- `c4_structural`：RGB结构+base+diff。

预期：只有`c1/c2/c3/c4`超过`c0`，才能证明DiffFusion提供了新证据。

### 层C：邻居如何参与分类

- `x0_keys`：只让查询通过Cross-Attention读取邻居Key；
- `x1_label_geometry`：加入训练集邻居标签和几何；
- `x2_label_geometry_event`：再加入邻居来源事件；此项风险最高，只作对照。

关键结论必须满足：RAG Macro-F1高于Direct和kNN，且P3错误恢复率提高、潜在伤害不增加。

### 层D：安全路由

- `v1_direct`：P3+Direct，不使用RAG证据；
- `v2_rag`：Keep P3 / Direct / RAG三路选择，加入来源事件多样性。

只有`v2_rag > v1_direct`才能把Memory写成最终贡献。

## 3. 服务器运行顺序

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

export B2_CKPT=/root/data/BRIGHT-master/ChangeMamba-master/results/checkpoints_B2_DiffFusion/B2_ChangeMamba_BRIGHT_DiffFusion_seed3407_C512_I80000_BS2/ChangeMamba-MMBDA_B2_DiffFusion_1783263533.424174/best_model.pth

bash experiments/spi_srac_mechanism_ablation_v1/run.sh preflight
bash experiments/spi_srac_mechanism_ablation_v1/run.sh smoke
bash experiments/spi_srac_mechanism_ablation_v1/run.sh build_train
bash experiments/spi_srac_mechanism_ablation_v1/run.sh build_val
```

先比较池化：

```bash
for pool in p0_mean p1_stats p2_attn_skip p3_attn_tail; do
  bash experiments/spi_srac_mechanism_ablation_v1/run.sh evidence c0_base "$pool"
done
```

若`p3_attn_tail`最好，再比较B2证据：

```bash
for source in c1_diff c2_enhanced c3_dual c4_structural; do
  bash experiments/spi_srac_mechanism_ablation_v1/run.sh evidence "$source" p3_attn_tail
done
```

选定最优source后，以`c4_structural`为例：

```bash
for rag in x0_keys x1_label_geometry x2_label_geometry_event; do
  bash experiments/spi_srac_mechanism_ablation_v1/run.sh rag c4_structural p3_attn_tail "$rag"
done
```

对最佳RAG运行随机标签机制对照：

```bash
bash experiments/spi_srac_mechanism_ablation_v1/run.sh random_control c4_structural p3_attn_tail x1_label_geometry
```

只有最佳RAG超过Direct后才运行：

```bash
bash experiments/spi_srac_mechanism_ablation_v1/run.sh router c4_structural p3_attn_tail x1_label_geometry
bash experiments/spi_srac_mechanism_ablation_v1/run.sh summarize
```

## 4. 准入条件

- 区域证据：Direct Macro-F1提升，P3错误恢复率至少提高3pp；
- 检索：RAG相对Direct至少+0.20pp，随机标签后收益消失；
- 最终Val450：相对P3至少+0.30pp；Damaged F1不下降；Destroyed IoU下降不超过0.10pp；至少10/14事件不下降；接受率不超过70%；
- 事件宏平均bootstrap CI下界必须大于0；代码固定执行10000次重采样；
- Test在所有结构和阈值冻结前始终禁止访问。

## 5. 存储预期

记录缓存保存每个区域最多12个像素证据，Train缓存可能达到约0.5--1.2GB，取决于实际区域数量。它只建立一次，后续所有池化和B2消融复用同一缓存。
