# PI-M0：跨事件伪实例效用Memory审计

## 研究问题

本实验只回答：

> 从训练集检索到的相似LocCC伪实例，能否判断fixed S2对当前建筑整体有益还是有害？

它不重新预测损毁类别，不训练P3/S2，不生成第三种答案，不修改建筑范围，也不访问Val450、Test或官方实例。

## 为什么不是旧Memory的重复

| 旧方案风险 | PI-M0修正 |
|---|---|
| Memory标签是Intact/Damaged/Destroyed，容易替代P3分类 | 标签改为建筑级S2 utility positive/negative |
| 像素Gate让同一建筑产生冲突监督 | 一个LocCC只形成一个查询样本 |
| 同事件邻居可能记住灾害风格 | 正式检索强制排除同事件 |
| 大事件或多数类垄断Memory | 按event × utility label确定性限额 |
| 直接训练Attention，无法确认检索本身有效 | PI-M0完全无参数，先做kNN审计 |
| 只看总体指标 | 同时检查event-macro与正确事件数 |
| 没有拒绝机制 | 邻居不足时标记unavailable，后续只能回退 |

## 数据隔离

PI-M0直接复用BU-E0缓存：

```text
meta_train       -> Memory建库、LOEO选择representation/k/tau/signal
meta_calibration -> 固定配置后一次正式评估
meta_confirm     -> 不访问
Val450/Test      -> 不访问
```

`run.sh audit`会验证BU-E0结果与两个缓存SHA256一致，并要求Building Oracle已经通过。Oracle失败时脚本故意停止。

## Memory键与标签

当前版本先复用BU-E0已经缓存的建筑级、无GT输入特征，避免再次推理导致伪实例漂移：

```text
geometry       = shape
semantic       = P3 entropy/margin/class fractions
s2_behavior    = S2 change amount/direction/purity
base           = geometry + semantic + s2_behavior
base_direct    = base + direct RGB-SAR evidence
base_relation  = base + relation evidence
base_matching  = base + matching evidence
base_grid      = base + grid evidence
base_slic      = base + fixed-SLIC evidence
```

只使用meta_train Memory拟合median/IQR稳健归一化，常数通道被删除，异常值裁剪后L2归一化。

Memory标签：

```text
utility = (corrected - harmed) / changed
utility >= +0.10 -> positive
utility <= -0.10 -> negative
其他             -> ambiguous，不入Memory
```

## 无参数检索

每个utility类别分别取Top-k，再以跨事件上限限制单一事件邻居数量。训练配置选择使用meta_train leave-one-event-out；representation、k、temperature和读出signal均不能在meta_calibration上选择。

候选signal：

- `utility_vote`：邻居真实训练效用的相似度加权均值；
- `affinity_margin`：positive与negative邻居平均相似度差；
- `label_vote`：相似度加权的正负标签票。

## 必须控制组

- geometry-only：排除纯建筑形状捷径；
- base：判断高级RGB-SAR证据是否真有增量；
- same-event：只作事件泄漏诊断；
- shuffled-memory-label：打乱Key与utility的对应；
- random-score：随机基线；
- target permutation：计算正式分数的置换零分布。

## 准入条件

正式跨事件结果必须同时满足：

```text
AUROC >= 0.60
event-macro AUROC >= 0.58
至少10个事件可计算且AUROC >= 0.50
meta_calibration正负建筑各 >= 200
neighbor availability >= 0.95
超过geometry-only >= 0.02 AUROC
若选择高级representation，超过base >= 0.01 AUROC
置换检验p <= 0.05
超过shuffled-memory control
```

通过才输出：

```text
ADMIT_PI_M1_LOW_CAPACITY_VERIFIER
```

否则：

```text
STOP_PI_MEMORY_NO_CROSS_EVENT_UTILITY_SIGNAL
```

## 运行

先完成BU-E0，并确认生成：

```text
/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/cache_seed7341/meta_train.pt
/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/cache_seed7341/meta_calibration.pt
/root/data/BRIGHT-eval-results/SPI_BUILDING_UTILITY_V1/bu_e0_seed7341.json
```

然后：

```bash
cd /root/data/BRIGHT-master/bda_benchmark

bash experiments/spi_pseudo_instance_memory_v1/run.sh preflight

nohup env SEED=7341 \
bash experiments/spi_pseudo_instance_memory_v1/run.sh audit \
> /root/data/BRIGHT-train-logs/SPI_PSEUDO_INSTANCE_MEMORY_M0_seed7341.log 2>&1 &
```

查看：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_PSEUDO_INSTANCE_MEMORY_M0_seed7341.log
```

结果：

```text
/root/data/BRIGHT-eval-results/SPI_PSEUDO_INSTANCE_MEMORY_V1/pi_m0_seed7341.json
/root/data/BRIGHT-checkpoints/SPI_PSEUDO_INSTANCE_MEMORY_V1/utility_memory_seed7341.pt
```

## 与论文/代码的关系

- SRAC公开架构：借鉴“区域token + 外部Memory + 元数据/邻居”的组织方式，但不声称复现其未公开细节。
- RAC（CVPR 2022）：借鉴并行非参数外部Memory用于长尾知识，而不是把所有知识压入参数。
- Iscen等（CVPR 2023）：借鉴检索邻居应被筛选/加权，不能默认所有邻居有益。
- SSN（ECCV 2018）：可微超像素只作为PI-M0/PI-M1通过后的后续模块。
- SelectiveNet：后续PI-M1应报告risk-coverage，而不只报告固定阈值准确率。
- FAISS：当前Memory规模先用PyTorch精确内积；只有规模增长后再更换索引，避免近似检索成为新变量。

主要来源：

- <https://openaccess.thecvf.com/content/CVPR2022/papers/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.pdf>
- <https://openaccess.thecvf.com/content/CVPR2023/papers/Iscen_Improving_Image_Recognition_by_Retrieving_From_Web-Scale_Image-Text_Data_CVPR_2023_paper.pdf>
- <https://github.com/NVlabs/ssn_superpixels>
- <https://proceedings.mlr.press/v97/geifman19a.html>
- <https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index>

