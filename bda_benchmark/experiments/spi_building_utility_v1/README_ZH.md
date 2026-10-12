# BU-E0：建筑级S2效用可学性审计

## 目的

本实验不训练新模型，只回答两个问题：

1. 整栋建筑在P3与fixed S2之间二选一，理论上还能超过fixed S2多少？
2. 建筑级形状、语义、S2行为或RGB–SAR证据能否区分“S2对该建筑总体有益/有害”？

BU-E0不修改P3、不修改S2、不训练GNN，也不访问Val450、Test或官方实例。

## 决策单位与标签

只保存fixed S2发生argmax修改的P3 LocCC建筑。对建筑B计算：

```text
net_B = corrected_B - harmed_B
utility_B = net_B / changed_B
```

预注册标签：

```text
utility_B >= +0.10：positive
utility_B <= -0.10：negative
其余：ambiguous，只统计，不参与AUROC
```

Building Oracle使用GT判断整栋建筑接受P3还是fixed S2，选择标准是“正确像素净增益是否大于0”。它是不可部署的正确像素上界，不应表述为全局mIoU最优化Oracle。

## 新的三分隔离

输入是原来的2627张`gate_train`，按14个事件重新冻结：

```text
meta_train       80%：选择标量通道和方向
meta_calibration 10%：BU-E0可行性准入；未来BU-E1阈值选择
meta_confirm     10%：BU-E1结构与阈值冻结后才允许构建缓存并访问一次
```

BU-E0运行脚本只构建`meta_train`和`meta_calibration`缓存。它只创建`meta_confirm.txt`名单，不读取对应图像。旧291张development calibration不进入本实验。

## 特征组

- `shape`：面积、bbox填充率、长宽比、边界比例；
- `semantic`：P3建筑内entropy、margin及基础类别比例；
- `s2`：修改覆盖率、概率移动、区域纯度/margin、16种修改方向；
- `direct`：同位RGB–SAR直接差的建筑级统计；
- `relation`：局部模态内关系差的建筑级统计；
- `matching`：IM4CD式局部匹配证据的建筑级统计；
- `grid/slic`：等预算Grid与固定pre-RGB SLIC聚合统计。

当前`grid/slic`仍是区域统计，不是真正的建筑内图关系SP-E0。只有BU-E0证明建筑决策上限足够后，才有资格实现共享节点图。

## 准入

Building Oracle相对fixed S2必须至少提高`+0.30pp` mIoU，并且至少`10/14`事件不低于fixed S2。

证据组必须满足：

```text
AUROC >= 0.60
event-macro AUROC >= 0.58
至少10个事件可计算且方向正确
meta_calibration正负建筑各至少200
```

高级RGB–SAR证据还必须超过简单建筑控制至少`0.02` AUROC；SLIC还必须超过等预算Grid至少`0.01`。

只有Oracle上限和至少一个证据组同时通过，才输出：

```text
ADMIT_BU_E1_SINGLE_BUILDING_VERIFIER
```

## 服务器运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark

bash experiments/spi_building_utility_v1/run.sh preflight
bash experiments/spi_building_utility_v1/run.sh smoke
```

两个步骤通过后运行完整BU-E0：

```bash
nohup env \
NUM_WORKERS=4 \
MATCHING_MAX_SIDE=80 \
MATCHING_WINDOW=5 \
CELL_SIZE=4 \
SEED=7341 \
bash experiments/spi_building_utility_v1/run.sh audit \
> /root/data/BRIGHT-train-logs/SPI_BUILDING_UTILITY_E0_seed7341.log 2>&1 &
```

查看进度：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_BUILDING_UTILITY_E0_seed7341.log
```

提取关键结果：

```bash
grep -E 'BU cache|mIoU=|Oracle gain|AUC=|Admitted|Decision:|Saved:' \
/root/data/BRIGHT-train-logs/SPI_BUILDING_UTILITY_E0_seed7341.log
```

结果位置：

```text
/root/data/BRIGHT-eval-results/SPI_BUILDING_UTILITY_V1/bu_e0_seed7341.json
```

## 停止规则

- Building Oracle不足`+0.30pp`：停止建筑选择路线；
- Oracle足够但无证据通过：停止BU-E1，考虑真正SP-E0无参数图审计；
- 简单建筑特征通过：先训练低容量BU-E1，不增加图；
- 高级证据独立通过：BU-E1只接一个通过组；
- 不得根据结果修改`utility_margin=0.10`后把同一meta_calibration重新当正式准入。

缓存会被复用。如果修改minimum area、matching窗口、SLIC/Grid尺度或证据组，必须指定新的`CACHE_ROOT`，不能混用旧缓存。
