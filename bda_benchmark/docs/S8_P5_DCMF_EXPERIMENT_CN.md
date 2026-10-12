# S8-P5 DCMF：原生 SAR 解耦交叉模态融合实验

## 一、实验定位

P4-A1/A2/A3 已经证明：只替换 SAR 归一化、单通道 PatchEmbed 或 ConvStem，
无法让冻结的 P3 SAR 主干理解新的输入分布。因此 P5 不再继续修改 P4 输入头，
而是建立一条独立学生路线：

```text
灾前 RGB → 冻结的 P3 RGB Swin → RGB 多尺度特征 → 建筑定位

灾后单通道 SAR
    → 固定 SAR 滤波增强
    → 独立 SAR 分层编码器
    → SAR 多尺度投影

RGB/SAR 特征
    → 局部双向交叉注意力
    → 公共/私有特征解耦
    → 损毁分类 + MICB-DSMIL 实例监督
```

P3 权重 `45fb...` 只承担三个职责：

1. 初始化并冻结可靠的 RGB 定位分支；
2. 初始化损毁分类头和 MICB-DSMIL；
3. 作为冻结教师提供损毁输出和特征蒸馏。

P5 的 SAR 编码器不与 RGB 共享输入头或权重。

## 二、模块依据

### 1. SAR 固定滤波增强

依据：[SARDet-100K / MSFA，NeurIPS 2024 Spotlight](https://github.com/zcablii/SARDet_100K)

MSFA 从输入滤波、领域过渡和模型迁移三个阶段处理 RGB 到 SAR 的差距。P5
采用可审计的轻量版本，生成：

```text
原始强度、局部低频、高频残差、Sobel 梯度幅值
```

滤波器没有可训练参数，因此不会把滤波增强和网络容量增加混为一谈。

### 2. 模态专用编码与投影

依据：[AnySat，CVPR 2025 Highlight](https://github.com/gastruc/AnySat)

AnySat 对不同传感器使用独立 projector，再进入共享表征空间。P5 对 SAR
使用独立四级编码器和多尺度 projector，不再套用 RGB 的 PatchEmbed。

### 3. 雷达—光学交叉注意力

依据：[CROMA，NeurIPS 2023](https://github.com/antofuller/CROMA)

CROMA 分别编码雷达和光学 token，再使用 Cross-Attention 得到联合表征。
P5 在 1/4 尺度采用双向窗口交叉注意力：

```text
RGB Query ← SAR Key/Value
SAR Query ← RGB Key/Value
```

窗口化把注意力复杂度限制在 640×640 crop 可承受范围内。

### 4. 公共/私有表征解耦

依据：[DeCUR，ECCV 2024 Oral](https://github.com/zhu-xlab/DeCUR)

P5 将两种模态投影为：

```text
96维公共特征 + 32维私有特征
```

公共特征的跨模态相关矩阵向单位阵靠近，私有特征的跨模态相关矩阵向零靠近。
因此 RGB/SAR 共享建筑变化语义，同时允许 SAR 保存散射与纹理特有证据。

### 5. 建筑实例监督

沿用已在本项目中取得正向结果的 MICB-DSMIL：

```text
建筑区域 = Bag
区域内位置 = Instance
建筑损毁等级监督 Bag，而不是强迫所有局部像素具有相同损毁等级
```

## 三、受控消融

| 变体 | 原生 SAR 编码器 | 滤波增强 | 交叉注意力 | 公共/私有解耦 |
|---|---:|---:|---:|---:|
| C0 | 是（原始SAR复制为4个相同视图） | 否 | 否 | 否 |
| C1 | 是 | 是 | 否 | 否 |
| C2 | 是 | 是 | 是 | 否 |
| C3 | 是 | 是 | 是 | 是 |

比较关系：

```text
C1-C0：MSFA 风格滤波增强贡献（输入通道数、Stem形状和参数量保持一致）
C2-C1：CROMA 风格交叉注意力贡献
C3-C2：DeCUR 公共/私有解耦贡献
```

不要首先只运行 C3 后直接声称全部模块有效。正式论文结论必须依赖上述嵌套消融。

## 四、训练策略

### 阶段一：SAR 对齐

```text
0～1000 optimizer steps
训练：SAR编码器、SAR多尺度投影、DCMF
冻结：RGB编码器、定位头、损毁头、MICB-DSMIL、全部BatchNorm
SAR教师特征蒸馏权重：1.0
```

### 阶段二：损毁适配

```text
1000步以后
继续训练：SAR编码器、SAR投影、DCMF
解冻：clf_2、MICB-DSMIL
保持冻结：RGB编码器、定位头、全部BatchNorm
SAR教师特征蒸馏权重降至：0.25
```

默认学习率：

```text
SAR编码器与投影：1e-4
DCMF融合模块：   1e-4
P3损毁/实例头：  1e-6
Weight decay：   5e-3
EBS：            16（BS=4，ACC=4）
```

损失：

```text
L = L_damage_CE+Lovasz
  + L_MICB_DSMIL
  + 1.00 L_damage_KD
  + 0.25 L_location_KD
  + 0.10 L_fused_feature_KD
  + λ_sar L_SAR_feature_KD
  + 0.05 L_DeCUR_common
  + 0.02 L_DeCUR_unique
```

定位损失只记录，不反向传播，因为 RGB 定位路径严格冻结。

## 五、运行命令

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_S8_P5_DCMF_5090.sh preflight
bash run/run_S8_P5_DCMF_5090.sh stats

VARIANT=c3 bash run/run_S8_P5_DCMF_5090.sh smoke
VARIANT=c3 bash run/run_S8_P5_DCMF_5090.sh pilot
```

连续完成 C0～C3：

```bash
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/S8_P5_DCMF
mkdir -p "$LOG_ROOT"

nohup env \
  VARIANTS="c0 c1 c2 c3" \
  MAX_RETRIES=1 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_S8_P5_DCMF_sequence.sh pilot \
  > "$LOG_ROOT/controller_pilot.log" 2>&1 &

tail -f "$LOG_ROOT/controller_pilot.log"
```

模式定义：

```text
smoke：     20步
pilot：     5000步
full：      8000步，对应当前单卡5090的128000 crops协议
paper50k：  50000 optimizer steps，只在明确采用论文50k协议时运行
```

## 六、准入标准

1. 每个变体必须先通过 Preflight 和 20 步 Smoke。
2. Pilot 只使用训练集和完整验证集，不访问 Test。
3. 3000 步前不使用旧 P4 的 0.3 pp 早停，因为新 SAR 编码器需要完成领域过渡。
4. 3000 步后连续两次低于 A0 超过 1.0 pp，Pilot 才提前停止。
5. C1/C2/C3 只有在对应前一变体上取得稳定增益时，才可归因于新增模块。
6. Full 前至少检查：
   - Localization F1 与 P3 基本一致；
   - Damaged/Destroyed IoU 没有以牺牲一类换取另一类；
   - `common_diag` 上升；
   - `unique_correlation` 下降；
   - 无 NaN、无显存溢出、严格加载保留参数。

## 七、正式实验记录模板

```text
实验名称：
变体：
随机种子：
基础教师SHA256：
完成optimizer steps：
总训练crops：

结构开关：
  Filter augmentation：
  Cross attention：
  Common/unique decoupling：
  MICB-DSMIL：

验证指标：
  Final mIoU：
  Valid mIoU：
  Localization P/R/F1/IoU：
  Intact/Damaged/Destroyed IoU：
  Damaged/Destroyed F1：
  Damage→Intact：
  Destroyed→Intact：
  Event-mean mIoU：

表征诊断：
  common_diag：
  unique_correlation：
  kd_sar_loss：
  kd_damage_loss：

结论：
  是否超过前一消融：
  是否超过A0：
  是否进入Full：
  是否允许访问Test：
```
