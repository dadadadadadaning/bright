# S8-P4 NS-MICB 原生 SAR 改造实验记录

## 1. 实验身份

- 方法家族：`S8-P4 NS-MICB`
- 英文全称：`S8-P4 Native-SAR Multi-Instance Cross-Modal Building`
- 推荐代码类名：`DamageFormer_S8_P4_NativeSAR_MICB`
- 母模型：`S8_MICB_DSMIL_P3`
- 母模型 checkpoint：

  ```text
  /root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/
  S8_P3_DSMIL_EBS16/best_val_miou.pth
  ```

- 母模型 SHA256：

  ```text
  45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7
  ```

- 母模型实际身份：`step=4000`，不是历史记录中的 step 8000 checkpoint。
- 状态：本地代码实现与静态检查完成，等待服务器 Preflight/Smoke/Pilot。
- 日期：2026-07-28

## 2. 命名规则

### 2.1 家族名称

推荐使用：

```text
S8-P4 NS-MICB
```

其中：

- `S8`：延续当前阶段编号；
- `P4`：表示在 P3 MICB-DSMIL 之后形成的新一代结构；
- `NS`：Native SAR，表示不再复制为伪 RGB；
- `MICB`：保留 P3 已验证的建筑级跨模态多实例监督。

不建议使用：

```text
S8-R6
```

因为 R0–R5 已经属于另一条 ReliableCrossModal 消融路线。

也不建议继续使用：

```text
S8-P3-A2
```

因为单通道输入头改变了 SAR Patch Embedding 和归一化协议，已经形成新的模型代际。

### 2.2 完整实验名

统一格式：

```text
S8P4_<变体>_<SAR输入>_<MICB状态>_base45fb_seed<种子>
```

示例：

```text
S8P4_A0_PseudoRGB_MICB_base45fb_seed3407
S8P4_A1_SARStatsRepeat3_MICB_base45fb_seed3407
S8P4_A2_NativePatch1C_MICB_base45fb_seed3407
S8P4_A3_NativeConvStem1C_MICB_base45fb_seed3407
S8P4_B1_NativePatch1C_RelToken_MICB_base45fb_seed3407
S8P4_C1_NativePatch1C_RelToken_PixelHierAux_base45fb_seed3407
S8P4_C2_NativePatch1C_RelToken_PixelHierBounded_base45fb_seed3407
```

## 3. 已锁定母模型

### 3.1 模型结构

`S8_MICB_DSMIL_P3` 的主体为：

```text
S3V2/BGMP-K3像素主干
+
建筑先验ROI
+
MICB多模态建筑Token
+
DSMIL建筑级辅助监督
+
DamageExist/Severity层级辅助头
+
Intact Guard
```

推理阶段保持：

```text
mil_residual_scale=0
```

建筑 Bag 预测不直接广播回像素 logits。

### 3.2 当前基线指标

验证集，共450张：

| 指标 | 数值 |
|---|---:|
| Final mIoU | 69.5772 |
| Valid 3-class mIoU | 60.5019 |
| Background IoU | 96.8031 |
| Intact IoU | 75.5791 |
| Damaged IoU | 43.0309 |
| Destroyed IoU | 62.8957 |
| Localization F1 | 90.5704 |
| Event-mean mIoU | 52.7280 |
| Damaged → Intact | 30.2791 |
| Destroyed → Intact | 12.3192 |

测试集，共878张：

| 指标 | 数值 |
|---|---:|
| Final mIoU | 67.2725 |
| Valid 3-class mIoU | 57.4086 |
| Background IoU | 96.8643 |
| Intact IoU | 75.2360 |
| Damaged IoU | 40.0857 |
| Destroyed IoU | 56.9040 |
| Localization F1 | 90.3354 |
| Event-mean mIoU | 52.6139 |
| Damaged → Intact | 33.2035 |
| Destroyed → Intact | 18.6801 |

测试集结果只用于最终报告，不用于 P4 结构选择和调参。

## 4. 当前问题

现有数据加载流程将单通道 SAR 执行：

```text
SAR B×1×H×W
→ 复制三次
→ B×3×H×W
→ ImageNet RGB mean/std
→ RGB预训练Patch Embedding
```

这同时引入三个输入偏差：

1. SAR 被人为复制成三个没有新增信息的通道；
2. SAR 使用了 RGB ImageNet 归一化统计；
3. SAR 进入了面向自然图像设计的三通道 Patch Embedding。

P4 的首要目标不是继续增加融合模块，而是先纠正 SAR 输入表征。

## 5. 核心假设

### H1：原生通道假设

如果消除伪 RGB 输入形式，SAR 分支应提供更有效的损毁散射证据，使：

```text
Damaged IoU提高
Destroyed IoU提高
Damaged/Destroyed → Intact错误率下降
```

同时 RGB 定位分支应基本保持稳定。

### H2：输入头容量假设

如果简单单通道 Patch Embedding 已有效，则说明主要问题是输入通道和统计错配。

如果双层卷积 Stem 进一步有效，则说明逐级局部散射建模也有贡献。

### H3：关系建模假设

只有原生 SAR 表征产生正增益后，局部结构关系变化才值得替换 MICB 中的简单：

```text
abs(rgb-sar)
rgb*sar
```

### H4：层级反馈假设

P3 的建筑级 DamageExist/Severity 目前只作为辅助监督。像素级层级证据只有在输入和关系模块稳定后，才能以有界残差形式进行独立消融。

## 6. 第一组消融：只验证 SAR 输入

这一组必须保持 MICB、DSMIL、ROI、损失权重和训练预算相同。

| 变体 | 正式名称 | SAR处理 | 目的 |
|---|---|---|---|
| P4-A0 | `PseudoRGB` | 复制三通道＋ImageNet归一化 | 同训练协议对照 |
| P4-A1 | `SARStatsRepeat3` | SAR训练集统计＋复制三通道 | 单独验证归一化 |
| P4-A2 | `NativePatch1C` | 单通道1→96 Patch Embedding | 验证原生通道和独立Patch投影 |
| P4-A3 | `NativeConvStem1C` | 双层3×3卷积Stem＋1×1投影 | 验证局部散射建模 |

### P4-A0

保持原 P3 输入形式，但必须接受与 A1/A2/A3 完全相同的继续训练、冻结、蒸馏和优化步数。

它用于排除：

```text
额外训练步数
蒸馏
学习率
冻结策略
```

本身带来的收益。

### P4-A1

流程：

```text
单通道SAR
→ 训练集p01/p99截断
→ 训练集mean/std标准化
→ 复制三通道
→ 原SAR Swin Patch Embedding
```

只回答：

> SAR 专用归一化是否有效？

### P4-A2

流程：

```text
B×1×H×W
→ Conv2d(1,96,kernel=4,stride=4)
→ NCHW转NHWC
→ LayerNorm(96)
→ 原SAR Swin Stage1–4
```

初始化建议：

```text
W_1ch = W_R + W_G + W_B
```

这是合理初始化，但不是严格等价迁移，因为旧流程的三个复制通道使用了不同的 ImageNet mean/std。

最关键比较：

```text
P4-A0 vs P4-A2
```

### P4-A3

流程：

```text
1→32，3×3，stride=2，GroupNorm，GELU
32→64，3×3，stride=2，GroupNorm，GELU
64→96，1×1
NHWC＋LayerNorm(96)
→ 原SAR Swin Stage1–4
```

关键比较：

```text
P4-A2 vs P4-A3
```

## 7. 第二组消融：关系 Token

只有 A2 或 A3 相对 A0 通过验收后才启动。

### P4-B0

使用第一组胜出的 SAR 输入，保持原 MICB Token：

```text
T=phi[R,S,abs(R-S),R*S,U]
```

### P4-B1

第一版只在 MICB/DSMIL 辅助分支中替换关系特征：

```text
T=phi[R,S,D_relation,U]
```

其中 Stage2/Stage3 使用共享关系投影器，确保 RGB/SAR 在同一映射规则下比较。

第一轮不把关系模块注入像素主路径，避免破坏 P3 的安全边界。

后续可测试：

```text
独立投影器＋弱关系一致性约束
```

但不能与 SAR Stem 同轮改变。

## 8. 第三组消融：像素层级监督

只有输入和关系模块稳定后才启动。

### P4-C1：PixelHierAux

在 `fused_roi` 上增加像素级：

```text
DamageExist
Destroyed | Damage
```

只增加辅助监督，不反馈主 logits。

### P4-C2：PixelHierBounded

将层级概率转换为对数证据：

```text
hI = log(1-Pdamage+epsilon)
hD = log(Pdamage*(1-Pseverity)+epsilon)
hX = log(Pdamage*Pseverity+epsilon)
```

仅修正三个建筑类别：

```text
main_logits[:,1:4] += beta_h * [hI,hD,hX]
```

建议：

```text
beta_h初始值=0
warmup后逐步增加
beta_h最大值=0.05
```

不得直接将概率与 logits 相加。

## 9. 数据统计协议

训练前必须先确认 SAR 文件物理含义：

```text
幅度 / 功率 / dB / 8位显示图
```

在物理含义未确认前，不自动使用 `log1p`。

SAR 统计量只能由训练集计算：

```text
p01_train
p99_train
mean_train
std_train
```

验证集和测试集只能读取冻结统计，禁止重新估计。

推荐归一化：

```text
x_clip = clip(x,p01_train,p99_train)
x_norm = (x_clip-mean_train)/(std_train+epsilon)
```

统计文件需要记录：

```text
训练列表SHA256
样本数
有效像素数
数据类型
最小值/最大值
p01/p99
mean/std
计算脚本版本
```

## 10. 固定实验条件

第一组消融中下列变量必须完全一致：

```text
母模型SHA256=45fb0e...
seed=3407
train/val/test划分
训练增强
optimizer steps
crops_seen
effective batch size
MICB token_dim=128
MICB topk=8
max_tokens_per_bag=256
min_instance_area=32
ROI alpha=0.1
mil_residual_scale=0
use_asymmetric_regions=false
```

P3 辅助损失保持：

```text
lambda_mil=0.25
lambda_exist_aux=0.25
lambda_severity_aux=0.25
lambda_intact_guard=0.05
lambda_coverage=0
```

不得在 A0/A1/A2/A3 之间同时更改关系 Token、层级反馈或可靠性门。

## 11. 保基线训练协议

固定 P3 作为冻结教师：

```text
Teacher=S8_MICB_DSMIL_P3
Teacher SHA256=45fb0e...
```

推荐训练阶段：

### 0–1000 optimizer steps

训练：

```text
SAR输入头（A0/A1为原P3 Patch Embedding，A2/A3为新输入头）
```

冻结：

```text
RGB Encoder
SAR Swin Stage1–4
Localization Head
fusion_layer_1/2/3
clf_2
全部BatchNorm统计
```

### 1000步以后

继续训练：

```text
新SAR输入头，lr=1e-4
```

选择性解冻：

```text
fusion_layer_3
clf_2
```

核心学习率：

```text
5e-7～1e-6
```

继续冻结：

```text
RGB Encoder
Localization Head
全部BatchNorm统计
SAR Swin Stage1–4
MICB/DSMIL Head
```

蒸馏建议：

```text
Damage logits KL：1.0
Localization logits KL：0.25
Fused feature consistency：0.1
```

定位分支完全冻结时，Localization KL 理论上接近零；代码仍保留权重
0.25 作为一致性保护，以暴露空间增强错位或冻结边界漂移。

## 12. 训练预算

### Smoke

```text
optimizer steps=20
只检查有限值、显存、形状、梯度和严格加载
```

### Pilot

```text
optimizer steps=3000
eval every=500
checkpoint every=1000
```

### Candidate

```text
optimizer steps≤10000
连续两次验证低于基线0.3 pp时提前停止
```

一张 RTX 5090 推荐：

```text
train batch size=4
accumulation=4
effective batch size=16
```

## 13. 指标与验收门槛

### 主指标

```text
Final mIoU
Valid 3-class mIoU
Damaged IoU
Destroyed IoU
Event-mean mIoU
```

### 安全指标

```text
Localization Precision/Recall/F1/IoU
Damaged → Intact
Destroyed → Intact
Background IoU
Intact IoU
```

### Pilot最低门槛

相对于同协议 P4-A0：

```text
Final mIoU下降不超过0.3 pp
Localization F1下降不超过0.2 pp
Damaged/Destroyed平均IoU不下降
Event-mean mIoU下降不超过0.3 pp
无NaN/Inf
严格加载无未解释的missing/unexpected keys
```

### 候选模型门槛

单种子：

```text
Final mIoU提升≥0.2 pp
或
Valid 3-class mIoU提升≥0.3 pp
```

最终模型：

```text
3个种子平均Final mIoU提升≥0.3 pp
Damaged和Destroyed不能出现系统性退化
事件级最差结果不能明显恶化
```

测试集只允许最终选定模型访问一次。

## 14. 实验登记表

| 日期 | 实验名 | 变体 | 母模型SHA | Seed | Steps | Val Final mIoU | Val Valid mIoU | Dmg IoU | Des IoU | Loc F1 | Event mIoU | 状态 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2026-07-28 | `S8_MICB_DSMIL_P3` | 母模型 | `45fb0e...` | 3407 | 4000 | 69.5772 | 60.5019 | 43.0309 | 62.8957 | 90.5704 | 52.7280 | LOCKED | 当前实际P3 |
|  | `S8P4_A0_PseudoRGB_MICB_base45fb_seed3407` | A0 | `45fb0e...` | 3407 |  |  |  |  |  |  |  | IMPLEMENTED | 待服务器预检；同训练协议对照 |
|  | `S8P4_A1_SARStatsRepeat3_MICB_base45fb_seed3407` | A1 | `45fb0e...` | 3407 |  |  |  |  |  |  |  | IMPLEMENTED | 待服务器预检；只改SAR统计 |
|  | `S8P4_A2_NativePatch1C_MICB_base45fb_seed3407` | A2 | `45fb0e...` | 3407 |  |  |  |  |  |  |  | IMPLEMENTED | 待服务器预检；原生单通道Patch |
|  | `S8P4_A3_NativeConvStem1C_MICB_base45fb_seed3407` | A3 | `45fb0e...` | 3407 |  |  |  |  |  |  |  | IMPLEMENTED | 待服务器预检；三层卷积Stem |

## 15. 每次实验必须保存

```text
config.json
checkpoint_identity.json
base_checkpoint_sha256.txt
sar_train_stats.json
train.log
metrics.jsonl
best_val_miou.pth
latest_resume.pth
final.pth
training_summary.json
event_metrics.csv
per_class_metrics.csv
```

其中 `event_metrics.csv` 和 `per_class_metrics.csv` 属于最终官方评估
产物；Pilot 训练器先保存 `training_summary.json`，连续控制器另外生成
`ablation_pilot3000_seed3407.json/.tsv`。

实验目录示例：

```text
/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/
S8P4_A2_NativePatch1C_MICB_base45fb_seed3407/
```

## 16. 迁移检查表

- [ ] P3 母模型 SHA256 为 `45fb0e...`
- [ ] P3 严格加载，无未解释键
- [ ] RGB 分支权重完整迁移
- [ ] Localization Head 权重完整迁移
- [ ] MICB/DSMIL Head 权重完整迁移
- [ ] `mil_residual_scale=0`
- [ ] SAR 输入为真实 `B×1×H×W`
- [ ] 未执行三通道复制
- [ ] 未使用 ImageNet RGB mean/std 归一化 SAR
- [ ] SAR 统计只由训练集计算
- [ ] 验证/测试读取同一冻结统计文件
- [ ] A0/A1/A2/A3 的训练步数和 crops_seen 相同
- [ ] BatchNorm 统计按协议冻结
- [ ] 测试集未参与模型选择

## 17. 结果结论模板

### 正向结果

```text
在相同 P3 母模型、训练预算、随机种子和 MICB-DSMIL 配置下，
P4-A2 将 SAR 从三通道复制与 ImageNet 归一化改为原生单通道
Patch Embedding。相对同训练协议 P4-A0，验证 Final mIoU 提升
X.XX pp，Damaged/Destroyed IoU 分别变化 X.XX/X.XX pp，
Localization F1 变化 X.XX pp。结果支持“消除伪 RGB 输入偏差能够
改善灾后 SAR 损毁表征”的假设。
```

### 无效结果

```text
P4-A2 未超过同训练协议 P4-A0，说明当前数据中的主要限制不能仅由
单通道 Patch Embedding 解决。需要检查 SAR 物理量、训练集统计、
权重迁移和输入尺度，而不能直接得出原生 SAR 建模无效。
```

### A3相对A2

```text
在原生单通道和 SAR 专用归一化均固定的条件下，P4-A3 相对 P4-A2
的变化衡量双层卷积局部散射建模的独立贡献，不能与 A0 直接替代
归一化和输入通道两项变量混合解释。
```

## 18. 论文方法表述

推荐表述：

> 在 S8 MICB-DSMIL 建筑级多实例辅助监督框架上，我们进一步消除灾后单通道 SAR 被复制为伪 RGB 的输入偏差。模型使用由训练集 SAR 统计驱动的模态专用归一化，并通过独立单通道 Patch Projection 或逐级卷积 Stem 将 SAR 映射到 Swin 表征空间。该设计保留 P3 已验证的建筑先验 ROI、跨模态建筑 Token、DSMIL 聚合与 DamageExist/Severity 层级辅助监督，同时保持建筑级预测不直接回填像素 logits 的安全边界。

第一轮不得写成：

```text
融合R1、R4和原生SAR后得到提升
```

因为 A0–A3 只允许改变 SAR 输入。R1关系 Token 和 R4像素层级反馈必须在后续 B/C 组中独立验证。

## 19. 本地实现状态（2026-07-28）

当前已实现并完成本地语法检查：

| 文件 | 用途 |
|---|---|
| `model/DamageFormer_S8_P4_NativeSAR_MICB.py` | A0/A1/A2/A3 模型、P3 权重迁移、冻结边界 |
| `dataset/make_data_loader_p4_native_sar.py` | 同步空间增强、学生 SAR 视图、P3 教师视图 |
| `script/standard_ML/compute_S8_P4_sar_stats.py` | 仅用训练集计算并冻结 p01/p99/mean/std |
| `script/standard_ML/train_S8_P4_native_sar_micb.py` | P3 教师蒸馏、MICB 损失、验证、早停与断点 |
| `script/standard_ML/eval_S8_P4_native_sar_micb.py` | 训练后验证集/测试集严格评估 |
| `script/standard_ML/summarize_S8_P4_ablation.py` | 汇总 A0–A3 统一指标及相对 A0 差值 |
| `run/run_S8_P4_NSMICB_5090.sh` | 单变体预检、统计、Smoke、Pilot、Full |
| `run/run_S8_P4_NSMICB_sequence.sh` | A0→A3 容错连续控制器 |
| `tests/test_s8_p4_native_sar.py` | 输入形状、A2 等价迁移、A3 Stem、冻结边界合成测试 |

明确未实现于本轮模型：

```text
R1局部关系Token
R4像素层级反馈
可靠性门/质量门
```

### 19.1 四个变体的实际代码协议

| 变体 | 学生输入 | SAR 输入头 | P3 迁移 |
|---|---|---|---|
| A0 | 复制 3 通道 + ImageNet 归一化 | 原 P3 Patch Embedding | 全参数严格加载 |
| A1 | 训练集统计归一化后复制 3 通道 | 原 P3 Patch Embedding | 全参数严格加载 |
| A2 | 训练集统计归一化的原生 1 通道 | `Conv2d(1,96,k=4,s=4)+LN` | 三通道卷积权重按输入通道求和，其余严格加载 |
| A3 | 训练集统计归一化的原生 1 通道 | `1→32→64→96` 卷积 Stem | 只允许新 Stem 缺失；其余 P3 参数严格加载 |

A2 的权重求和保证：如果输入仍是同一个标量复制成三通道，则新旧
卷积响应严格相等。由于正式 A2 同时改用 SAR 训练集归一化，所以这
不代表 A2 整体输入与 A0 数值相等；它只消除了通道数变化造成的随机
初始化干扰。

### 19.2 冻结与蒸馏协议

```text
step 0–1000:
  只训练 SAR 输入头，lr=1e-4

step >1000:
  SAR 输入头继续训练，lr=1e-4
  只额外解冻 fusion_layer_3、clf_2，lr=1e-6

始终冻结:
  P3教师、RGB编码器、定位头、SAR Swin Stage1–4、MICB/DSMIL参数、
  fusion_layer_1/2、所有BatchNorm参数与运行统计
```

蒸馏损失：

```text
Damage logits KL        × 1.0
Localization logits KL  × 0.25
fused_roi 特征 MSE      × 0.1
```

定位分支只依赖冻结的 RGB 分支，因此 Localization KL 正常情况下应
接近 0；保留该项是为了及时发现数据对齐或冻结边界被破坏。

### 19.3 服务器运行顺序

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

# 1. 公共预检
bash run/run_S8_P4_NSMICB_5090.sh preflight

# 2. 只用训练集生成一次冻结SAR统计
SAR_REPRESENTATION=display \
bash run/run_S8_P4_NSMICB_5090.sh stats

# 3. A0基线身份保护；只评估，不训练
VARIANT=a0 bash run/run_S8_P4_NSMICB_5090.sh guard

# 4. A0-A3：各自Smoke成功后连续运行3000步Pilot
mkdir -p /root/shared-nvme/BRIGHT-train-logs/S8_P4_NSMICB
nohup bash run/run_S8_P4_NSMICB_sequence.sh pilot \
  > /root/shared-nvme/BRIGHT-train-logs/S8_P4_NSMICB/controller_pilot.log 2>&1 &
```

查看控制器：

```bash
tail -f /root/shared-nvme/BRIGHT-train-logs/S8_P4_NSMICB/controller_pilot.log
```

单独运行某个变体：

```bash
VARIANT=a2 bash run/run_S8_P4_NSMICB_5090.sh smoke
VARIANT=a2 bash run/run_S8_P4_NSMICB_5090.sh pilot
```

控制器行为：

- 所有变体开始前只执行一次公共预检和 SAR 统计检查；
- 每个变体先执行 20 步 Smoke；
- Smoke 成功后执行 3000 步 Pilot；
- 主实验失败后自动重试一次；
- 存在 `latest_resume.pth` 时自动续训；
- 存在 `final.pth` 时重新运行会跳过；
- 单个变体失败默认记录后继续；
- 每 1000 步保存编号 checkpoint 与 `latest_resume.pth`；
- 1000 步之后，连续两次完整验证低于 A0 基线 0.3 pp 时提前停止。

### 19.4 Pilot 后验证命令示例

```bash
VARIANT=a2
EXP=S8_P4_A2_pilot3000_seed3407
ROOT=/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB

CUDA_VISIBLE_DEVICES=0 python -u \
  script/standard_ML/eval_S8_P4_native_sar_micb.py \
  --variant "$VARIANT" \
  --checkpoint "$ROOT/$EXP/best_val_miou.pth" \
  --dataset_path dataset \
  --data_list_path dataset/splitname/standard_ML/val_set.txt \
  --split val \
  --sar_stats_json "$ROOT/sar_train_stats.json" \
  --output_json "$ROOT/$EXP/val_best.json" \
  --device cuda:0
```

第一轮只比较验证集 A0/A1/A2/A3。测试集仍保持锁定，直到根据验证集
选定最终结构后再访问。
