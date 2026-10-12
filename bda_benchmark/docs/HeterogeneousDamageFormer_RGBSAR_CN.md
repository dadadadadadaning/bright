# Heterogeneous DamageFormer：从原始 DamageFormer 重建 RGB–SAR 路线

## 1. 结论

这条路线不是给原 DamageFormer 的第二个 Swin 更换一个单通道输入层，也不继承
P4、P5、C2-Recovery 或冻结教师。它从原始 DamageFormer 的任务分工重新设计：

- 灾前 RGB 单独负责建筑几何与定位；
- 灾后 SAR 使用独立的一通道层级主干；
- RGB 与 SAR 在四个尺度分别建立局部对应；
- 公共信息和模态独有信息分开表示；
- 损毁分类使用联合表征，定位输出不受 SAR 噪声直接影响；
- 训练和选模只访问训练集、验证集，测试集不进入训练脚本。

## 2. 原始代码为什么不适合直接改输入头

原始 `model/DamageFormer.py` 中两路都是 ImageNet Swin-T。数据加载器把单通道
SAR 复制三次，再使用 RGB 的 ImageNet 均值和标准差。两路多尺度特征各自拼接后，
跨模态关系只有：

```text
[RGB feature, pseudo-RGB SAR feature] -> concat -> ResBlock
```

这隐含了四个不成立的前提：

1. RGB 与 SAR 的通道统计相同；
2. RGB 纹理和 SAR 散射可以由同一种 ImageNet 表征解释；
3. 同一像素位置的两路特征可直接比较；
4. 两种模态中所有信息都应进入同一个共享空间。

所以仅改 `PatchEmbed(3 -> 1)` 只修复张量形状，没有修复表征、对应和融合。

## 3. 采用的研究依据

### CROMA（NeurIPS 2023）

CROMA 对雷达和光学使用两个独立的单模态编码器，并增加联合编码器，而不是把
SAR 当作伪 RGB。这里采用其“独立编码、随后联合建模”的结构原则。

- 论文：<https://openreview.net/pdf?id=ezqI5WgGvY>
- 官方代码：<https://github.com/antofuller/CROMA>

### DeCUR（ECCV 2024 Oral）

DeCUR 指出只学习跨模态公共空间会压制模态独有信息。这里在每个尺度把 RGB/SAR
特征分别投影为 common 和 unique；只在背景、完好建筑区域约束 common 对齐，
避免把真实损毁变化错误对齐掉。

- 论文：<https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04236.pdf>
- 官方代码：<https://github.com/zhu-xlab/DeCUR>

### CMX（RGB-X 官方实现）

CMX 使用双路层级主干，在每个阶段进行特征校正和融合，而不是只在网络末端拼接。
这里同样在四个尺度分别构造联合表征。

- 论文与官方代码：<https://github.com/huaaaliu/RGBX_Semantic_Segmentation>

### Keep the Balance（CVPR 2025）

该工作指出全局跨模态注意力会把其他位置的噪声引入当前特征，采用动态稀疏跨模态
融合。这里把全局注意力改成 3×3 局部对应，并在 H3 中只保留 top-3 关系。

- 论文：<https://openaccess.thecvf.com/content/CVPR2025/html/Cai_Keep_the_Balance_A_Parameter-Efficient_Symmetrical_Framework_for_RGBX_Semantic_CVPR_2025_paper.html>

本仓库实现是针对 BRIGHT 损毁任务的重新实现，没有复制上述仓库源码。

不能直接把 CROMA checkpoint 塞进当前模型：CROMA 的标准 SAR 输入是 Sentinel-1
双极化两通道，光学输入是 Sentinel-2 多光谱，而且预训练图像通常是同地点/近同时相；
当前数据是灾前 RGB 与灾后单通道显示 SAR，标签还包含真实变化。盲目强制全图对齐会
把 damaged/destroyed 当成“模态差异”消除。因此这里只采用其双单模态编码器和联合
编码器原则，并把 common 对齐限制在标签 0/1。

## 4. 新模型数据流

```text
灾前 RGB [B,3,H,W]                    灾后 SAR [B,1,H,W]
       │                                      │
 ImageNet Swin-T                         独立 SAR 层级主干
       │                                      │
 RGB 四尺度特征                         SAR 四尺度特征
       │                                      │
       ├──────── 每尺度 common / unique 分解 ─┤
       │                                      │
       └──── 3×3 局部对应；H3 动态 top-3 ─────┘
                              │
                        四尺度联合解码
                              │
                 RGB 定位软先验（只调制、不清零）
                              │
                     四类损毁分类输出

RGB 四尺度特征 ── 独立定位解码 ── 建筑/背景输出
```

SAR 主干使用 GroupNorm，不依赖大 batch 的 BatchNorm 统计，也没有 ImageNet RGB
输入层。其层级分辨率和通道数与 Swin-T 对应：

| Stage | 分辨率 | RGB 通道 | SAR 通道 |
|---|---:|---:|---:|
| 1 | 1/4 | 96 | 96 |
| 2 | 1/8 | 192 | 192 |
| 3 | 1/16 | 384 | 384 |
| 4 | 1/32 | 768 | 768 |

## 5. 受控消融

| 变体 | 输入与融合 | 研究问题 |
|---|---|---|
| H0 | 原始双 Swin、SAR 三通道复制、末端拼接 | 清理后的原始 DamageFormer 对照 |
| H1 | 独立一通道 SAR 主干、普通多尺度联合 | 仅消除伪 RGB 与共享主干假设是否有效 |
| H2 | H1 + common/unique + 3×3 局部对应 | 表征分解和局部对齐是否有效 |
| H3 | H2 + top-3 稀疏对应 + RGB 软建筑先验 | 完整异构融合是否优于 H2 |

实验顺序必须是 H0 → H1 → H2 → H3。不能只跑 H3，否则无法判断增益来自独立
主干、表征分解还是稀疏融合。

## 6. 损失

```text
L = Lloc_ce + 0.5 Lloc_lovasz
  + Ldamage_ce + 0.75 Ldamage_lovasz
  + 0.25 Lsar_aux
  + 0.10 Lrepresentation
```

`Lrepresentation` 包含：

- `common_unchanged_loss`：只对标签 0/1 的背景和完好建筑对齐 RGB/SAR common；
- `common_unique_orthogonality_loss`：降低同一模态 common/unique 冗余。

标签 2/3 的受损和摧毁像素不会受到 common 对齐约束。这一点与普通 RGB-X
语义分割不同，是从“跨时相损毁变化”任务本身得到的必要修正。

## 7. 训练协议

推荐先跑 3,000 步 Pilot，再决定是否跑 8,000 步 Full。

| 模式 | Optimizer steps | EBS | 总 crops | 用途 |
|---|---:|---:|---:|---|
| Smoke | 20 | 16 | 320 | 检查有限损失和保存 |
| Pilot | 3,000 | 16 | 48,000 | 消融筛选 |
| Full | 8,000 | 16 | 128,000 | 单卡 5090 统一比较 |
| Paper | 50,000 | 16 | 800,000 | 原论文尺度复现 |

优化器为 AdamW：

```text
RGB ImageNet 主干 LR = 1e-5
SAR 独立主干 LR      = 1e-4
联合融合与解码 LR    = 1e-4
Weight decay         = 5e-3
```

每 1,000 步使用完整验证集评估和保存 checkpoint。训练脚本没有测试集参数。

默认采用紧凑保存策略：

```text
best_val_miou.pth = 仅模型权重、最佳指标和配置
latest_resume.pth = 模型、AdamW、GradScaler和当前步数，覆盖更新
```

默认不再保留 `checkpoint_step_1000/2000/...pth`。只有显式设置环境变量
`KEEP_MILESTONE_CHECKPOINTS=1` 才会额外保存里程碑。写入失败时残缺的 `.tmp`
会自动删除。

## 8. 服务器运行

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

# 代码和合成测试
bash run/run_HeterogeneousDamageFormer_5090.sh preflight

# 只用训练集计算一次 SAR 统计
bash run/run_HeterogeneousDamageFormer_5090.sh stats

# 四个变体依次 Smoke，成功后依次跑 3000 步 Pilot
mkdir -p /root/shared-nvme/BRIGHT-train-logs/HeterogeneousDamageFormer
nohup env \
  VARIANTS="h0 h1 h2 h3" \
  MAX_RETRIES=1 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_HeterogeneousDamageFormer_sequence.sh pilot \
  > /root/shared-nvme/BRIGHT-train-logs/HeterogeneousDamageFormer/controller_pilot.log \
  2>&1 &

tail -f /root/shared-nvme/BRIGHT-train-logs/HeterogeneousDamageFormer/controller_pilot.log
```

若只单独运行 H3：

```bash
VARIANT=h3 bash run/run_HeterogeneousDamageFormer_5090.sh smoke
VARIANT=h3 bash run/run_HeterogeneousDamageFormer_5090.sh pilot
```

## 9. 进入 Full 的门槛

以 H0 Pilot 为同一训练协议基线，不使用历史 P3 作为结构相同的基线，因为 P3 已经
加入了后续 BGMP、MICB、DSMIL 等模块。

H1/H2/H3 满足以下条件才进入 Full：

1. 完整验证集 `valid_mIoU` 不低于 H0 0.3 pp 以上；
2. `Damaged IoU` 或 `Destroyed IoU` 至少一项明确改善；
3. 定位 F1 不发生异常下降；
4. `Intact -> Damaged` 不因 SAR 误检明显恶化；
5. 不出现 NaN/Inf，checkpoint 可 strict load；
6. 至少使用 3 个随机种子复核最终候选。

## 10. 文件清单

```text
model/DamageFormer_RGBSAR_Heterogeneous.py
dataset/make_data_loader_heterogeneous_rgbsar.py
script/standard_ML/compute_HeterogeneousSAR_stats.py
script/standard_ML/train_DamageFormer_RGBSAR_Heterogeneous.py
run/run_HeterogeneousDamageFormer_5090.sh
run/run_HeterogeneousDamageFormer_sequence.sh
tests/test_damageformer_rgbsar_heterogeneous.py
docs/HeterogeneousDamageFormer_RGBSAR_CN.md
```
