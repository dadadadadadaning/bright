# H5-B/C：稀疏跨模态融合与 RGB–SAR 联合预训练

## 1. 实验目的

本轮不再沿用 H5-Q1/Q2 的硬实例严重程度回填。Q1/Q2 已证明低分辨率
RGB 连通域并不等同于真实建筑实例，硬回填会在训练前直接损失约 0.78 pp
Valid mIoU。

本轮只验证两个相互独立的问题：

1. 建筑区域内的稀疏跨模态采样本身是否有效；
2. RGB–SAR 联合特征预训练是否优于 SAR-only 像素重建。

两条路线都以锁定的 H4-A4 step 5500 为起点，不增加正式监督训练损失，
不访问测试集。

## 2. 变体定义

| 变体 | 改动 | 实例 Query | 硬回填 | 新监督损失 |
|---|---|---|---|---|
| B0 | H4-A4 严格对照 | 无 | 无 | 无 |
| B1 | 建筑内 stage 3/4 K 点稀疏 SAR 采样 | 仅用连通域控制采样 | 无 | 无 |
| C1 | 加载 RGB/SAR 联合预训练的 stage 3/4 对称适配器 | 无 | 无 | 无 |

当前不运行 B1+C1。只有 B1、C1 至少有一条独立通过后，才建立组合变体，
以免无法归因。

## 3. H5-B1 的结构

```text
冻结 RGB 定位特征
        ↓
预测建筑概率 → 连通域只作为采样范围
        ↓
stage 3/4 每栋建筑聚合 RGB、SAR 和差异描述
        ↓
每栋建筑预测 4 个 SAR 采样偏移和对应权重
        ↓
grid_sample 取得少量 SAR 候选特征
        ↓
零初始化残差写回原 A4 joint feature
        ↓
原 A4 像素级层级损毁头
```

关键约束：

- stage 1/2 完全不改；
- RGB 定位分支只读取原始 RGB 特征；
- 采样残差只在建筑连通域内有效；
- 输出投影零初始化，因此 B1 加载 A4 后初始预测严格等价；
- 不输出 `instance_query_output_dam`，验证始终使用标准 `output_dam`。

该设计借鉴了 Deformable DETR 的少量采样点思想和 CVPR 2025 Keep the
Balance 的动态稀疏跨模态融合方向，但没有复制它们的完整检测器或
RGB+X 框架。

## 4. H5-C1 的两阶段结构

### 4.1 独立联合预训练

```text
冻结 A4 RGB 编码器 ─→ stage 3/4 RGB 适配器 ─┐
                                             ├─ 特征空间预训练
冻结 A4 SAR 编码器 ─→ stage 3/4 SAR 适配器 ─┘
```

预训练只使用训练集，并执行：

- 背景、完好建筑：跨模态特征对齐；
- 受损、摧毁：合并成一个 changed 集合，只要求保留模态差异，不将两类
  作为不同监督目标；
- 完好建筑：随机遮挡后进行 RGB→SAR、SAR→RGB 的特征空间预测；
- 方差约束：避免对齐表示塌缩。

预训练权重固定为：

```text
未变化对齐       1.00
建筑特征预测     0.50
变化区域独特性   0.50
方差约束         0.05
```

建筑特征预测使用余弦距离，避免高维归一化特征逐通道 Smooth-L1 被
`1/C` 缩小。日志同时报告 unchanged/changed cosine，相比只看总损失，
可以直接判断变化区域是否被错误地拉向共同表示。

这是任务感知预训练，不是严格无监督预训练。训练标签只用于划分
background/intact/changed 区域，严重程度标签不作为预训练分类目标。

预训练完成后只保存 stage 3/4 RGB/SAR 适配器。临时投影器和预测器被丢弃。

### 4.2 正式监督微调

C1 加载上述适配器后，继续使用 A4 原有的 CE、Lovasz、SAR 辅助、表示、
流场和层级损失。联合预训练损失不会进入监督训练，因此 C1 的变量是
“初始化和特征接口”，不是“继续堆损失函数”。

## 5. 参考依据

- Keep the Balance, CVPR 2025：动态稀疏跨模态融合与对称模态适配。
  <https://openaccess.thecvf.com/content/CVPR2025/html/Cai_Keep_the_Balance_A_Parameter-Efficient_Symmetrical_Framework_for_RGBX_Semantic_CVPR_2025_paper.html>
- CROMA, NeurIPS 2023：雷达/光学独立编码和联合表示预训练。
  <https://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html>
- AnySat, CVPR 2025：多传感器 JEPA 和特征空间预测。
  <https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html>
- CroCo, NeurIPS 2022：跨视图遮挡补全。
  <https://papers.nips.cc/paper_files/paper/2022/hash/16e71d1a24b98a02c17b1be1f634f979-Abstract-Conference.html>
- Deformable DETR, ICLR 2021：围绕参考位置使用少量动态采样点。
  <https://arxiv.org/abs/2010.04159>

## 6. 上传文件

将以下本地文件上传到服务器同名目录：

```text
model/DamageFormer_RGBSAR_H5_BC.py
script/standard_ML/pretrain_H5_C_joint_rgbsar.py
script/standard_ML/train_DamageFormer_RGBSAR_H5_BC.py
script/standard_ML/summarize_H5_BC_results.py
script/standard_ML/train_DamageFormer_RGBSAR_H4.py
run/run_H5_BC_5090.sh
run/run_H5_BC_sequence.sh
tests/test_damageformer_rgbsar_h5_bc.py
docs/H5_BC_稀疏融合与联合预训练实验说明.md
```

## 7. 推荐运行顺序

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

export SOURCE_A4_CKPT=/root/shared-nvme/BRIGHT-checkpoints/H5_BIQ/source/H4_A4_step5500_valid602784_locked.pth

bash run/run_H5_BC_5090.sh preflight

SOURCE_A4_CKPT="$SOURCE_A4_CKPT" \
VARIANT=b0 \
bash run/run_H5_BC_5090.sh guard

SOURCE_A4_CKPT="$SOURCE_A4_CKPT" \
bash run/run_H5_BC_5090.sh joint_smoke
```

预检、B0 guard 和联合预训练 smoke 都通过后，可由一个控制器连续执行
B1、C1：

首次运行建议先做 500 步联合预训练诊断：

```bash
JOINT_PRETRAIN_STEPS=500 \
bash run/run_H5_BC_5090.sh joint_pretrain
```

确认 `unchanged_cosine_similarity` 上升、`changed_cosine_similarity` 未持续
上升且建筑特征预测损失正常下降后，再续到 3000 步：

```bash
JOINT_PRETRAIN_STEPS=3000 \
bash run/run_H5_BC_5090.sh joint_pretrain
```

脚本会从同一目录的 `latest_resume.pth` 自动续训。

```bash
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/H5_BC
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/controller_b1_c1_pilot_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  SOURCE_A4_CKPT="$SOURCE_A4_CKPT" \
  VARIANTS="b1 c1" \
  MAX_RETRIES=1 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_H5_BC_sequence.sh pilot \
  > "$LOG" 2>&1 &

echo $!
tail -f "$LOG"
```

控制器会：

1. 统一预检；
2. 完整验证 B0 是否复现 A4；
3. 运行 C1 联合预训练 Smoke；
4. 若最终适配器不存在，完成一次 3000 步联合预训练；
5. B1/C1 各运行 20 步 Smoke；
6. Smoke 成功后各运行 3000 步 Pilot；
7. 每个失败实验自动重试一次，单个失败默认不阻断另一个；
8. 每个训练目录只保留 best 权重和覆盖式 latest resume，不累计里程碑权重。

## 8. 晋级标准

A4 锁定基线：

```text
Valid mIoU   60.2784
Final mIoU   69.3559
Damaged IoU  41.4082
Destroyed    64.5644
Event mean   52.2208
Event worst  40.4959
Loc F1       89.8602
```

建议 Pilot 晋级条件：

- Valid mIoU 至少高于 A4 0.10 pp；
- Event mean 不下降；
- Damaged/Destroyed 不能通过明显牺牲另一类来换取提升；
- Loc F1 应与 A4 基本一致；
- B1 的 stage 3/4 offset、residual 必须非零且有限；
- C1 必须严格加载联合预训练来源，不能静默漏载适配器。

未通过时不运行 Full/Test。

## 9. 后续可探索但本轮不加入的方案

1. **LoFTR 式粗到细对应**：只在 RGB 建筑内先做 stage 4 粗匹配，再在
   stage 3 局部细化。适合 B1 若动态偏移仍不稳定。
2. **xMUDA 式双分支互模拟**：RGB/SAR 保持独立分类头，仅在高置信区域
   相互学习。它会引入新训练目标，应作为单独的 H6，不与当前 B/C 混合。
3. **模态专家路由**：借鉴其他多模态任务的 MoE，让 intact、纹理破坏、
   结构破坏分别选择不同专家。当前训练集较小，必须先从 2–3 个低秩专家
   和负载监控开始，不能直接使用大规模 MoE。
4. **缺失模态/质量退化训练**：随机降低 SAR 质量或遮挡某一模态，测试
   模态偏置和鲁棒性。它更适合 B/C 确认有效后的稳健性实验。
