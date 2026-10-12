# SEC：合成位移等变对应实验方案

## 1. 这次只解决什么问题

已有 D1 会在 RGB 建筑附近寻找 SAR 信息，但审计显示它的偏移输出几乎不会随人工位移改变。因此这次不再增加新的注意力层，而是给现有 D1 一个可核对的“位移答案”：

```text
灾前 RGB + 原始 SAR       -> clean offset
同一 RGB + 人工平移 SAR   -> shifted offset

要求：shifted offset - clean offset ≈ 人工加入的位移
```

监督的是两次偏移之差，不是假定原始 RGB 与 SAR 完全对齐。原始数据即使本来偏了 2 像素，也不会把这 2 像素错误地当成零。

## 2. 明确不重复的内容

| 已完成内容 | 本轮处理 |
|---|---|
| SALT-D1 8k | 冻结结果，作为共同初始化和参考，不重训 |
| D2/D3/D2R | 已未通过机制或类别安全检查，停止延长 |
| E1/E2/E3A/E3 | 保留为拼接/滤波/Cross-Attention消融证据，不重训 |
| P3 + fixed S2 | 保留当前最佳语义结果，不在SEC中修改 |
| LocCC、分水岭/RGID | 直接复用已有实现，不再重新写超像素或分水岭 |
| Metric-Aligned Utility M2/M3/M4 | 直接复用已有实验目录，不在SEC重复实现 |
| 事件均衡/LOEO | 等SEC结构选定后复用已有事件鲁棒Verifier |

## 3. 唯一变量消融

| 实验 | 与D1相比增加什么 | 它单独回答什么 |
|---|---|---|
| A0 = 已完成D1 | 无 | 当前局部对应基线 |
| SEC-S1 | 训练时同时看clean SAR和随机平移SAR；损毁损失取两者平均 | 仅靠位移增强能否提高稳定性 |
| SEC-S2 | S1 + relative offset-delta loss | 给出明确位移答案后，offset是否真的学会移动 |
| SEC-S3 | S2 + 建筑内clean/shift预测KL一致性 | 对应学会后，损毁结论能否对位移保持一致 |

S1非常重要：如果没有它，就无法区分S2的收益来自“多看了一张增强图”，还是来自“offset监督”。S1、S2、S3使用完全相同的D1初始权重、裁剪数、batch、优化器和随机种子。

## 4. 训练目标

共同分割损失：

```text
Lseg = Lloc(clean) + 0.5 * [Ldamage(clean) + Ldamage(shifted)]
```

S2增加：

```text
Loffset = SmoothL1(
  offset_shifted - stopgrad(offset_clean),
  [dx/stride, dy/stride]
)
```

只在GT建筑区域和移动后仍有效的区域计算。Stage2/Stage3的stride分别为8/16。

S3增加：

```text
Lconsistency = KL(
  stopgrad(Pdamage_clean),
  Pdamage_shifted
)
```

总损失：

```text
S1: Lseg
S2: Lseg + 0.05 * Loffset
S3: Lseg + 0.05 * Loffset + 0.10 * Lconsistency
```

新增损失在前500 optimizer steps线性升温，防止一开始破坏D1已有分类能力。

## 5. 数据和训练预算

- 只使用原有train列表训练、原有validation列表选型。
- 不读取official test，也不读取official instance labels。
- 人工位移：每个crop随机选择一个轴，方向为正/负，幅度为4或8像素；空出来的位置补0，不循环回卷。
- Optimizer：AdamW；learning rate `1e-4`；weight decay `5e-3`。
- Loss：CE + Lovasz-Softmax，加上述SEC损失。
- effective batch size：16（batch 4 × accumulation 4）。

| 阶段 | optimizer steps | 唯一训练crop | 用途 |
|---|---:|---:|---|
| smoke | 20 | 320 | 检查代码、显存、loss |
| probe | 5,000 | 80,000 | 在可控成本下判断收敛趋势并淘汰无效方案 |
| pilot | 8,000 | 128,000 | 三个变体完整比较 |
| short | 20,000 | 320,000 | 只延长pilot胜者，确认趋势 |
| final | 50,000 | 800,000 | 只训练最终方法和必要等预算对照 |

这里的step是optimizer step，不是DataLoader iteration。每个crop产生clean和shifted两个视图，但`unique_training_crops`不因此翻倍。

## 6. 普通精度证据与机制证据

每个模型都要做clean、x4、y4、x8、y8、x16、y16验证。

模型选择采用双检查点审计：`best_val_miou.pth`回答“最终可部署的最佳clean模型是什么”；`latest_resume.pth`回答“训练到当前步数后，offset机制究竟有没有学会”。如果best仍为step 0，不能把它视为新方法通过，而应继续审计latest来区分“机制没学会”和“机制学会但损害主任务”。

普通精度：

- valid mIoU；
- Damaged F1；
- Intact/Damaged/Destroyed IoU；
- 错误流向；
- 14事件逐事件mIoU。

机制证据：

- 人工位移后性能下降曲线；
- `offset_shifted-offset_clean`的观测值；
- 与已知`dx/stride, dy/stride`的MAE；
- x方向监督是否只主要改变x偏移，y方向同理。

## 7. 准入门槛

### 5k筛选

- clean mIoU相对D1不低于-0.30pp；
- Damaged F1不下降；
- 至少10/14事件不下降；
- x8/y8平均性能下降不大于D1；
- S2/S3的offset-delta MAE相对D1审计误差至少降低30%。

### 8k进入20k

满足上述全部条件。若S2通过而S3不通过，停止S3；若S1与S2精度相似但S2机制明显通过，优先S2。

### 20k进入50k

- clean mIoU相对D1至少+0.30pp；
- Damaged F1不下降；
- 至少10/14事件不下降；
- 事件bootstrap 95% CI下界大于0；
- x8/y8及x16/y16曲线不劣于D1。

没有通过机制检查的模型，即使mIoU偶然升高，也只能称为增强模型，不能宣称“解决RGB-SAR错位”。

## 8. 实际运行顺序

服务器代码同步后，在仓库根目录运行：

```bash
conda activate bright4090
cd /root/data/BRIGHT-master/bda_benchmark

bash experiments/sec_alignment_v1/run.sh preflight

for variant in s1 s2 s3; do
  bash experiments/sec_alignment_v1/run.sh smoke "$variant"
done

for variant in s1 s2 s3; do
  bash experiments/sec_alignment_v1/run.sh probe "$variant"
done

# 只做一次：不重训D1，仅用同一套新审计计算旧日志中没有的relative-delta指标
bash experiments/sec_alignment_v1/run.sh audit_reference

TRAIN_TAG=probe5000 bash experiments/sec_alignment_v1/run.sh audit s1
TRAIN_TAG=probe5000 bash experiments/sec_alignment_v1/run.sh compare s1
TRAIN_TAG=probe5000 bash experiments/sec_alignment_v1/run.sh audit s2
TRAIN_TAG=probe5000 bash experiments/sec_alignment_v1/run.sh compare s2
TRAIN_TAG=probe5000 bash experiments/sec_alignment_v1/run.sh audit s3
TRAIN_TAG=probe5000 bash experiments/sec_alignment_v1/run.sh compare s3
```

如果输出显示`checkpoint_step: 0`，不需要重新训练，追加审计5,000步latest权重：

```bash
for variant in s1 s2 s3; do
  AUDIT_WEIGHTS=latest TRAIN_TAG=probe5000 \
  bash experiments/sec_alignment_v1/run.sh audit "$variant"

  AUDIT_WEIGHTS=latest TRAIN_TAG=probe5000 \
  bash experiments/sec_alignment_v1/run.sh compare "$variant"
done
```

人工看完三个`*_admission.json`后，才把通过者续到8k：

```bash
bash experiments/sec_alignment_v1/run.sh pilot s2
TRAIN_TAG=pilot8000 bash experiments/sec_alignment_v1/run.sh audit s2
TRAIN_TAG=pilot8000 bash experiments/sec_alignment_v1/run.sh compare s2
```

不要写一个无条件把s1/s2/s3全部跑到50k的脚本。每一级准入就是为了提前停止无效路线。

## 9. SEC通过后再做什么

顺序固定为：

```text
SEC最佳主网络
  -> Occurrence/Severity层级头（如果Damaged仍是主要瓶颈）
  -> 事件均衡/GroupDRO（只在已选定结构上）
  -> 冻结主网络输出
  -> 复用现有RGID/LocCC区域生成
  -> 复用现有Metric-Aligned M2/M3/M4
  -> meta_confirm
  -> Val450
  -> 最后才访问Official Test
```

主网络与Verifier必须分别报告增益，然后才报告组合增益，不能只给最终组合结果。

## 10. S2R：S2的安全修复实验

S2已经证明模型能学到位移方向，但响应幅度弱，同时Damaged F1明显下降。S2R不增加大模块，只修改“谁可以学习”和“用什么损失学习”：

```text
冻结D1的RGB/SAR编码器、融合层、定位头、损毁解码器
                      ↓
只开放Stage-2/3匹配器的query和key投影
                      ↓
clean SAR + 已知平移SAR（±4/±8像素）
                      ↓
只计算相对位移误差，不用错位SAR监督损毁类别
```

推荐先运行5000步：

```bash
bash experiments/sec_alignment_v1/run.sh preflight
bash experiments/sec_alignment_v1/run.sh probe s2r

TRAIN_TAG=probe5000 AUDIT_WEIGHTS=latest \
  bash experiments/sec_alignment_v1/run.sh audit s2r
TRAIN_TAG=probe5000 AUDIT_WEIGHTS=latest \
  bash experiments/sec_alignment_v1/run.sh compare s2r
```

这里必须审计`latest`。因为`best_val_miou.pth`可能仍是未经训练的step 0；S2R的第一目标是验证位移机制，最终准入仍由clean mIoU、Damaged F1、事件一致性和位移误差共同决定。

## 11. 建筑定位与粗到细代码变体

| 变体 | 位移监督区域 | 匹配顺序 | 证明什么 |
|---|---|---|---|
| `s2r0` | 全图有效区域 | Stage-2/3独立 | 不聚焦建筑的安全对齐对照 |
| `s2r` | GT建筑，仅训练损失使用 | Stage-2/3独立 | 准确建筑监督是否有帮助 |
| `s2rp` | 冻结RGB定位头预测的soft建筑概率 | Stage-2/3独立 | 推理可获得的建筑定位是否足够 |
| `s2rr` | 预测soft建筑 + 16像素/pixel外环 | Stage-2/3独立 | 外环能否容纳错位SAR证据 |
| `s2rcf` | 预测soft建筑 + 外环 | Stage-3粗匹配→Stage-2残差 | 粗到细是否比两个尺度独立匹配更好 |

严格顺序：

```text
先完成 s2r
  ├─ 位移MAE未下降30%：停止，不跑更多support变体，先改匹配表达
  └─ 通过：运行s2r0/s2rp/s2rr，选择建筑support
                  ↓
          最佳support通过类别安全门槛
                  ↓
             才运行s2rcf
```

运行建筑support消融（每个先20步，再决定是否5000步）：

```bash
for variant in s2r0 s2rp s2rr; do
  bash experiments/sec_alignment_v1/run.sh smoke "$variant"
done

# 下面三条只在s2r通过后逐条运行，不要并发：
bash experiments/sec_alignment_v1/run.sh probe s2r0
bash experiments/sec_alignment_v1/run.sh probe s2rp
bash experiments/sec_alignment_v1/run.sh probe s2rr
```

只有建筑support胜者通过后才运行粗到细：

```bash
bash experiments/sec_alignment_v1/run.sh smoke s2rcf
bash experiments/sec_alignment_v1/run.sh probe s2rcf
TRAIN_TAG=probe5000 AUDIT_WEIGHTS=latest \
  bash experiments/sec_alignment_v1/run.sh audit s2rcf
TRAIN_TAG=probe5000 AUDIT_WEIGHTS=latest \
  bash experiments/sec_alignment_v1/run.sh compare s2rcf

# 直接证明粗到细优于相同building+ring support的独立匹配：
TRAIN_TAG=probe5000 AUDIT_WEIGHTS=latest \
  bash experiments/sec_alignment_v1/run.sh compare_pair s2rcf s2rr
```

上面审计命令中的环境变量必须是`TRAIN_TAG`；不要使用其他拼写。

## 12. 带准入门控的连续运行

`run_building_cfsa_pipeline.sh`使用新的输出根目录，不会因为旧`SEC_ALIGNMENT_V1`中的同名结果而跳过，也不会覆盖旧实验。它按顺序运行，并在关键实验未通过时自动停止。

只运行S2R筛选：

```bash
nohup bash experiments/sec_alignment_v1/run_building_cfsa_pipeline.sh screen \
  > /root/data/BRIGHT-train-logs/SEC_BUILDING_CFSA_V1_screen.log 2>&1 &
```

S2R通过后，继续support与粗到细：

```bash
nohup bash experiments/sec_alignment_v1/run_building_cfsa_pipeline.sh support \
  > /root/data/BRIGHT-train-logs/SEC_BUILDING_CFSA_V1_support.log 2>&1 &
```

从头连续运行完整的门控流程：

```bash
nohup bash experiments/sec_alignment_v1/run_building_cfsa_pipeline.sh full \
  > /root/data/BRIGHT-train-logs/SEC_BUILDING_CFSA_V1_full.log 2>&1 &
```

默认新目录：

```text
checkpoints: /root/data/BRIGHT-checkpoints/SEC_BUILDING_CFSA_V1
audits:      /root/shared-nvme/BRIGHT-eval-results/SEC_BUILDING_CFSA_V1
```
