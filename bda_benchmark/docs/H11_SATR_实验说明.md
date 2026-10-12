# H11-SATR：SAR 表征迁移受控实验

## 1. 为什么不是把 H10-H3 继续训练到 5000 步

H10-H3 的测试结果已经显示：主要错误来自中度损毁没有通过
`Intact/Affected` 判断，而不是 `Damaged/Destroyed` 严重度头不够强。
同时 H10-H3 的最佳权重在 2000 步，继续延长同一结构并不能解决输入表征
缺少弱损毁证据的问题。

H11 因此只改变一个变量：**SAR 深层表征的来源**。RGB 定位、H6-A2
原生 SAR 专家、跨模态融合和四分类头均保持冻结。

## 2. 受控实验分别回答什么

| 变体 | 新增内容 | 可训练内容 | 回答的问题 |
|---|---|---|---|
| T0 | 无 | 无（仅验证） | 新代码是否严格复现 H6-A2 |
| T1 | 冻结 SARMAE ViT-B，注入 Stage3/4 | 两个投影适配器和两个零初始化系数 | 外部 SAR 表征本身是否提供互补证据 |
| T2 | 与 T1 相同 | T1 参数 + 预热后微调 ViT 最后两层 | 有限域适配是否优于完全冻结 |
| T1E | 官方三通道 PatchEmbed 与官方光度预处理，SARMAE 冻结 | 两个投影适配器和两个零初始化系数 | 修复输入契约后，预训练表征是否有效 |
| T2E | 与 T1E 相同 | T1E 参数 + 预热后微调 ViT 最后两层 | 严格输入契约下是否仍需有限域适配 |

第一轮不加入新的严重度头、不更改采样协议、不增加损失函数，也不访问测试集。

## 3. 模型数据流

```text
灾前 RGB ──> 冻结 P3 RGB 定位与 H6-A2 融合路径 ───────────────┐
                                                               ├─> 原 H6-A2 四分类头
灾后SAR ch0（训练集统计归一化）─> 冻结 H6-A2 NativeSARPyramid ─> Stage3/4 ─┤
       │                                                                    │
灾后SAR ch1（原始显示值）─> /255 ─> 复制3通道 ─> ImageNet归一化              │
       └─> 官方3ch SARMAE ViT-B ─> Block8/11 ─> 1×1+3×3投影 ─> α3/α4 ─────┘
```

实际公式为：

```text
SAR_stage3 = A2_stage3 + tanh(alpha3) * adapter3(SARMAE_block8)
SAR_stage4 = A2_stage4 + tanh(alpha4) * adapter4(SARMAE_block11)
```

`alpha3=alpha4=0`，因此初始化时残差严格为零。该安全设计用于排除
“换主干导致基线被破坏”的混杂因素。

## 4. 训练策略

### T1E：先修复输入契约，再检查冻结预训练特征

- 保留官方三通道 PatchEmbed，不再做通道平均；
- 使用 H4 数据契约携带的未裁剪显示域 SAR，复现官方灰度复制和 ImageNet 光度归一化；
- 为保留密集预测空间分辨率，不把 640×640 裁剪强制缩小到 224×224，位置编码使用双三次插值；
- SARMAE 全部冻结；
- 只训练 Stage3/4 投影和 `alpha3/alpha4`；
- 先做 20 步 Smoke，再做 1500 步 Screen；
- 不直接跑 5000 步。

### T2E：只有 T1E 通过后才运行

- 前 500 步仍只训练投影和接入系数；
- 500 步后以 `1e-5` 微调 SARMAE Block10/11 与最终 LayerNorm；
- 其余 SARMAE 层、完整 H6-A2、RGB/定位路径继续冻结；
- 有效批量保持 16，物理批量改为 1，避免 640×640 ViT 显存过高。

冻结不是性能限制本身，而是第一阶段的因果对照。T1 判断“预训练表征是否有
现成价值”，T2 才判断“有限微调是否必要”。如果一开始全部解冻，就无法知道
收益来自预训练还是重新拟合 BRIGHT 验证事件。

## 5. 服务器执行顺序

先准备官方 SARMAE ViT-B encoder 权重，并显式指定路径：

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

export SARMAE_CKPT=/path/to/official/SARMAE_ViT_B_checkpoint.pth

bash run/run_H11_SATR_5090.sh preflight
bash run/run_H11_SATR_5090.sh check_sarmae

VARIANT=t0 \
EXP_NAME=H11_T0_guard_seed3407 \
bash run/run_H11_SATR_5090.sh guard

VARIANT=t1e \
EXP_NAME=H11_T1E_smoke20_seed3407 \
bash run/run_H11_SATR_5090.sh smoke

VARIANT=t1e \
EXP_NAME=H11_T1E_screen1500_seed3407 \
bash run/run_H11_SATR_5090.sh screen
```

不要让脚本从不明 checkpoint 猜测结构。预检会验证 12 个 ViT-B encoder block、
PatchEmbed、位置编码和最终 LayerNorm 是否完整；T1E/T2E 保留官方三通道
PatchEmbed，decoder/head 会被明确丢弃。旧 T1 的通道均值迁移仅作为失败对照保留。

T1E 通过后再运行 T2E：

```bash
VARIANT=t2e \
EXP_NAME=H11_T2E_smoke20_seed3407 \
bash run/run_H11_SATR_5090.sh smoke

VARIANT=t2e \
EXP_NAME=H11_T2E_controlled5000_seed3407 \
bash run/run_H11_SATR_5090.sh pilot
```

## 6. 进入 T2E/测试集的门槛

以 H6-A2 验证结果为锁定基线。T1E Screen 至少满足：

1. `Damaged IoU` 不下降超过 0.10 pp；
2. `Damaged gate miss` 有下降趋势；
3. `Valid mIoU` 或 `Event mean` 至少一个提高，另一个不下降超过 0.05 pp；
4. 最佳步不是只能依靠零接入的初始 checkpoint；
5. `alpha3/alpha4` 至少一个稳定离开 0，但绝对值不应快速饱和。

只有验证锁定的最终候选允许做一次独立测试。测试仍然不参与权重选择。

## 7. 如何读取结果

独立验证：

```bash
EXP=/root/shared-nvme/BRIGHT-checkpoints/H11_SATR/H11_T1E_screen1500_seed3407
H11_CKPT="$EXP/best_val_miou.pth" \
EVAL_OUTPUT_JSON="$EXP/independent_val.json" \
bash run/run_H11_SATR_5090.sh eval_val
```

评估文件额外记录：

```text
effective_transfer_transitions.stage3
effective_transfer_transitions.stage4
affected_recall
damaged_gate_miss
conditional_destroyed_f1
```

如果接入系数接近零，结论是外部特征被模型拒绝，不是训练步数不足。如果接入
系数明显非零但验证/跨事件指标下降，说明迁移特征存在域偏差，应停止 T2，而不
是强制放大残差。

## 8. 与后续严重度模型的关系

H11 只解决“弱/中度损毁有没有被识别”的 SAR 表征问题。只有 H11 使
`Damaged gate miss` 稳定下降后，才考虑建筑内证据聚合或有序严重度头。这样论文
逻辑是连续的：

```text
SAR 专属预训练迁移
→ 提高弱损毁发生检测
→ 再研究建筑范围与严重程度
```

而不是同时更换 SAR 主干、融合模块和分类头，导致无法解释性能变化来源。
