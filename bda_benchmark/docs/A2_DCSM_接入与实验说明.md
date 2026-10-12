# A2-DCSM 原生 SAR 专家升级说明

## 1. 核心目的

本阶段只回答一个问题：能否让原生单通道 SAR 专家提供更多 P3 没有的正确损毁证据，尤其提高 `Damaged IoU` 与跨灾害事件泛化。

本阶段不修改 P3，不训练实例 Query，不设计新的 P3/A2 路由器，也不在监督阶段叠加新的损失函数。只有当 A2 专家本身获得稳定提升后，才重新运行 P3/A2 Oracle 并决定是否恢复专家选择路线。

## 2. 方法不是论文模块拼盘

代码只吸收三类与当前失败证据直接对应的思想：

1. **SARDet-100K 的完整迁移思想**：问题不只在第一层，预训练、输入适配和下游深层投影必须形成连续迁移链。因此 M3 不只保存 SAR encoder，还迁移 Stage3/4 的 SAR common/unique projection neck。
2. **CROMA 的模态独立编码思想**：RGB 与 SAR 保持各自编码器；RGB 只在确认未变化的背景/完好建筑上提供语义坐标，不要求变化区跨模态一致。
3. **AnySat 的特征预测思想**：预训练预测被遮挡的深层表征而不是还原 8-bit SAR 像素，减少显示产品灰度重建与损毁语义的错配。

M2 的固定 Scharr 结构视图只是 SARDet/MSFA 风格结构增强的最小受控版本。由于 BRIGHT 当前输入是 8-bit display SAR，不是定标的 sigma0/gamma0，代码不会把它解释为物理散斑模型。

## 3. 受控实验

| 变体 | 起点 | 预训练 | 结构视图 | 迁移到监督阶段 |
|---|---|---|---|---|
| M0 | 锁定 H6-A2 | 无 | 无 | 原 A2，严格对照 |
| M1 | 锁定 H6-A2 | 损毁条件的掩码特征预测 | 无 | SAR encoder |
| M2 | 锁定 H6-A2 | 同 M1 | 固定 Scharr + 零初始化残差 | Adapter + SAR encoder |
| M3 | 锁定 H6-A2 | 同 M2 | 同 M2 | Adapter + encoder + Stage3/4 SAR projection neck |

预训练只有一个优化目标：`damage_conditioned_masked_feature_prediction`。

- 背景和完好建筑：EMA SAR target 可以少量吸收冻结 RGB 的语义锚点；
- Damaged/Destroyed：target 保持 SAR 自身表征，不与 RGB 强制对齐；
- predictor 与 EMA target 在预训练结束后丢弃；
- 监督训练继续使用既有 A2 损失，不加入预训练损失。

## 4. 数据与迁移保护

- 统计量、预训练和监督训练只读取 train split；
- 事件均衡采样，避免大事件支配梯度；
- 75% 训练裁剪聚焦建筑类别，Damaged/Destroyed 权重更高；
- 每个 pathway checkpoint 记录锁定 A2 的 SHA256；
- 监督训练只接受相同 A2 来源、相同变体和完全匹配的模块前缀；
- M0 的参数键和形状与 H6-A2 完全一致；
- 测试集不参与 checkpoint 选择。

## 5. 推荐执行顺序

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_A2_DCSM_5090.sh preflight
VARIANT=m0 bash run/run_A2_DCSM_5090.sh guard

# 自动依次执行 M1/M2/M3：预训练 Smoke → 正式预训练 → 监督 Smoke → Pilot
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/A2_DCSM
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/controller_pilot_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  VARIANTS="m1 m2 m3" \
  PRETRAIN_STEPS=5000 \
  MAX_RETRIES=1 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_A2_DCSM_sequence.sh pilot \
  > "$LOG" 2>&1 &

echo $!
tail -F "$LOG"
```

单独运行 M2：

```bash
VARIANT=m2 PRETRAIN_STEPS=5000 bash run/run_A2_DCSM_5090.sh pretrain
VARIANT=m2 PRETRAIN_STEPS=5000 bash run/run_A2_DCSM_5090.sh smoke
VARIANT=m2 PRETRAIN_STEPS=5000 bash run/run_A2_DCSM_5090.sh pilot
```

## 6. 独立验证与最终测试

先在验证集锁定候选：

```bash
EXP=/root/shared-nvme/BRIGHT-checkpoints/A2_DCSM/A2_M3_pilot3000_seed3407
CKPT="$EXP/best_val_miou.pth"

DCSM_CKPT="$CKPT" bash run/run_A2_DCSM_5090.sh eval_val
```

只有验证候选锁定以后运行一次测试：

```bash
DCSM_CKPT="$CKPT" bash run/run_A2_DCSM_5090.sh eval_test
```

评估 JSON 除总体指标外还保存每个事件以及 flood/conflict/hurricane/wildfire/volcano/earthquake/explosion 等灾害类型的宏平均结果。

## 7. 决策规则

先比较 M0/M1/M2/M3，不能只看总 mIoU：

- `Damaged IoU` 是第一主指标；
- `Valid mIoU`、`Destroyed IoU` 不应明显下降；
- `Event mean` 与 `Event worst` 用于判断跨事件稳定性；
- 查看各灾害类型，确认收益不是由 volcano/explosion 等少数事件独占；
- 至少两个种子重复后再宣称稳定提升。

建议进入 Full 的最低条件：相对锁定 A2，`Damaged IoU >= +0.5 pp`，`Valid mIoU >= +0.3 pp`，且 `Event mean` 不下降；若只是 Destroyed 上升、Damaged 下降，则仍然没有解决当前核心问题。

若 M1 有效，说明主要矛盾是预训练任务错配；若 M2 继续提升，说明局部结构视图有增量价值；若只有 M3 提升，说明过去的主要断点是 encoder 与深层融合投影没有共同迁移。若三者均无效，应停止继续堆输入结构，转向增强 A2 的建筑级表征或可靠性建模。

## 8. 主要依据与官方代码

- SARDet-100K / MSFA（NeurIPS 2024）：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html>；<https://github.com/zcablii/SARDet_100K>
- CROMA（NeurIPS 2023）：<https://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html>；<https://github.com/antofuller/CROMA>
- AnySat（CVPR 2025）：<https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html>；<https://github.com/gastruc/AnySat>
- BRIGHT Challenge 2026：<https://arxiv.org/abs/2607.22746>；<https://github.com/ChenHongruixuan/BRIGHT>
