# H5-BIQ：建筑实例 Query RGB–SAR DamageFormer

## 1. 实验目标

H5 不再增加新的监督损失，而是把损毁预测单位由独立像素改为建筑实例。

当前 H4-A4 的主要误差是：

- damaged 被预测为 intact；
- 同一栋建筑内部出现多个严重程度；
- stage3/stage4 的 SAR 门控接近全局常数。

H5 使用冻结的 RGB 定位结果生成建筑实例，并在每栋建筑内部聚合 RGB/SAR
证据。RGB 编码器、定位解码器和定位头继续冻结。

## 2. 受控变体

| 变体 | 建筑实例 Query | stage3/4 稀疏可变形采样 | 新增监督损失 |
|---|---:|---:|---:|
| Q0 | 否 | 否 | 否 |
| Q1 | 是 | 否 | 否 |
| Q2 | 是 | 是 | 否 |

Q0 是 H4-A4 的严格 checkpoint 视图。Q1/Q2 新模块的输出投影均为零初始化，
因此加载同一个 A4 权重后，初始定位和损毁输出都必须与 Q0 完全一致。

## 3. H5-A / Q1：建筑实例 Query

1. 冻结 RGB 定位分支输出建筑概率。
2. 在低分辨率定位图上执行四连通域，过滤面积小于 4 个特征像素的区域。
3. 每张图最多保留 256 个建筑区域，并按面积确定稳定顺序。
4. 对每栋建筑分别池化：
   - damage decoder 特征；
   - RGB stage3/stage4 特征；
   - SAR stage3/stage4 特征。
5. Query 之间执行一次轻量 self-attention。
6. 每个 Query 输出 intact/damaged/destroyed 三维残差。
7. 同一 Query 的残差广播回整栋建筑，背景保持不变。
8. 验证和推理使用 `instance_query_output_dam`：在每个建筑内部池化
   基础 logits 后加上 Query 残差，并把同一个类别回填整栋建筑。因此同一
   实例的最终类别严格唯一；实例外继续使用原像素预测。

最终仍使用原有四类像素监督，因此没有加入 Query 分类损失、Hungarian matching
损失或额外蒸馏损失。

可微训练路径 `output_dam` 在初始化时严格等价 A4；严格实例解码单独作为
验证/推理输出。训练器检测到该输出后会自动用它计算验证指标和选择 checkpoint。

模型预留了 `instance_map` 参数。当前训练和验证默认使用冻结 RGB 定位图的连通
域；以后可直接接入 S9 center/offset 或 watershed 实例图，而不改 BIQ 主体。

## 4. H5-B / Q2：深层实例条件稀疏采样

Q2 只在 stage3 和 stage4 工作：

1. 对每栋建筑池化 RGB、SAR 和绝对差异；
2. 实例 Query 预测 4 个二维 SAR 采样偏移和对应权重；
3. 偏移经过 `tanh`，最大不超过 2 个特征像素；
4. 对每个建筑像素仅采样这 4 个 SAR 候选位置；
5. 融合残差只回填到建筑区域；
6. 最终投影零初始化，初始模型严格等价 A4。

这不是重复 H3 的局部 top-k correspondence。H3 在所有尺度、每个像素的固定
3×3 邻域内选择 top-k；Q2 只在深层建筑区域工作，候选位置由实例 Query 动态
预测。

## 5. 训练策略

- 来源权重：H4-A4 的 `best_val_miou.pth`；
- 前 1000 optimizer steps：只训练 H5 新模块；
- 之后：
  - H5 新模块学习率 `1e-4`；
  - SAR stage3/4 学习率 `1e-5`；
  - 原有损毁融合路径学习率 `1e-6`；
- RGB 编码器、RGB 定位解码器和定位头全程冻结；
- 有效 batch size 为 16；
- 默认 Pilot 为 3000 optimizer steps；
- 每 500 步验证；
- 只保留 `best_val_miou.pth`、`best_event.pth`、`best_balance.pth` 和
  `latest_resume.pth`。

## 6. 上传后的预检

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_H5_BIQ_5090.sh preflight
VARIANT=q0 bash run/run_H5_BIQ_5090.sh guard
```

Q0 guard 必须在完整 450 张验证集上复现 A4。脚本会直接读取来源 checkpoint
内的 metrics 比较，不依赖手工填写的基线数值。

## 7. Pilot

```bash
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/H5_BIQ
mkdir -p "$LOG_ROOT"

LOG="$LOG_ROOT/controller_q1_q2_pilot_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  VARIANTS="q1 q2" \
  MAX_RETRIES=1 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_H5_BIQ_sequence.sh pilot \
  > "$LOG" 2>&1 &

echo $!
echo "$LOG"
tail -f "$LOG"
```

控制器顺序执行：

```text
共享预检
→ Q0完整验证guard
→ Q1 smoke20
→ Q1 pilot3000
→ Q2 smoke20
→ Q2 pilot3000
→ 自动生成验证对比
```

## 8. Pilot 判定

相对 Q0/A4，候选至少满足：

- Valid mIoU 提升 `0.15 pp`；
- Damaged IoU 提升 `0.30 pp`；
- Event mean 下降不超过 `0.10 pp`；
- Localization F1 下降不超过 `0.01 pp`。

Q1 通过才能说明“改变预测单位”有效。Q2 必须在 Q1 之上继续改善，否则拒绝
稀疏可变形模块并保留 Q1。

不得在 Pilot 结束前访问 Test 集。

## 9. 迁移检查表

- [ ] A4 来源 checkpoint 的 `variant` 为 `a4`
- [ ] Q0 guard 完整验证严格通过
- [ ] Q1/Q2 初始 `output_loc` 与 A4 一致
- [ ] Q1/Q2 初始 `output_dam` 与 A4 一致
- [ ] 训练日志显示 `New losses: NONE`
- [ ] RGB/Localization 显示 `FROZEN / FROZEN`
- [ ] Q2 只声明 `stage3+stage4`
- [ ] 前 1000 步 phase 为 `new_modules_only`
- [ ] 每个实验目录相互独立
- [ ] Test 集未访问
