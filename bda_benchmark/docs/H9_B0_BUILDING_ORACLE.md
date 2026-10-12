# H9-B0：建筑级 P3/A2 互补上限审计

## 研究问题

H7 的像素级 Oracle 已证明 P3 与 H6-A2 存在互补，但它不能回答这些互补证据在“每栋建筑只能有一个损毁等级”的约束下还能保留多少。

H9-B0 只在 CVPRW26 公开的 697 张实例验证集上分析，不训练模型，也不访问 holdout/test。它回答两个问题：

1. **B0-GT 上限**：给定官方建筑实例边界，逐栋建筑汇总 P3/A2 的三类概率，再用真实标签选择正确专家。该结果只表示严重程度专家选择的理论上限。
2. **B0-Deploy 上限**：先使用预测实例边界，再与官方实例按 IoU 一对一匹配，最后计算专家选择上限。它同时暴露实例几何质量造成的损失。

默认的 Deploy 几何是 P3 建筑定位掩膜的四连通域，仅作为代理诊断。只有提供与标准 450 张验证集严格对应的统一 COCO 实例预测后，审计才允许给出“可以实现 H9-B1”的正式结论。

## 严格协议

- P3 和 H6-A2 均严格加载并全程冻结。
- 每栋建筑只产生一个 `intact / damaged / destroyed` 标签。
- 建筑标签来自建筑掩膜内三类概率均值，背景概率不参与严重程度选择。
- Oracle 只在 A2 正确且 P3 错误时切换到 A2；其他情况保持 P3。
- 官方实例从 `dataset/target_instance_level/*_instance_damage.json` 读取，或使用与其完全对应的 `cvprw26/data/instance_annotations/val.json`。
- 评估列表固定为 `cvprw26/data/splits/val_set.txt` 的 697 张公开实例验证样本。
- 不使用标准语义验证集450张：两套验证划分交集为0，而且标准450张中有366张属于实例挑战 holdout，其实例真值不公开。
- RGB/SAR读取器不依赖语义 `target/*.tif`；实例真值直接来自COCO JSON。
- 无 `test` 运行模式，输出固定记录 `test_accessed=false`。

## 服务器运行

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_H9_building_oracle.sh preflight
bash run/run_H9_building_oracle.sh smoke
```

Smoke 通过后运行完整 450 张验证审计：

```bash
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/H9_building_oracle
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/H9_B0_validation_$(date +%Y%m%d_%H%M%S).log"

nohup bash run/run_H9_building_oracle.sh val > "$LOG" 2>&1 &
echo $!
tail -F "$LOG"
```

默认输出目录：

```text
/root/shared-nvme/BRIGHT-eval-results/H9_building_oracle
```

其中包括总结果 JSON，以及事件、灾害类型和类别级 TSV。

## 使用真实实例预测

外部预测必须是 CVPRW26 公开697张验证集对应的 COCO results JSON，并与统一 COCO 标注文件共享同一组 `image_id`：

```bash
INSTANCE_COCO_JSON=/path/to/cvprw26_val_instances.json \
DEPLOY_PREDICTIONS_JSON=/path/to/cvprw26_val_predictions.json \
bash run/run_H9_building_oracle.sh val
```

逐图官方 JSON 可能重复使用局部 `image_id`，所以不能直接与一个外部预测 JSON 配对；运行入口会拒绝这种不安全组合。

## 决策规则

默认进入 H9-B1 的门槛为：

- B0-GT 建筑 Macro-F1 提升不低于 `1.0 pp`；
- B0-GT Damaged F1 提升不低于 `1.0 pp`；
- 至少 `10/14` 个事件的建筑 Macro-F1 有正收益；
- B0-Deploy 的语义 Valid mIoU 上限提升不低于 `0.5 pp`；
- Deploy 必须来自真实实例预测，而不是连通域代理。

若 GT 上限高而 Deploy 上限低，当前瓶颈是实例几何，应先改善实例分割；若二者都高，才实现 H9-B1 小型建筑级专家选择器；若 GT 上限本身不足，则停止这条路线。
