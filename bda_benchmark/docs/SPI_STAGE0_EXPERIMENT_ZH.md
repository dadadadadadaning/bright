# SPI Stage-0：语义到伪实例的同 Split 准入实验

## 1. 这一步回答什么

Stage-0 不训练模型，只回答：

> 在同一批 CVPRW26 公开验证图像上，如果建筑区域逐步从像素预测、简单连通域、RGID，提升到官方真实实例，P3 的建筑内概率池化能否稳定提高原始语义指标和建筑级损毁指标？

只有官方实例池化 E4 相对 Raw P3 的提升通过预注册门槛，才允许进入 T1 组件均衡联合训练。

本实验不会访问隐藏 Test，不修改 P3，不把真实实例当作部署输入。

## 2. 统一符号

- 图像像素域：`Omega`
- 类别：`c in {0,1,2,3}`，分别表示 Background、Intact、Damaged、Destroyed
- P3 logits：`z_c(x)`
- P3 概率：

  `p_c(x) = exp(z_c(x)) / sum_r exp(z_r(x))`

- P3 定位头输出的建筑掩膜：

  `B(x) = 1[argmax z_loc(x) = building]`

- 官方第 `i` 栋建筑掩膜和标签：`G_i`、`y_i`
- 某种几何方法生成的第 `j` 个区域：`R_j`

## 3. 六个比较实验

### E0：Raw P3

不做任何实例处理：

`Y_E0(x) = argmax_{c in {0,1,2,3}} p_c(x)`

它是主基线，所有增益都相对 E0 计算。

### E1：Binary CC

对 P3 定位建筑掩膜 `B` 做四邻域连通域：

`{R_j^bin} = CC4(B)`

在每个区域内汇总三个损毁类别的概率：

`s_j(c) = (1 / |R_j|) sum_{x in R_j} p_c(x), c in {1,2,3}`

`hat_y_j = argmax_c s_j(c)`

然后把整个 `R_j` 赋为 `hat_y_j`。区域外保持 Background。

作用：检查最简单的“整块建筑共享一个损毁等级”是否有效。

### E2：Damage-class CC

先得到 Raw P3 损毁类别，再分别做连通域：

`R_{c,k} = CC4(B and [Y_E0 = c]), c in {1,2,3}`

每个区域仍使用上述概率均值确定统一标签。

重要限制：两个接触建筑如果具有相同类别，仍然属于同一个组件。因此 E2 是 pseudo blob，不是可靠实例。

### E3：Learned RGID

使用 learned RGID 输出的实例掩膜 `R_j^rgid`，再使用相同的 P3 概率池化：

`hat_y_j^rgid = argmax_c mean_{x in R_j^rgid} p_c(x)`

E3 必须以和官方统一 COCO 验证集共享 `image_id` 的 COCO predictions JSON 输入。没有该文件时 E3 明确标记为 unavailable，不用连通域冒充 RGID。

### E4：Official GT instance pooling

使用官方真实实例 `G_i` 作为建筑区域，但类别仍来自 P3：

`hat_y_i^GT-P3 = argmax_c mean_{x in G_i} p_c(x)`

这是 Stage-0 最重要的上界：只给 P3 正确建筑归属，不给正确损毁标签。

E4 如果不能超过 E0，说明即使建筑几何完美，当前 P3 概率也不适合通过建筑池化改善语义，后续 T1–T5 缺乏依据。

### E5：Official GT instance + P3/A2 correctness oracle

先分别得到 P3 和 A2 的建筑标签：

`hat_y_i^P3 = argmax_c mean_{x in G_i} p_c^P3(x)`

`hat_y_i^A2 = argmax_c mean_{x in G_i} p_c^A2(x)`

然后只在 A2 正确且 P3 错误时切换：

`hat_y_i^oracle = hat_y_i^A2, if hat_y_i^A2 = y_i and hat_y_i^P3 != y_i`

其他情况保持 P3。E5 使用真实标签选择专家，只是总上界，不可部署，也不参与 Stage-0 通过判定。

## 4. 语义指标

对每个变体构造像素混淆矩阵：

`H_ab = sum_x 1[Y_gt(x)=a and Y_pred(x)=b]`

类别 `c` 的 IoU：

`IoU_c = H_cc / (sum_b H_cb + sum_a H_ac - H_cc)`

前景三类平均：

`valid-mIoU = (IoU_I + IoU_D + IoU_X) / 3`

Damaged F1：

`F1_D = 2 H_DD / (sum_b H_Db + sum_a H_aD)`

所有像素都来自同一批官方实例验证图像。不能拿标准450张语义验证结果与697张实例验证结果直接相减。

## 5. 建筑级指标

对每个官方 GT 建筑 `G_i`，从某个变体的最终语义图中进行多数投票：

`tilde_y_i = mode{Y_pred(x): x in G_i}`

这里 Background 也参与投票。如果一栋真实建筑大部分被预测为 Background，它会被记为 missed building，而不是从建筑评价中删除。

对 Intact、Damaged、Destroyed 分别计算 Precision、Recall、F1，再对 GT 中出现的类别取宏平均：

`Building-Macro-F1 = mean_c F1_c`

## 6. 实例几何指标

预测区域与 GT 实例按 IoU 阈值0.5做一对一贪心匹配：

`IoU(R_j,G_i) = |R_j intersect G_i| / |R_j union G_i|`

`Precision = TP / (TP + FP)`

`Recall = TP / (TP + FN)`

`F1 = 2PR / (P+R)`

Merge 与 Split 使用相对较小实例面积的覆盖率：

`O_ij = |G_i intersect R_j| / min(|G_i|, |R_j|)`

- 若同一 `R_j` 与至少两个 `G_i` 满足 `O_ij > 0.1`，记一次 Merge。
- 若同一 `G_i` 与至少两个 `R_j` 满足 `O_ij > 0.1`，记一次 Split。

## 7. Stage-0 决策公式

所有收益以百分点计算。例如：

`Delta_mIoU = 100 * (valid-mIoU_E4 - valid-mIoU_E0)`

默认五项门槛全部满足才通过：

1. `Delta_mIoU >= +0.5 pp`
2. `Delta_Building-Macro-F1 >= +1.0 pp`
3. `Delta_Damaged-F1 >= +1.0 pp`
4. 至少10个事件的 E4 Building-Macro-F1 不低于 E0
5. `Delta_Intact-IoU >= -0.5 pp`

通过输出：

`PROCEED_TO_T1_COMPONENT_BALANCED_JOINT_TRAINING`

失败输出：

`STOP_SEMANTIC_FEEDBACK_STUDY_KEEP_INSTANCE_ONLY`

E5 不参与通过判定，因为它使用真实类别。E3 也不决定是否进入 T1，因为 Stage-0 首先验证的是“实例结构是否存在语义价值”；但 E3 决定现有 RGID 距离 E4 上界还有多远。

## 8. 服务器运行

```bash
cd ~/shared-nvme/BRIGHT-master/bda_benchmark
conda activate bright5090

bash run/run_SPI_STAGE0_isolated.sh preflight
bash run/run_SPI_STAGE0_isolated.sh smoke
bash run/run_SPI_STAGE0_isolated.sh audit
```

默认输出：

```text
/root/shared-nvme/BRIGHT-eval-results/SPI_STAGE0_ISOLATED/
```

如已有 RGID COCO predictions：

```bash
INSTANCE_COCO_JSON=/path/to/public_val.json \
RGID_PREDICTIONS_JSON=/path/to/rgid_public_val_predictions.json \
bash run/run_SPI_STAGE0_isolated.sh audit
```

RGID predictions 必须是 COCO results 列表，每行至少包含：

```json
{
  "image_id": 1,
  "segmentation": {"size": [1024, 1024], "counts": "..."},
  "score": 0.93
}
```

## 9. Stage-0 通过后才实现的 T1

当前代码没有提前实现 T1，避免在假设未通过时继续增加复杂度。T1 的预注册形式为组件均衡语义损失。

对训练语义标签中类别 `c` 的第 `k` 个四连通组件 `R_ck`：

`L_cb = 1/N_R sum_{c=1}^3 sum_k [1/|R_ck| sum_{x in R_ck} -log p_c(x)]`

其中 `N_R` 是有效前景组件数。完整损失：

`L_T1 = L_semantic_original + lambda_cb L_cb`

它保留原来的像素语义损失与 Background 学习，只额外让每个前景组件拥有接近相等的总权重。T1 不声称组件等于真实实例。

如果 T1 通过，再依次加入：

### T2 Center

对 reliable 伪区域生成 Gaussian center target `h*(x)`，使用 CenterNet focal：

`L_center = -sum_x [(1-hat_h)^alpha log(hat_h), h*=1; (1-h*)^beta hat_h^alpha log(1-hat_h), otherwise]`

### T3 Offset

第 `i` 个伪区域中心为 `mu_i`：

`o_i*(x) = mu_i - x`

按区域等权：

`L_offset = 1/N sum_i 1/|R_i| sum_{x in R_i} SmoothL1(o(x)-o_i*(x))`

### T4 Boundary

仅对可靠接触边界使用：

`L_boundary = L_BCE + L_Dice`

### 置信度加权

若伪区域质量为 `q_i in [0,1]`：

`L_geometry = sum_i q_i (lambda_c L_center_i + lambda_o L_offset_i + lambda_b L_boundary_i)`

ambiguous 区域降低权重，ignored 区域 `q_i=0`。所有权重只在 Train/cross-fit 中选择并锁定到 Validation。

## 10. 隔离保证

新增代码只位于：

- `script/standard_ML/spi_stage0/`
- `tests/spi_stage0/`
- `run/run_SPI_STAGE0_isolated.sh`
- `docs/SPI_STAGE0_EXPERIMENT_ZH.md`

没有修改旧 P3、H7、H9、RGID、CMR、V1、V2 或 C0 文件。

## 11. GPU与几何评价执行约定

- P3与A2前向、Softmax和实例概率池化保持在同一CUDA设备。
- E1/E2/E4/E3几何评价不再对每个预测/GT组合反复扫描整幅mask。
- 常见的互斥实例使用GPU id-map加`bincount`一次统计全部交集。
- 若RGID COCO预测内部存在重叠，则使用分块CUDA矩阵乘法精确回退；不会为了加速而静默消除重叠。
- 一次得到的交集矩阵同时用于IoU贪心匹配、Merge和Split，指标定义与旧实现保持一致。
- 运行日志每10张图报告耗时和吞吐率，结果JSON记录推理、池化和几何设备。
