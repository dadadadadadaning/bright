# RGB结构锚定—跨模态变化细化软超像素 V3

## 设计

只维护一套共享软区域：

```text
灾前RGB结构 -> L_struct
局部对齐RGB-SAR -> DeltaL_change + reliability gate
Q = softmax(L_struct + alpha * DeltaL_change)
```

RGB决定稳定结构，SAR不能独立生成第二套硬分区，只能在建筑内部对归属进行有界修正。

## 消融

- P0：`audit_grid`，规则方格；
- P1：`audit_slico`，固定灾前RGB SLICO；
- P2：`p2_rgb_structure`，学习RGB结构分区；
- P5：`p5_anchored_change`，主方法；
- P6：`p6_shuffled_sar`，平移SAR负对照；
- P7：`p7_no_confidence`，去掉对齐可靠性；
- P8：`p8_no_damage_boundary`，去掉损毁语义重建与边界监督。

旧V1中的`c4_raw_sar_assoc/c5_aligned_assoc`分别作为P3/P4的一步联合分区历史对照，不混入V3权重。

## 推荐运行顺序

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/damageformer_rgb_anchored_change_superpixel_v3/run.sh preflight
bash experiments/damageformer_rgb_anchored_change_superpixel_v3/run.sh audit_grid
bash experiments/damageformer_rgb_anchored_change_superpixel_v3/run.sh audit_slico

for variant in p2_rgb_structure p5_anchored_change p6_shuffled_sar p7_no_confidence p8_no_damage_boundary; do
  bash experiments/damageformer_rgb_anchored_change_superpixel_v3/run.sh pipeline "$variant"
  bash experiments/damageformer_rgb_anchored_change_superpixel_v3/run.sh audit_learned "$variant"
done
```

也可以连续运行完整因果链：

```bash
nohup bash experiments/damageformer_rgb_anchored_change_superpixel_v3/run_ablation.sh quick \
  > /root/data/BRIGHT-train-logs/DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3/ablation_quick.log 2>&1 &
```

第一轮可设置`STAGE0_STEPS=500 STAGE1_STEPS=1000 VAL_MAX_BATCHES=64`快速筛选；正式结论必须使用完整Val450。

## 判定

- P2>P1：学习RGB结构优于固定SLICO；
- P5>P2：对齐变化证据改善分区；
- P5>P6：正确SAR对应关系必要；
- P5>P7：可靠性信息必要；
- P5>P8：损毁边界监督必要。

先比较`boundary_recall / purity / undersegmentation_error / affected_retention / oracle macro_iou`，通过后才比较最终像素mIoU并进入Memory。

Test始终禁止访问。
