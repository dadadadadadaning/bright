# DamageFormer A1 跨模态可学习超像素 V1

## 一句话说明

先在建筑 ROI 内让 RGB 到附近寻找可信的 SAR 响应，再根据“RGB 结构 + 可信变化证据”学习局部区域，最后把区域上下文作为零初始化残差送回 A1 的像素四分类分支。

本实验不包含 Memory，不以超像素多数投票替代像素分类，也不修改旧实验。

## 隔离边界

- 父权重只读：A1 `best_val_miou.pth`；
- 新参数统一使用 `cmsp.*`；
- 新权重：`DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1`；
- 新结果：`DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1`；
- Test 被代码锁定，只允许 Train/Val；
- C0 从父权重直接审计，不重新训练。

## 消融含义

| 变体 | 唯一主要变化 | 回答的问题 |
|---|---|---|
| C0 | 原始 A1 | 基线是否被准确复现 |
| C1 | A1 + LCA | 局部位置容错是否有价值 |
| C2 | 固定 RGB-SLICO | 固定外观区域是否有价值 |
| C3 | RGB 可学习区域 | 学习区域是否优于固定区域 |
| C4 | RGB + 未对齐 SAR | 原始 SAR 参与划分是否造成干扰 |
| C5 | RGB + 对齐 SAR + 置信度 | 核心完整方案 |
| C6 | 去掉置信度 | 可靠性门控是否必要 |
| C7 | 打乱 SAR | 收益是否真的来自正确跨模态对应 |
| C8 | 加入 A1 类别概率 | 检验是否只是复制 A1 判别边界 |
| C9 | C5 + 边界约束 | 弱边界监督是否进一步有效 |

## 推荐运行顺序

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

export A1_CKPT=/root/data/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth

bash experiments/damageformer_a1_cross_modal_superpixel_v1/run.sh preflight
bash experiments/damageformer_a1_cross_modal_superpixel_v1/run.sh audit_a1

# 先做 20 步功能验证
bash experiments/damageformer_a1_cross_modal_superpixel_v1/run_sequential.sh \
  smoke c1_lca c2_slico c3_rgb_assoc c4_raw_sar_assoc c5_aligned_assoc

# 再做核心 5000 步消融
nohup bash experiments/damageformer_a1_cross_modal_superpixel_v1/run_sequential.sh \
  probe c1_lca c2_slico c3_rgb_assoc c4_raw_sar_assoc c5_aligned_assoc \
  c6_no_confidence c7_shuffled_sar c8_damage_hint c9_boundary \
  > /root/data/BRIGHT-train-logs/DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1/launcher_probe5000.log \
  2>&1 &
```

## 准入标准

C5 只有同时满足以下条件才进入 20000 步确认：

1. C5 的 Valid mIoU 高于 C0，建议至少 `+0.20pp`；
2. C5 高于 C3，证明 SAR 损毁证据对区域形成有帮助；
3. C5 高于 C4，证明“先对齐”优于直接混入 SAR；
4. C5 高于 C6，证明置信度不是装饰；
5. C7 收益明显消失，证明正确 RGB-SAR 对应关系被利用；
6. Damaged F1 不下降，Destroyed IoU 不出现明显牺牲；
7. 人工位移审计下，C5 的下降小于 C0/C3。

任何单独的“区域重建损失下降”都不能证明最终方法有效，最终结论以 Val 像素指标、事件稳定性与人工错位鲁棒性为准。

