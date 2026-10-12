# 01 模型库存

- 含正式候选 checkpoint 的模型目录：6
- 正式候选 checkpoint：26
- 可执行 checkpoint：26
- 参数或架构不确定 checkpoint：0
- 当前缺少正式权重：M07_A4DG_b03、M11_A4HDC_a03
- 当前不存在任务书提到的 iter_46000.pth / iter_44000.pth。
- M06C 的真实 args.json 为 lambda_align=0.0001、warmup=10000、ramp=10000；lambda_align 属于训练损失参数，推理只需恢复模型结构参数。

## 已识别模型

| 优先级 | 标准模型 | checkpoint 数 | 参数状态 |
|---:|---|---:|---|
| 1 | A4_DamageExistHead | 4 | ARGS_JSON |
| 1 | M01_BGCA_none | 5 | ARGS_JSON |
| 1 | M02_A4_noROI_MDDMbase | 4 | ARGS_JSON |
| 1 | M03_A4_MDDMbase_ROId5 | 5 | PATH_RULE |
| 1 | M04_A4DG_b10 | 4 | ARGS_JSON |
| 1 | M05_A4D_MDDMlarge | 4 | ARGS_JSON |
