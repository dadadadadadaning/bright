# BRIGHT Windows CPU official 评估模型选择报告

- 已成功汇总模型：1
- 已成功汇总 checkpoint：1
- 缺失输出 warning：0

## 最佳结果

- target_3class_miou 最佳：M01_BGCA_none / best_model.pth（0.182305）
- damaged_iou 最佳：M01_BGCA_none / best_model.pth（0.007630）
- destroyed_iou 最佳：M01_BGCA_none / best_model.pth（0.008174）
- destroyed_to_intact 最低：M01_BGCA_none / best_model.pth（0.841682）

## 研究问题判断

- A4 稳定性：尚无 M03_A4 的成功 official 结果，不能判断。
- large-scale：相对 A4，destroyed_recall 数据不足，damaged_iou 数据不足。若前者上升而后者下降，说明它帮助 destroyed 但伤害 damaged。
- A4-DG-beta0.3：相对 beta1.0，damaged_iou 数据不足，destroyed_to_intact 数据不足。damaged 恢复且误分不反弹才说明 beta 控制有效。
- HDC：相对 A4，destroyed_to_intact 数据不足，target_3class_miou 数据不足。前者下降且主指标不明显下降才值得继续。

## 下一步

优先查看 event_wise_destroyed_diagnosis.csv：destroyed_to_background 高时应改 ROI/定位；destroyed_to_damaged 高时应改 severity 边界；destroyed_to_intact 高时优先评估 DG-beta、ExistHead 或 HDC。
