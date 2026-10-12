# 03 模型改进效果解释

## A4_DamageExistHead

文件夹：`BGCA_final\M08_A4_ExistBCE010_ROId5_C640_I120k_BS2\DamageFormer`  
checkpoint：best_balanced_score.pth, best_damaged_iou.pth, best_destroyed_iou.pth, best_valid_miou.pth  
模型改进：A4 + DamageExistHead BCE  
设计目标：降低 damaged/destroyed 被错成 intact  
训练日志已有结论：本报告不从目录名臆测训练数值，需结合原始训练日志。  
official 评估结果：target_3class_miou=0.526786  
相比 M03_A4 的变化：
- target_3class_miou: -0.001090
- damaged_iou: -0.006054
- destroyed_iou: +0.005352
- damaged_to_intact: +0.046192
- destroyed_to_intact: -0.021801

效果判断：按上述差值判断。  
启示：降低 damaged/destroyed 被错成 intact。  
是否建议继续：结合主指标和 destroyed 错误流决定。

## M01_BGCA_none

文件夹：`BGCA_final\M01_BGCA_none_C640_I200k_BS2\DamageFormer_BGCA_custom_none_multiplicative_beta1.0_alpha0.1_d0_noalign_nodiff_nomddm_ce1.0_lovasz1.0_noordinal_20260610_211712`  
checkpoint：best_balanced_score.pth, best_damaged_iou.pth, best_destroyed_iou.pth, best_model.pth, best_valid_miou.pth  
模型改进：无 ROI、无 MDDM 基线  
设计目标：作为 MDDM/ROI 增益参照  
训练日志已有结论：本报告不从目录名臆测训练数值，需结合原始训练日志。  
official 评估结果：target_3class_miou=0.514675  
相比 M03_A4 的变化：
- target_3class_miou: -0.013202
- damaged_iou: -0.018294
- destroyed_iou: +0.000604
- damaged_to_intact: +0.000558
- destroyed_to_intact: -0.055817

效果判断：按上述差值判断。  
启示：作为 MDDM/ROI 增益参照。  
是否建议继续：结合主指标和 destroyed 错误流决定。

## M02_A4_noROI_MDDMbase

文件夹：`BGCA_final\M02_A4_noROI_MDDMbase_C640_I200k_BS2\DamageFormer_BGCA_custom_none_multiplicative_beta1.0_alpha0.1_d0_noalign_nodiff_mddm_gamma0.0_roi0_ce1.0_lovasz1.0_noordinal_20260611_102058`  
checkpoint：best_balanced_score.pth, best_damaged_iou.pth, best_destroyed_iou.pth, best_valid_miou.pth  
模型改进：加入 base MDDM，不使用 ROI  
设计目标：判断 MDDM 是否有效  
训练日志已有结论：本报告不从目录名臆测训练数值，需结合原始训练日志。  
official 评估结果：target_3class_miou=0.488967  
相比 M03_A4 的变化：
- target_3class_miou: -0.038910
- damaged_iou: -0.013594
- destroyed_iou: -0.089056
- damaged_to_intact: +0.017989
- destroyed_to_intact: +0.095581

效果判断：按上述差值判断。  
启示：判断 MDDM 是否有效。  
是否建议继续：结合主指标和 destroyed 错误流决定。

## M03_A4_MDDMbase_ROId5

文件夹：`BGCA_final\M03_A4_MDDMbase_ROId5_C640_I200k_BS2\DamageFormer_BGCA_custom_dilated_soft_multiplicative_beta1.0_alpha0.1_d5_noalign_nodiff_mddm_gamma0.0_roi1_ce1.0_lovasz1`  
checkpoint：best_balanced_score.pth, best_damaged_iou.pth, best_destroyed_iou.pth, best_model.pth, best_valid_miou.pth  
模型改进：base MDDM + dilated soft ROI  
设计目标：判断 ROI 对建筑损毁分类的价值  
训练日志已有结论：本报告不从目录名臆测训练数值，需结合原始训练日志。  
official 评估结果：target_3class_miou=0.527876  
相比 M03_A4 的变化：
- target_3class_miou: +0.000000
- damaged_iou: +0.000000
- destroyed_iou: +0.000000
- damaged_to_intact: +0.000000
- destroyed_to_intact: +0.000000

效果判断：按上述差值判断。  
启示：判断 ROI 对建筑损毁分类的价值。  
是否建议继续：结合主指标和 destroyed 错误流决定。

## M04_A4DG_b10

文件夹：`BGCA_final\M04_A4DG_b10_ROId5_C640_I120k_BS2\1`  
checkpoint：best_balanced_score.pth, best_damaged_iou.pth, best_destroyed_iou.pth, best_valid_miou.pth  
模型改进：dual-scale gate，large beta=1.0  
设计目标：降低 destroyed 被错成 intact  
训练日志已有结论：本报告不从目录名臆测训练数值，需结合原始训练日志。  
official 评估结果：target_3class_miou=0.524720  
相比 M03_A4 的变化：
- target_3class_miou: -0.003156
- damaged_iou: -0.009271
- destroyed_iou: -0.004535
- damaged_to_intact: +0.033607
- destroyed_to_intact: -0.023813

效果判断：按上述差值判断。  
启示：降低 destroyed 被错成 intact。  
是否建议继续：结合主指标和 destroyed 错误流决定。

## M05_A4D_MDDMlarge

文件夹：`BGCA_final\M05_A4D_MDDMlarge_ROId5_C640_I120k_BS2\beta1.0_alpha0.1_d5_c_ce1.0_lovasz1.0_noordinal`  
checkpoint：best_balanced_score.pth, best_damaged_iou.pth, best_destroyed_iou.pth, best_valid_miou.pth  
模型改进：large-scale MDDM  
设计目标：增强 destroyed 大尺度损毁证据  
训练日志已有结论：本报告不从目录名臆测训练数值，需结合原始训练日志。  
official 评估结果：target_3class_miou=0.518489  
相比 M03_A4 的变化：
- target_3class_miou: -0.009387
- damaged_iou: -0.067164
- destroyed_iou: +0.037659
- damaged_to_intact: +0.070021
- destroyed_to_intact: -0.116297

效果判断：按上述差值判断。  
启示：增强 destroyed 大尺度损毁证据。  
是否建议继续：结合主指标和 destroyed 错误流决定。

