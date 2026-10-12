# 条件错误家族 Memory 审计

该实验只验证诊断能力，不改P3、不写回像素。

训练集按P3预测类别分别计算不确定性80%分位点；只有高于该点的建筑进入复查候选。该阈值不看GT，也不从meta_calibration选择。

在候选内比较四种相同容量的错误家族诊断器：P3概率、P3+真实Residual Memory、P3+打乱Memory、P3+零Memory。真实Memory必须同时优于P3、打乱Memory和零Memory，才允许进入下一阶段“局部专家Head”研究。

