# Affected Occurrence Head V1

此实验冻结 P3，仅训练一个高分辨率受损发生残差头。所有变体都向 Damaged 与 Destroyed
加入相同残差，因此不改变二者的 logit 差。

`m1` 是共享基础分支；`m2` 增加预测建筑边界特征和仅训练期边界权重；`m3` 再增加仅训练期
小损毁权重；`m4` 增加连续 SAR 质量门控。GT 生成的边界与小损毁掩码绝不进入推理。

运行 `run.sh auto` 会从同一个冻结 P3 起点，依次训练并评估四个独立变体。只使用
meta_train 训练、meta_calibration 内部评估；不访问 meta_confirm、Val450 或 Test。
