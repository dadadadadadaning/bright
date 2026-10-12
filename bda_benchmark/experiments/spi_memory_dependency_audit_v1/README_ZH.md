# Memory Dependency Audit V1

该实验连续验证两个问题，不重新推理P3，不重新划分数据：

1. R-M3角色Token收益是否真正依赖检索到的外部病例；
2. R-M4困难病例路由是否真正依赖Memory，并超过“始终使用B2专家”。

R-M3包含B2基线、真实跨事件支持、随机跨事件支持、同事件内打乱对应、自身角色和零支持。所有训练臂使用相同初始化和相同采样序列，确保差异只来自支持Token。

R-M4包含P3、始终使用专家、仅P3置信度、P3特征、真实Memory分歧和打乱Memory分歧。路由阈值只按meta_train冻结为20%目标路由率，绝不使用meta_calibration标签选阈值。

科学STOP不会阻止另一部分运行；技术错误才中止。实验仅使用既有meta_train/meta_calibration缓存，不访问meta_confirm、Val450或Test。
