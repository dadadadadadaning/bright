# Superpixel Relation Occurrence V2（修正版）

本实验不修改 P3。SLIC 只从已经完成数据增强的灾前 RGB crop 生成区域 ID；区域 ID 不是建筑实例，也不携带四类损毁标签。P3 继续逐像素输出四类结果。

训练目标在 occurrence 输出网格上计算：只有 P3 预测为 Background/Intact、预测建筑概率至少 0.5，并且 GT 建筑覆盖率至少 25% 的 token 进入候选。GT 在训练时只用于构造正负目标和决策距离，不进入超像素特征。损毁感知 crop 提高含漏判正样本 batch 的比例。

运行顺序：`preflight` → `smoke` → `screen` → `formal`。`screen` 固定为500步，只筛查模块是否学得动；`formal` 固定为5000步，用于正式比较 SP1、SP2、SP3、SP4、SP5，并自动生成 SP0-SP5 汇总。二者都不会访问 meta_confirm、Val450 或 Test。`auto` 仅保留给需要通过 `STEPS` 自定义步数的情况，默认步数为5000。

SP1 为参数量匹配对照；SP2 为真实 SLIC；SP3 加低权重可靠区域一致性；SP4 使用镜像 RGB 产生统计相近但与当前图像错配的 SLIC；SP5 使用规则网格。只有 SP3 同时优于 SP1/SP4/SP5，才可归因于正确的 RGB 超像素结构。

V2 还做了三项安全修正：SLIC 区域在预测建筑边界处拆开，避免道路/背景污染建筑均值；超像素数量随输入面积缩放，保持 640 crop 与 1024 全图的区域尺度接近；缺少 scikit-image 时直接停止，不再静默退化为规则网格。
