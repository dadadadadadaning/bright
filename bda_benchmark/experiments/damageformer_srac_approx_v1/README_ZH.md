# DamageFormer SRAC approximate V1

不是官方SRAC复现。原文训练协议、损失、Memory配置和指标定义尚未获得。
保留Transformer A1主干，冻结其参数；不加载VMamba，不启用Bag，不启用旧Memory。

## 已可训练
- s0_grid：固定网格归属、均值汇总、四类区域分类再软投影到像素。
- s1_learned：RGB特征+坐标预测9邻域软归属；不是SSN原版迭代聚类，无UNet++，不保证硬区域连通。
- s2_modulated：额外使用预测建筑概率及硬分区边界进行特征级调制；不是SRAC图中的原图调制再编码。硬边界不可微，分区梯度来自软投影与重建损失。
- s3_weighted：全特征网格加权区域Pooling+均值/标准差旁路；不做12点采样。
- 所有区域参与四类像素损失，含背景；GT不参与分区前向。
- 训练目标：四类CE + 0.1*(RGB归一化特征重建MSE + 0.1坐标重建MSE + 建筑重建BCE)。均为本项目设定，不宣称来自原文。
- 全精度训练避免旧BCE autocast问题。默认512裁剪，batch1；每个变体相同训练预算。
- 每次验证同时报告A1和区域模型的前景mIoU，仅保存best/last。随机初始化的区域头不保证初始等于A1。

## Memory范围（重要）
modules.py提供RetrievalClassifier：query、support Key、support标签/可选灾害类型及几何Value、Cross-Attention、FFN、四分类。
本目录当前train.py只训练分区/调制/Pooling，不训练或建立Memory，不得把s3叫作完整SRAC。
尚需实现并单独验证：冻结最佳区域模型后从Train全图生成region embedding与GT标签、事件均衡Memory、冻结检索表、同容量无Memory与Memory分类头训练、逐像素投影评估。
旧E0-E5缓存及train_memory.pt属于另一特征空间，禁止直接复用。
灾害类型应使用Train定义的固定类型映射（未知为UNK），不是随意将事件ID视为灾害类型。几何定义建议归一化中心x/y、面积比、长宽比，必须统一Train/Val。

## 限制与验证
仅使用Train/Val。已有Test已查看，不以Test反馈调参。第一步检查同流程A1复现；其次检查软归属和分区可视化；然后比较s0→s1→s2→s3，指标包括前景mIoU和逐类混淆矩阵。
REGION_STRIDE按特征网格定义，Train裁剪与Val整图采用相同stride，区域数不固定为400。训练/验证分辨率改变会改变区域数量。
训练前运行run.sh preflight；新输出目录独占创建，现有实验不覆盖。无GPU真实验证结果或性能保证。
