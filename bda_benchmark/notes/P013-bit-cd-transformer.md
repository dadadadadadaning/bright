# P013 — Remote Sensing Image Change Detection With Transformers

## 基本信息

- 论文：Hao Chen, Zipeng Qi, Zhenwei Shi, *Remote Sensing Image Change Detection With Transformers*，IEEE TGRS 60 (2022), 1–14。
- DOI：10.1109/TGRS.2021.3095166
- 全文：https://levir.buaa.edu.cn/static/pdfs/2022_hao_chen_remote.pdf
- 代码：https://github.com/justchenhao/BIT_CD
- 任务边界：同一区域、已配准的双时相光学影像，输出二分类“变化/未变化”图。

## 一句话结论

BIT 的核心不是“用了 Transformer”，而是先把大量像素压缩为少量语义 token，在小规模 token 空间中比较两个时相，再把变化上下文传回像素；这一思想适合迁移到 BRIGHT 的建筑内部上下文修正，但原模型不能直接当作 RGB–SAR 三分类建筑损毁模型使用。

## 方法流程

1. 两个时相分别经过共享的 ResNet18，得到像素特征图。
2. Semantic tokenizer 为每个时相生成 L 张软注意力图，用加权平均把 H×W 个像素压缩成 L 个 token：`T_i = softmax(phi(X_i))^T X_i`。
3. 把两个时相的 token 拼接，使用 Transformer encoder 建模跨时相、跨空间关系。
4. Transformer decoder 让每个像素查询处理后的 token，把全局语义重新写回像素特征。
5. 对两个增强特征取绝对差，再由浅层卷积头输出二分类变化图。

默认轻量版本采用 ResNet18 的 1/4 分辨率特征、32 维通道、每个时相 4 个 token、1 层 encoder 和 8 层 decoder。关键计算发生在 `L << H×W` 的 token 空间，因此比直接对全部像素做自注意力便宜。

## 主要证据

- 三个数据集：LEVIR-CD、WHU-CD、DSIFN-CD；主指标为变化类别 F1，同时报告 Precision、Recall、IoU 和 OA。（论文 Sec. IV-A）
- BIT 的 F1/IoU：LEVIR-CD 89.31/80.68，WHU-CD 83.98/72.39，DSIFN-CD 69.26/52.97。（Table I）
- 相比作者的 Base，F1 分别由 87.57、81.61、58.48 提升到 89.31、83.98、69.26。（Table I）
- 轻量 BIT 为 3.55M 参数、4.35G FLOPs；作者的较深 Base 为 11.85M、12.99G。（Table II）
- Tokenizer、encoder、decoder 的组合优于只加普通 non-local 模块；token 数实验中 L=4 最佳，过多 token 可能引入冗余。（Tables III–IV、Sec. IV-E）

## 对 BRIGHT 的适用边界

- 论文是光学–光学、二分类变化检测；当前项目是灾前 RGB、灾后 SAR、三类损毁分级，模态与标签目标都不同。
- BIT 的双分支共享权重适用于同模态输入，不宜原样用于 RGB–SAR；至少需要模态专属投影或编码器。
- 绝对差适合检测“是否变化”，但不能天然区分 Damaged 与 Destroyed。
- 论文没有按灾害事件留一验证，因此它不能直接证明未见事件泛化。
- 论文依赖较好配准；BRIGHT 中的跨模态偏移会污染 token 对齐与差分。

## 与当前 P3 的关系

当前 P3 已经包含双 Swin 编码器和建筑级 MICB-DSMIL token/池化，不能把“加入 Transformer”本身当作新贡献。当前实现还把建筑 logit 到像素的回写默认关闭（`mil_residual_scale=0`）；因此 BIT 最有价值的缺口是“让建筑上下文以受控残差方式回写像素”，而不是再次增加全局编码器。

## 证据定位

- 研究问题：Abstract、Sec. I，pp. 0–1。
- 网络结构：Fig. 2、Sec. III，pp. 2–5。
- Tokenizer：Eq. (1)，p. 4。
- 差分与预测头：Eq. (7)，p. 5。
- 数据和训练：Sec. IV-A，pp. 5–6。
- 主结果：Table I，p. 8。
- 效率与消融：Tables II–IV，pp. 9–10。
- 超参数与 token 数：Sec. IV-E，p. 10。

