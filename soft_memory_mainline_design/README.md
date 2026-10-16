# Soft＋Memory 主线设计包

这是对上传《新图.pptx》与当前 `soft_memory_q_rebased_v1_fix2` 实现的对齐设计。原PPT未修改，原数值未沿用，所有新案例框均为待填模板。

优先阅读：

1. `soft_memory_mainline_design_editable.pptx`：9页可编辑框架、对照与案例设计。
2. `soft_memory_mainline_design_editable.pdf`：直接预览。
3. `EXPERIMENT_PLAN.md`：完整实验顺序、消融定义、训练退化诊断、评价与结果解释。
4. `ABLATION_MATRIX.csv`：组别与实现状态；新增消融明确标记尚未实现。
5. `CASE_MANIFEST_TEMPLATE.csv`：真实案例的溯源记录空表。
6. `RESULTS_TEMPLATE.csv`：按协议、组别、种子登记真实结果的空表。
7. `ORIGINAL_PPT_REVIEW.md`：原12页的逐页修改建议。
8. `PROVENANCE.json`：原文件与核对代码的SHA256，便于追溯。

`build.py` 是生成PPT的脚本，不是训练脚本。已有训练结果不在这个包中，新增的单槽、错位B、随机案例、标签消融训练驱动也未在本次交付中实现。

方案关键点：

- 固定当前Q；soft只改变Q后的双槽汇总和回投，不宣称已实现逐栋建筑内部区域。
- 无Memory接受/拒绝门控；像素、父区域、双槽、历史参考共同分类。
- 原A-D首先做同源、配对种子重复；新消融按要证明的论断选择。
- 先完成500/1500/5000固定Train/Val诊断；短训练保留原学习率轨迹需要分离日程长度和停止步数。
- 旧A1案例不能改名为D；真实图需同图同裁剪重新导出。

可在本机打开PPTX或PDF查看；这些设计文件不会自动出现在研究服务器。
