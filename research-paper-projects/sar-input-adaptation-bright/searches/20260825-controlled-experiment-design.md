# BRIGHT 输入适配与受控实验设计（2026-08-25）

1. **P010｜Net2Net: Accelerating Learning via Knowledge Transfer**
   - 收录：ICLR 2016
   - 核心方法：用函数保持变换改变网络规格，新模型在变换前后表示同一个函数。
   - 对本项目的意义：A4不是“新SAR模型”，而是把P3三通道输入契约严格折叠成单通道的零步等价对照。
   - 原文：[Google Research](https://research.google/pubs/net2net-accelerating-learning-via-knowledge-transfer/)

2. **P011｜Bright dataset paper**
   - 收录：Earth System Science Data，2025
   - 核心发现：BRIGHT包含14个事件、单极化VHR SAR、明显事件分布差异、长尾建筑尺度和类别不平衡。
   - 对本项目的意义：输入头适配和跨事件适配必须分开验证；训练集统计只能由训练集产生。
   - 原文：[ESSD](https://essd.copernicus.org/articles/17/6217/2025/)

3. **P012｜Superpixel Sampling Networks**
   - 收录：ECCV 2018
   - 核心方法：深度像素特征 + 可微SLIC迭代聚类 + 任务监督。
   - 对本项目的意义：一次软网格平均不是完整SSN；分区必须先通过边界、归属和泄漏审计，才能进入分类或检索。
   - 原文：[CVF](https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html)

4. **P013｜SRAC**
   - 收录：IGARSS 2026（录用题目已由会议列表核实）
   - 公开信息：VMambaSSN结构分区 + 标注记忆检索 + 稀有类别增强。
   - 证据边界：截至检索日未找到公开全文或官方代码，不能确认损失、分区准入、记忆更新和纯监督/传导式协议细节。
   - 来源：[IGARSS accepted papers](https://2026.ieeeigarss.org/papers/accepted_papers.php)、[机构介绍](https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026)

5. **P014｜Group DRO**
   - 收录：ICLR 2020
   - 核心方法：优化预定义群组中的最坏风险，但论文明确指出还需要正则化或早停。
   - 对本项目的意义：事件均衡/最坏事件训练应作为模型结构锁定后的训练策略对照，不能用来掩盖输入或融合错误。
   - 原文：[ICLR](https://iclr.cc/virtual/2020/poster/1491)

数据来源：Google Research、ESSD、CVF Open Access、IGARSS 2026、江海实验室、ICLR。
