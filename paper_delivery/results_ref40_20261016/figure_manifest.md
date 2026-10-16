# 可编辑图件页码对照

figures_editable.pptx 共12页；所有原始图片对象已按SHA256验证，保存并重新打开后字节一致。矢量图、文字、边框保留独立编辑能力。不同尺寸来源在统一画布中等比居中；几何坐标、字号与线宽同步缩放。

| PPT页 | 论文图号/文件名 | 内容 |
| --- | --- | --- |
| 1 | Fig1_framework | 方法总体流程 |
| 2 | Fig2_region_encoding | 建筑感知区域编码 |
| 3 | Fig3_retrieval_reader | 检索与Reader流程 |
| 4 | Fig4_dataset | 数据集场景与分布 |
| 5 | Fig5a_external_models | 地震与冲突的六个局部案例 |
| 6 | Fig5b_external_models | 火山与野火的六个局部案例 |
| 7 | Fig5c_external_models | 飓风、洪灾与爆炸的六个局部案例 |
| 8 | Fig6_noto | 能登地震局部放大案例 |
| 9 | Fig7_turkey | 土耳其地震局部放大案例 |
| 10 | Fig8_myanmar | 缅甸飓风局部放大案例 |
| 11 | Fig9_structure_diagnostic_template | 结构诊断空模板（待填实测输出） |
| 12 | Fig10_retrieval_diagnostic_template | 检索诊断空模板（待填实测输出） |

## 使用边界

- 图5a、5b、5c共18个固定裁剪案例，灾害分组已分别核对为地震与冲突、火山与野火、飓风/洪灾/爆炸。
- 图6—8取自原始PPT的能登、土耳其、缅甸三例。MPF为原Ours(A1)像素前端，未添加Memory预测。
- 图9/10仅为空模板，仍需填入区域分配、building prior、attention/retrieved-support等真实输出。不可把空白占位框或示意连线作为实验发现。
- 蓝框是用于局部放大的人工标记，不是学习得到的注意力。
- 图5a—8支持局部现象描述；逐阶段因果归因需在同一病例增加base/region/roi/memory预测并绑定检查点。
- 数据集图和方法框架仅说明设置与计算流程，不代替效果实验。
