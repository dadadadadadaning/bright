# 建筑条件超像素 Memory V2 设计

- 用户研究方向：灾前 RGB 与灾后 SAR 建筑损毁像素级分割
- 用户研究问题：以超像素作为真实 Memory 条目，替代每图每类平均原型与简单 `RGB+SAR+|RGB-SAR|` 描述
- 分析日期：2026-09-16

## 关键判断

旧 Building Prototype Memory V1 在形式上是 Memory，但每图每类平均会消除区域差异，无监督 PCA 不保证保存损毁语义，单一 occurrence 残差也无法使用 Damaged/Destroyed 的区别。新版本应当让“一个合格超像素对应一条真实 Memory 记录”，并专门训练可检索的跨模态 Key。

## 推荐流程

```text
灾前 RGB + 冻结建筑概率
  -> RGB-only SLIC/soft-superpixel 分区
  -> 只保留建筑内的连通超像素
  -> RGB/SAR 模态专用投影
  -> 局部 SAR 容错匹配
  -> 关系图 [r,s,|r-s|,r*s,fused]
  -> 区域 mean/std/max/attention + 内外环差异 + 几何/可靠性
  -> 监督度量 Key Encoder
  -> 每个纯净超像素一条 Train-only Memory 记录
  -> 全局 Top-k 检索与标签加权
  -> 置信门控的 region-to-pixel 上下文
  -> 原像素分类头
```

## 超像素规定

1. 分区输入只使用推理时可获得的灾前 RGB、冻结建筑概率和 RGB 边缘，不使用 GT 或 SAR 生成边界。
2. 第一版使用固定 SLIC；只有固定 SLIC 通过随机 ID 和规则网格对照后才升级可学习 SSN。
3. 超像素 ID 必须与图像做完全相同的裁剪、翻转、90 度旋转，ID 只允许最近邻变换。
4. 超像素与预测建筑区相交后重新做连通域编号；不把两个不连通区域作为同一个节点。
5. Memory 写入要求：建筑覆盖率不低于 0.7、有效 Token 数不低于 4、三类标签纯度不低于 0.8。混合区域不写入，而不是强行多数投票。
6. 保存三类标签 Intact/Damaged/Destroyed，不先改成二分类；同时保存完整软标签直方图、event/image provenance、面积、边界和检索可靠性。

## Key 构造

像素关系特征先由模态专用投影和局部 SAR 容错匹配生成：

`z_i = Phi([P_rgb(f_rgb), P_sar(Align_local(f_sar)), |r-s|, r*s, f_fused])`

超像素 Key 不只取均值，而由以下信息编码：

- 区域内部 `mean/std/max/attention pooling`；
- 超像素核心与外围 ring 的差异；
- RGB 边缘强度和方向统计；
- SAR 均值、标准差、分位数和局部匹配熵；
- 面积、紧致度、质心和建筑覆盖率；
- 作为门控而非类别证据的 P3 熵、margin 和配准可靠性。

最终 Key 使用监督度量学习，而不是 PCA：同类不同图/不同事件超像素为正对，异类为负对，Damaged/Destroyed 作为重点硬负对。

## Memory 规定

- 每个合格超像素保存一个实际 exemplar，不再把同图同类区域平均成一个原型。
- 三类等容量；每事件每类设置上限，避免某个大事件占满 Memory。
- 每条记录包含 `key, soft_label, hard_label, event_id, image_id, geometry, reliability`。
- 训练查询排除同图；跨事件审计排除同事件；Validation/Test 永远只读。
- 第一版使用全局 class-balanced Top-k 检索；只有检索规模成为瓶颈后才聚类压缩。

## 最小实验

| 变体 | 内容 | 问题 |
|---|---|---|
| SM0 | 冻结 A1 | 基线 |
| SM1 | 参数匹配，无 Memory | 排除参数量 |
| SM2 | 简单均值 Key + 超像素 exemplar | 超像素条目本身是否有效 |
| SM3 | 多统计关系 Key | 新 Key 是否有效 |
| SM4 | SM3 + 度量学习 | 可检索空间是否必要 |
| SM5 | SM4 + 打乱标签 | 是否真正使用标签语义 |
| SM6 | SM4 + 规则网格 | SLIC 边界是否必要 |
| SM7 | SM4 + 跨事件排除 | 未见事件检索是否成立 |

在接入像素输出前先做裸检索审计：Top-k Accuracy、Macro-F1、每类 Recall、leave-one-image/event 结果、邻居一致率、相似度 margin。若正确标签不明显优于打乱标签，停止，不训练回写头。

## 论文迁移依据

- SLIC：内容自适应且贴合边界的低层区域单元。
- SSN：可学习、可微的任务特定超像素，但适合放在固定 SLIC 已被验证之后。
- SegSort：以区域原型组织嵌入，并使用跨批 Memory 将不同图像的同语义片段联系起来；与“每超像素一个 exemplar + 度量 Key”高度匹配。
- XoFTR/MINIMA/HOMO-Feature：提示跨模态关系不能依赖同坐标直接作差，应使用局部匹配、模态不变结构和准确对应监督。

## 风险和停止条件

- 若 SM4 不优于 SM3，度量训练无价值；
- 若 SM4 不优于 SM5，Memory 类别语义仍未被使用；
- 若 SM4 不优于 SM6，SLIC 不优于普通网格，不能宣称超像素贡献；
- 若跨事件检索 Macro-F1 明显下降，Memory 记住的是事件风格而不是损毁；
- 若像素 mIoU 上升但边界/小损毁指标下降，region-to-pixel 回写过强，应保留像素残差而禁止硬广播。

## 来源

- SLIC: https://doi.org/10.1109/TPAMI.2012.120
- SSN: https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html
- SegSort: https://openaccess.thecvf.com/content_ICCV_2019/html/Hwang_SegSort_Segmentation_by_Discriminative_Sorting_of_Segments_ICCV_2019_paper.html
- XoFTR: https://openaccess.thecvf.com/content/CVPR2024W/IMW/html/Tuzcuoglu_XoFTR_Cross-modal_Feature_Matching_Transformer_CVPRW_2024_paper.html
- MINIMA: https://openaccess.thecvf.com/content/CVPR2025/html/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.html
- HOMO-Feature: https://openaccess.thecvf.com/content/ICCV2025/html/Gao_HOMO-Feature_Cross-Arbitrary-Modal_Image_Matching_with_Homomorphism_of_Organized_Major_Orientation_ICCV_2025_paper.html
