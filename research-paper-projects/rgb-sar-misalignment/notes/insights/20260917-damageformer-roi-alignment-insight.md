# RGB–SAR轻微错位方法向DamageFormer-ROI的迁移判断

## 最终推荐

第一选择不是完整外部配准器，而是ROI约束的局部相关对齐残差：以RGB建筑特征作为Query，在扩张建筑ROI内对SAR特征做3×3或5×5局部相关搜索，得到软匹配特征、对齐熵和期望偏移；只有匹配置信度高时，通过零初始化残差加入原DamageFormer融合特征。

## 原因

- BRIGHT输入已经大致地理配准，目标是容忍几像素残差，而不是恢复大范围全局几何变换。
- 全图XoFTR/MINIMA/SOMA会增加额外预训练、对应标签、RANSAC/Flow和显存成本，且配准错误可能直接扭曲损毁区域。
- CASP式上下文对齐与DamageFormer双分支特征最接近，但应改为单向RGB锚定SAR，避免SAR反向污染RGB结构。
- HOMO的梯度方向可作为匹配Key的辅助通道，降低RGB与SAR强度分布不同造成的相关性失败。

## 建议模块

```text
RGB特征 R -> Q(R)
SAR特征 S -> K(S), V(S)
P_building -> 扩张ROI
Q与ROI内邻域K做局部相关 -> softmax权重
权重汇总邻域V -> aligned_S
熵 -> confidence
F_parent + gamma * confidence * Conv[R, aligned_S, |R-aligned_S|, R*aligned_S]
```

其中gamma零初始化，window按特征尺度依次测试1、3、5。第一版不预测稠密Flow、不直接重采样原始SAR图像，只对特征做软聚合。

## 必要消融

- A0：原始拼接，window=1。
- A1：扩张ROI但无局部搜索。
- A2：3×3局部相关，无置信度门控。
- A3：3×3局部相关+熵门控。
- A4：A3+梯度方向Key。
- A5：5×5窗口，检查是否超过3×3。

机制验证必须报告clean、shift_x/y 4、8、16像素的下降曲线，并对SAR随机配对进行负对照。仅clean mIoU提高不能证明错位容忍。
