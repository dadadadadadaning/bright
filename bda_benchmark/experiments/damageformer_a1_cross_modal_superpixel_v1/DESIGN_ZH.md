# A1-LCA 跨模态可学习超像素 V1：代码设计方案

## 1. 研究目标

在不运行第二套完整主干的前提下，将局部 SAR 对齐和可学习超像素嵌入 A1：

1. A1 保持当前最优 DamageFormer 主路径；
2. LCA 只负责在建筑 ROI 邻域内读取位置容错的 SAR 证据；
3. SCN 式关联头根据 RGB 结构与可靠的跨模态变化证据生成软超像素；
4. 超像素只组织局部证据，不直接输出最终类别；
5. 通过零初始化残差把区域信息安全回写到四分类像素 logits。

## 2. 为什么选择 SCN 式关联头

候选方法包括 SSN、SEAL、SCN、AINet、LNS-Net 和 CDS。V1 选择 SCN 的像素—网格关联形式：

- 输出固定为每个像素对周围 3×3 网格中心的 9 个关联概率；
- 全卷积、非迭代，容易在单卡上训练；
- 可直接使用 A1-LCA 的中间特征，不需要新建大型特征主干；
- 软关联矩阵可用于 Token 池化、重投影和端到端损失；
- 可公平比较 RGB、未对齐 SAR、对齐 SAR 与质量门控。

AINet 的显式像素—网格上下文和边界损失作为后续增强，不在 V1 同时加入。CDS 的风格去耦作为跨事件泛化扩展，也不在 V1 同时加入。

## 3. 目录隔离

```text
experiments/damageformer_a1_cross_modal_superpixel_v1/
├── __init__.py
├── variants.py
├── association.py
├── losses.py
├── model.py
├── train.py
├── audit.py
├── summarize.py
├── run_sequential.sh
├── run.sh
├── test_contract.py
└── DESIGN_ZH.md
```

不得修改原始 D0、A0、A1、ROI-LCA 和旧 Memory 实验。A1 权重以父模型方式加载；新增参数使用独立前缀 `cmsp.`。

## 4. 模型数据流

```text
Pre-RGB -> A1 RGB encoder -> RGB feature --------------------+
                         -> localization head -> soft ROI     |
Post-SAR -> A1 SAR encoder -> SAR feature -> local alignment  |
                                      -> aligned SAR          |
                                      -> entropy/confidence   |
                                                               v
RGB structure + confidence * cross-modal relation -> SCN association head
                                                               |
                                                        Q: H×W×9
                                                               |
content feature --soft pooling--> local Tokens --reprojection--+
                                                               |
A1 base fused feature + zero-init gated region residual --------+
                                                               v
                                               four-class damage head
```

## 5. 核心张量

```python
rgb_feat:          [B, C, H, W]
sar_feat:          [B, C, H, W]
aligned_sar:       [B, C_v, H, W]
align_confidence:  [B, 1, H, W]
building_prob:     [B, 1, H, W]
spixel_embedding:  [B, C_s, H, W]
association:       [B, 9, H, W]
token:             [B, K, C_t]
region_context:    [B, C_t, H, W]
```

## 6. 跨模态划分特征

不能直接拼接最终 damage logits。推荐：

```python
rgb_structure = rgb_sp_proj(rgb_feat)
relation = relation_proj(torch.cat([
    rgb_value,
    aligned_sar,
    torch.abs(rgb_value - aligned_sar),
    rgb_value * aligned_sar,
], dim=1))

reliable_relation = (
    building_prob.detach()
    * align_confidence.detach()
    * relation
)

spixel_embedding = spixel_fuse(torch.cat([
    rgb_structure,
    reliable_relation,
    rgb_edge,
    building_prob.detach(),
], dim=1))
```

`rgb_structure` 决定建筑轮廓和屋顶连续性；`reliable_relation` 只在 SAR 匹配可信时影响区域划分。最终类别概率不进入该分支。

## 7. SCN 式关联头

```python
class CrossModalAssociationHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 9, 1),
        )

    def forward(self, feature):
        return torch.softmax(self.net(feature), dim=1)
```

9 个通道表示当前像素属于周围 3×3 初始网格中心的概率。需要实现 `association_to_global_ids()`，把局部 9 路关联映射到全局 K 个超像素中心。

初始配置：

```text
feature resolution: A1 融合层分辨率
grid stride: 8
candidate centers per pixel: 3×3
hard connectivity: 仅用于可视化与最终导出
training/token pooling: 使用 soft association
```

## 8. 软 Token 池化和重投影

```python
def soft_pool(content, assignment, eps=1e-6):
    # content: [B,C,HW], assignment: [B,K,HW]
    mass = assignment.sum(-1, keepdim=True).clamp_min(eps)
    token = torch.einsum("bkn,bcn->bkc", assignment, content) / mass
    return token, mass

def soft_unpool(token, assignment):
    # output: [B,C,HW]
    return torch.einsum("bkn,bkc->bcn", assignment, token)
```

Token 内容使用分类前内容特征，而不是超像素划分特征本身：

```python
content = content_proj(torch.cat([
    normalized_rgb,
    aligned_sar,
    relation,
], dim=1))
```

区域上下文回写：

```python
region_context = soft_unpool(token, assignment)
residual = region_refine(torch.cat([base_fused, region_context], dim=1))
gate = building_prob.detach() * align_confidence.detach()
fused = base_fused + torch.tanh(gamma) * gate * residual
```

`gamma` 初始化为 0，确保插入模块时输出严格等于 A1。

## 9. 损失函数

总损失：

```text
L = L_A1_damage
  + λ_loc      L_A1_localization
  + λ_xy       L_coordinate_compactness
  + λ_building L_building_reconstruction
  + λ_evidence L_reliable_relation_reconstruction
  + λ_balance  L_region_size_balance
  + λ_boundary L_boundary_affinity
```

建议初值：

```text
λ_xy       = 0.10
λ_building = 0.50
λ_evidence = 0.20
λ_balance  = 0.02
λ_boundary = 0.10
```

约束：

- `L_building_reconstruction` 训练时可用 GT 建筑掩膜，推理时不需要 GT；
- `L_evidence` 只在高对齐置信度建筑 ROI 内重建 stop-gradient 关系特征；
- 不用 GT 损毁类别直接生成超像素边界；
- 最终 damage loss 可以通过软池化与重投影间接训练关联图。

## 10. 分阶段训练

### Phase A：A1 复现

严格复现 A1 Validation/Test 锁定结果，不增加模块。

### Phase B：A1-LCA

加载 A1，冻结父网络，只训练 LCA；`gamma=0` 初始化。验证局部错位鲁棒性。

### Phase C：RGB-SCN

暂时关闭 SAR 关系输入，只训练 RGB 深特征关联头和区域残差。

### Phase D：跨模态 SCN

加载 Phase C，开启 `confidence * aligned_relation`，先冻结 A1，再以低学习率联合微调。

### Phase E：Memory

只有 Phase D 在像素指标和超像素指标上通过后，才重建 Train Token/Memory；旧 SLICO Memory 不得直接复用。

## 11. 严格消融

```text
C0  A1
C1  A1 + LCA
C2  C1 + fixed RGB SLICO + region feedback
C3  C1 + RGB-feature SCN
C4  C1 + RGB + raw-SAR SCN
C5  C1 + RGB + aligned-relation SCN
C6  C5 without alignment-confidence gate
C7  C5 with shuffled SAR locations
C8  C5 with damage probabilities as partition input (risk control only)
C9  C5 + boundary affinity loss
```

关键比较：

```text
C3 > C2: 学习型超像素优于固定 SLICO
C5 > C4: 对齐后的 SAR 证据有效
C5 > C6: 可靠性门控有效
C5 > C7: 正确跨模态对应有效
C5 >= C8: 不复制分类答案仍能形成有效区域
C9 > C5: 边界损失有额外价值
```

## 12. 评价指标与准入

超像素本身：Boundary Recall、Boundary Precision、Under-segmentation Error、ASA、建筑边界跨越率、区域数和面积分布。

最终任务：Final mIoU、F1_clf、Damaged F1、四类 IoU、14 事件均值/最差值、人工 SAR 位移曲线。

进入 Memory 阶段的最低要求：

```text
C5 相对 A1 Final mIoU >= +0.30 pp
Damaged F1 不下降
Destroyed IoU 下降 <= 0.10 pp
至少 10/14 事件不下降
C5 明显优于 C7
建筑边界跨越率低于固定 SLICO
```

## 13. 参考实现来源

- SSN: https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html
- SCN: https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.html
- AINet: https://openaccess.thecvf.com/content/ICCV2021/html/Wang_AINet_Association_Implantation_for_Superpixel_Segmentation_ICCV_2021_paper.html
- CDS: https://ojs.aaai.org/index.php/AAAI/article/view/28454
- BRDG: https://openaccess.thecvf.com/content/CVPR2026/html/Ahmed_Boundary-Responsive_Differentiable_Gating_for_Superpixel-Based_Segmentation_CVPR_2026_paper.html
