# ChangeMamba Deep Region Retrieval V1

这是一个完全隔离的机制验证协议，不修改原始 DamageFormer、ChangeMamba 或旧实验。

## 固定研究问题

1. `SLICO > Grid`：结构区域是否比规则网格有效？
2. `Structural > Base`：RGB结构、ChangeDecoder和DiffFusion联合表示是否优于单一变化特征？
3. `Attention > Mean`：深度区域池化是否优于平均池化？
4. `RAG > Direct`：历史实例是否给当前区域分类带来独立增量？
5. `Correct > Controls`：增量是否确实依赖正确标签和正确邻居？

## 严格边界

- 主干、建筑先验、区域状态全部来自同一 B2 ChangeMamba；不加载P3。
- preflight 会检查 `diff_fusion`、`diff_proj` 和 `diff_gamma`，拒绝把B0权重冒充B2。
- GT只在区域生成之后提供Train监督或Val审计标签。
- Memory只由Train建立，Val只查询，Test永久禁止。
- 主配置预先固定为 `SLICO + c4_structural + p3_attn_tail + x1_label_geometry`。
- 本阶段只做区域级机制准入，不能宣称最终像素mIoU提升。
- 只有RAG同时超过Direct及三种负对照，才允许开发可学习超像素和像素软回投。

## 运行

```bash
bash experiments/changemamba_deep_region_retrieval_v1/run.sh preflight
bash experiments/changemamba_deep_region_retrieval_v1/run.sh all
```

输出独立写入：

- `BRIGHT-checkpoints/CHANGEMAMBA_DEEP_REGION_RETRIEVAL_V1`
- `BRIGHT-eval-results/CHANGEMAMBA_DEEP_REGION_RETRIEVAL_V1`
- `BRIGHT-train-logs/CHANGEMAMBA_DEEP_REGION_RETRIEVAL_V1`
