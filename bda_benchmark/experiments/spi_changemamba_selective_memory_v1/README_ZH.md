# ChangeMamba Selective Memory V1

目的不是用 ChangeMamba 替换 P3，而是检验其分类前变化特征能否提供 P3 缺少的独立证据，并用低容量 Verifier 控制回写风险。

## 三个 Token 对照

- `k1_local`：Change Decoder 最浅层、最高分辨率变化特征。
- `k2_multiscale`：局部变化特征 + 较深层语义变化特征。
- `k3_structural`：K2 + RGB建筑结构特征 + Core/Ring多环证据。

三个变体使用相同的 P3 建筑图和 SLICO 区域，确保差异只来自 Token 内容。P3 最终概率不进入检索 Key。

## 安全回写

`v1_direct` 只允许 ChangeMamba Direct Head 提议；`v2_memory` 同时允许 Direct 和 Memory 提议。Verifier用Train的mIoU效用标签训练，Val只读评估。阈值候选包含1.01，即拒绝全部候选并完整回退P3。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

CM_ROOT=/root/shared-nvme/BRIGHT-master/ChangeMamba-master \
CM_CKPT=/实际路径/best_model.pth \
bash experiments/spi_changemamba_selective_memory_v1/run.sh preflight
```

先用2张Validation图像验证双模型权重和特征接口：

```bash
CM_CKPT=/实际路径/best_model.pth \
bash experiments/spi_changemamba_selective_memory_v1/run.sh smoke
```

依次完成三个Token裸审计：

```bash
nohup env CM_CKPT=/实际路径/best_model.pth \
bash experiments/spi_changemamba_selective_memory_v1/run.sh stage_all_tokens \
> /root/data/BRIGHT-train-logs/cm_memory_tokens.log 2>&1 &
```

查看结果后，只有K3合理才运行安全回写：

```bash
nohup env CM_CKPT=/实际路径/best_model.pth \
bash experiments/spi_changemamba_selective_memory_v1/run.sh stage_safe_k3 \
> /root/data/BRIGHT-train-logs/cm_memory_safe_k3.log 2>&1 &
```

禁止访问Test。只有`v2_memory`在Val450超过`v1_direct`且通过Damaged F1、事件数和接受率检查，Memory才能保留。
