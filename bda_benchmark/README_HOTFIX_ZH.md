# CMR-V1 component replay hotfix

## 为什么需要这个补丁

P0 历史缓存只保存了建筑 component 的 `bbox / area / Raw-Mean key`，没有保存完整二值 mask。
V1 最初只允许 `bbox + area` 完全相等的 replay。冻结 P3 在不同 CUDA/cuDNN 执行中，建筑边界可能出现极少数像素级数值漂移，于是同一建筑会因为 1--2 个边界像素变化而被误判为 replay 失败。

本补丁不使用宽松 bbox 替代。匹配策略是：

1. 优先 exact `bbox + area`；
2. exact 失败时，候选必须同时满足：
   - bbox IoU >= 0.90
   - symmetric area ratio >= 0.90
   - 重新计算的 Frozen-P3 Raw-Mean key 与 P0 缓存 key cosine >= 0.995
   - 多候选时最佳 key cosine margin >= 0.001
3. verified fallback 占全部 replay 的比例默认不能超过 2%；超过立即报错。

因此它仍然是 identity-safe replay，而不是用相近 bbox 偷换实例。

## 安装

在工程根目录：

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
unzip -o /path/to/CMR_V1_replay_hotfix.zip
```

只覆盖 CMR-V1 隔离命名空间和独立 runner，不修改旧 CMR / P3 / S7 / S9。

## 运行

```bash
bash run/run_CMR_V1_isolated.sh preflight
FORCE_CACHE=1 bash run/run_CMR_V1_isolated.sh build
```

成功时每个 cache 会额外打印：

```text
Replay exact   : ...
Replay verified: ...
```

并在输出 `.pth` 的 `component_replay` 字段中记录每个 verified fallback 的 image、source_row、bbox_iou、area_ratio、key_cosine、key_margin。

如果仍失败，不要降低阈值；保留报错的 image/row，再针对该行做 P0/P3 preprocessing 一致性诊断。
