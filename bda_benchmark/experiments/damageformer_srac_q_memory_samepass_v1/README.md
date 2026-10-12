# Q × Memory 同次前向诊断

这个独立包解决旧 Q 分层诊断的一个额外严格条件：它要求第二次运行的 J3 Top-8 **行号逐位等于 E0**。首张 `bata-explosion_00000018` 的 `mismatch.json` 显示，1024 个查询中 5 个排名改变、10 个槽位变化，但邻居集合均未改变；同一进程的前端重复计算有 token 最大差值约 `2.93e-5`、Q 最大差值约 `1.94e-6`。这些数值说明精确行号重放对微小数值变化敏感，**不能由此判断是哪一个 CUDA 算子造成，也不能认定 Memory 效果有效或无效**。

本包仍要求已通过的 `legacy/native/all` E0；可选 E1 也必须通过 450 张、复现和不变性检查。E0 固定权重、源码、数据列表、设备和数值协议。每张 Val 图片只做一次冻结前端前向，同一个 Q/token 当场生成 J2/J3 Top-8 和 A1、直接区域、J1/J2/J3 五组预测。**Val 真值只在五组预测与支持 ID 全部固定后**用于混淆矩阵和 Q 混合度分层。新运行的 Top-8 行号可以与 E0 不同，且不会据此失败；保存每张图片的实际支持 ID 和五组预测，保证解释的是这次真实使用的 Memory。

正式结论仍要满足：完整 Val450，原训练参考的所有类别 IoU、precision、recall、F1 和 mIoU 在原 `1e-5` 绝对误差内复现；E0/E1 身份、旧源码、权重、输入、运行数值条件、新包源码和池化补丁均不变。`audit.json` 还单独报告与 E0 指标的差异，但**两个分别通过原参考门槛的运行彼此不必落在 `1e-5` 内**，故这个二次比较不作为新门槛。任何正式门槛失败均保留报告并标记 `complete=false`、`full_validation_evidence=false`，不会放宽原门槛。

Q 分层与旧诊断同定义：前景占比 `<0.2`、`[0.2,0.8)`、`>=0.8`；前景类别纯度 `<0.8`、`[0.8,0.95)`、`>=0.95`，另列无前景或无标签。硬归属只用于易解释的像素分组；另报告 Q 软路由暴露度，其中“前景类别混合”的软暴露目前专指 **前景占比 `>=0.8` 且前景类别纯度 `<0.8`** 的区域。交叉分层仍覆盖其它前景占比下的类别混合。这里统计的是像素和 Q 区域，不是建筑实例；相关性不证明 Q 混合是误报的原因。

服务器执行（将整个新目录同步到仓库，不改旧包）：

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
bash bda_benchmark/experiments/damageformer_srac_q_memory_samepass_v1/run.sh preflight
export RUN_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130
export E0_PROTOCOL="$RUN_ROOT/protocol"
export E1_EVIDENCE="$RUN_ROOT/evidence"
export Q_SAMEPASS_OUTPUT="$RUN_ROOT/q_memory_samepass_$(date +%Y%m%d_%H%M%S)"
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC Q_SAMEPASS_LIMIT
bash bda_benchmark/experiments/damageformer_srac_q_memory_samepass_v1/run.sh diagnose
```

输出包含 `summary.json`、`audit.json`、`images.jsonl`、`predictions/*.npz`。逐图文件保存本次的 Q、区域索引、区域四类标签比例和有效质量，以及五组预测与 J2/J3 邻居；结束前核对 450 条记录的顺序和文件 SHA256。请用全新目录运行，不覆盖旧的失败证据。`preflight` 只跑 CPU 合成测试，不是 Val 结果；本地没有服务器图像/权重，不能代替服务器 Val450。若要先调试两张，使用全新的目录并设置 `Q_SAMEPASS_LIMIT=2`，结果的 `full_validation_evidence=false`。
