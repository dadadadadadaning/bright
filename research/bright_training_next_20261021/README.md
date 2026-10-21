# 第1500步续训，以及独立的 D/H 无支持读取对照

此包针对已完成独立 Val450 评价的六分支实验，提供两个训练入口。不会修改旧实验、旧 checkpoint、原模型代码、数据或 Memory bank。不会读取 Test。

## 先运行六分支续训

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B -u /tmp/bright_training_next_v1.pyz resume
```

固定来源：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/resume_20261020_161047_4184092
checkpoints/checkpoint_001500.pt
SHA256 ce9a4fd3876ed947bb6399fe96bcba5137f9ea44c96fee1629c60df738691b3d
```

启动前核对已有独立 Val450 评价的 summary、完整结果及逐图日志：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1/eval1500_y5qlliyp
```

e/f/b/d/r/l 六个分支均恢复第1500步模型、AdamW、RNG及同一训练序列，继续到2500步。新目录保存在原 `SRAC_SLOT_MEMORY_ABLATION_RESUME_V1` 下，启动器输出具体路径。

原续训合同会先重新评价恢复边界的 Val450 和固定 Train 子集，然后开始第1501步。之后评价2000/2500步，训练结束评价固定 Train 子集。**最初看到验证进度，不表示启动成了仅评价任务。** 不沿用旧结果冒充本次评价；历史失败目录保持原状态。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_training_next_v1.pyz logs
```

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_training_next_v1.pyz status
```

日志命令根据启动器更新的 `latest.txt` 定位目录，不依赖 SSH 会话变量。Ctrl+C 只退出日志查看。完成需同时看到 `complete=true`、`training_complete=true`、`full_validation_evidence=true` 和 `last_complete_step=2500`。

## 再运行独立 D/H 对照

原实验完成后运行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B -u /tmp/bright_training_next_v1.pyz baseline
```

这是一次新的配对实验：D/H 都从相同 seed3407 新初始化训练0→2500，不复用旧 D 的已训练权重或优化器。D 保留原 soft 双槽和合法 Memory 读取；H 将 `_read` 返回的支持上下文置为零，冻结仅属于 Reader 的投影/标签嵌入参数。

H 仍训练当前图像的像素、父区域、槽编码器和联合分类器，名称是“无支持读取的后端适配”。它不是只训练最后一层。H 保留候选准备/校验的计算框架，因此不宣称相同计算量或推理加速。它同时排除支持特征和支持标签对输出的贡献，区别于仅置乱标签的 L 或仅关闭标签嵌入的 N。

D/H 使用相同图像顺序、训练预算和损失。评价包含 Grid、D、H，保存四类混淆矩阵、逐图/逐事件统计、D−H 配对转移。按原配方在0/500/1000/1500/2000/2500步评价 Val450，固定 Train 子集在起点和终点评价。2500步为预设终点；如另做检查点选择，必须统一用 Val 四类 mIoU 规则。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_training_next_v1.pyz baseline-logs
```

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_training_next_v1.pyz baseline-status
```

新目录位于 `SRAC_D_H_NO_SUPPORT_READ_V1/pair_*`，不混入原六分支。两个入口共享运行锁，防止它们互相并发；该锁不管理其他无关训练程序。D/H 每50步保存一次新格式 checkpoint，保留模型、AdamW、RNG和日志校验。此版本的 `baseline` 只做从零启动；H checkpoint 加载必须重新安装无支持读取逻辑，**不能用六分支续训入口加载 D/H checkpoint**。

## 保持原配方的部分

- 原 NLL＋present-class Lovasz，KL=0，梯度裁剪1.0。
- 原 AdamW、weight decay=0.005、最高学习率1e-4、500步 warmup、5000步调度长度及原 post500 倍率。
- 原冻结前端、Q、分槽、Memory、Train/Val名单和支持排除规则。
- 基础续训程序完整取自 SHA256 `0f7fd0a85ca2db5727509875c84c1beba1011d2bd4300a8b6b5ea86b59e75f07` 的旧修正版。`BASE_HELPERS.json` 记录每个原始模块的字节校验，保留已修正的 NLL 汇总校验。

新增观察器只记录验证顺序和失败条件，不 clamp、不重新归一化、不放宽原概率校验。若再次遇到 `grid_probq` 问题，进程失败并保存 `evaluation_failure.json`；普通 ValueError 时尽量保存 CPU 概率案例，CUDA异常后不继续读取GPU张量。此前一次独立评价通过不代表历史异常根因已经解决。

只有所有计划评价及最终完整性检查通过才把新结果标成完成。失败时 checkpoint 保留，旧报告不被改写，也不会自动重试或自动重训。

## 验证范围与后续判读

`test_training_next.py` 覆盖真实原模型的 H 支持独立性、局部图像依赖、真实 AdamW 更新、checkpoint/RNG往返、原续训成功与失败恢复、全450项合成评价、失败时不提升报告、以及旧助手模块字节一致性。CPU合成数据不构成真实服务器CUDA或精度验证。

先比较同一新实验中的 D 和 H：D−H 若接近零，则缺乏支持读取贡献的证据；若有可复现收益，再对照已有 E/F/B/D/R/L 区分分槽和 Memory 的来源。不能通过只展示校准前后某个类别涨分证明方法有效。多种子和固定方案 Test 是后续步骤，此包不执行 Test，也不自动调整类别权重、温度或分类偏置。

构建：`python build.py /path/to/bright_slot_memory_resume_v1.pyz`。

测试需设置 `BRIGHT_RESUME_TEST_SOURCE`、`BRIGHT_ABLATION_TEST_SOURCE`、`BRIGHT_ABLATION_REFERENCE` 指向对应固定源码，然后在有 PyTorch 的环境运行 `python -B test_training_next.py -v`。先构建再测试；测试实际导入构建后的 pyz。
