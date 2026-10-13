# P/R 固定现场 GPU 对照

本工具只使用已经保存的 `check_02/scene.pt`。默认原报告目录为 `/root/shared-nvme/BRIGHT-checkpoints/SRAC_PRT_SEQUENCE_EXPERIMENTS_V1/run_20261012_175929_5f5edbb9`。

新证据：P 在最大差点的原分数与保存 F/权重的 CPU FP64 参考相差约0.07523，私有分数相差约0.000586；R 的原均值相差约0.003600，重算均值相差约6.7e-8，且区域质量正常、没有明显抵消。需要区分原调用时输入历史与固定算子计算，而不能仅凭这些数值宣布 TF32 根因。

## 执行

交付文件 `prt_fixed_scene_probe_v1.py` 是将 `probe_core.py` 和 `run_probe.py` 合成的独立单文件；ZIP仅包含这个脚本。上传ZIP到服务器 `/root/shared-nvme/BRIGHT-master/New/`，校验ZIP后解压到新目录，直接运行脚本，无需安装器或新依赖。

```bash
PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
PROBE=/你的解压目录/prt_fixed_scene_probe_v1.py
"$PYTHON" "$PROBE" --plan-only
"$PYTHON" -u "$PROBE"
```

启动前让目标GPU空闲。本工具不会终止其他实验，错误立即停止且不重试。每次创建新的 `SRAC_PRT_FIXED_SCENE_V1/run_*` 结果目录；已有源代码、现场、E0和案例库保持不变。不要移动运行中进程依赖的文件。

## 固定实验预算

按顺序运行四个独立新进程：

1. `legacy`：捕获时设置。
2. `no_tf32`：仅改变 cuDNN 的 `allow_tf32` 标志。
3. `cudnn_deterministic`：仅启用 cuDNN 的确定性标志。
4. `no_cudnn`：仅禁用 cuDNN。

每个进程都固定保存的 F/Q/ids、原/重算区域均值和权重。P 原/私有分类头各3次；R 在两份固定均值上分别用原/私有权重各3次；区域均值使用同一路由计划3次、重新建立同值计划3次。第一次调用也保留，没有 warmup 丢弃或重复至通过。输入与权重在前后做CPU独立快照哈希核对。

只调用分类器和原 `mean_pool`，不调用 A1 前端，不读 Train/Val/Test 图像，不读 GT、不建库、不训练、不重新生成 E0。固定概率输出只是当前场景对照，不能计作新的 Val 成绩。

在运行前验证完整现场SHA、所有内部张量/权重SHA、原源代码SHA、原E0、版本/设备/环境设置。P/R的参数梯度属性按原训练源码恢复（parent false、private true），所有运算都在 `no_grad` 中；`state_dict` 的分离张量元数据不能还原参数梯度属性。

## 输出

最后打印 `OVERVIEW=` 和 `SUMMARY=`。完整报告包含各模式的所有预定调用，最大差点与CPU参考的误差，以及输入/头前后哈希。

- `fixed_inputs_and_heads_unchanged`：固定输入和权重在前后没有变化。
- `p_fixed_repeat_max_abs`：同一模式下，固定 F 的原/私有分类分数之间最大差。
- `p_first_at_saved_point`：保存最大差点的新分类结果和CPU参考误差。
- `r_fixed_repeat_max_abs`：相同固定均值输入上，Linear重复差异。
- `pool_fixed_repeat_max_abs`：固定F/Q的区域汇总重复差异。
- `pool_first_vs_captured_original/recomputed`：新汇总与保存的两份旧结果分别比较。

所有张量门槛仍是 `atol=rtol=2e-5`。`diagnostic_complete=true` 仅表示预定观察完成；`historical_failure_resolved` 与 `downstream_authorized` 始终为false。

四个新进程不能重建旧分配/调用历史。cuDNN标志并不直接作用于 grouped gather/multiply/sum；如果pool跨模式不同，需检查工作区/分配历史等，不能说TF32直接修复了求和。新GPU固定输入若未复现，也不能宣布旧问题已解决。

CPU合成测试及四进程启动检查只能验证代码流程；真实RTX4090/CUDA仍需运行本工具验证。
