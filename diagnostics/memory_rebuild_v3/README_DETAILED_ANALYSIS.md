# V3 详细结果分析：直接复制版

用户当前错误来自缺少 SUMMARIZE_MEMORY_REBUILD_V3.py.sha256，Python 未执行。ANALYZE_MEMORY_V3_RESULTS.py 是自包含的标准库详细分析脚本，不依赖旧 helper、SHA 文件、torch 或实验模块。

默认结果目录：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665
```

可以把脚本完整复制到 `python - <<'PY' ... PY` 中直接运行。也可将源文件放到服务器 `New/` 后运行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/ANALYZE_MEMORY_V3_RESULTS.py
```

其他结果目录用 `--root /实际目录`。脚本只读现有小型 JSON，不读 NPZ、大 bank、数据或权重，不训练、不跑前向、不修改 V3 与结果。它验证记录的完成标记、正式检查与小型元数据来源绑定；这些不是对所有原始产物的重新校验。

输出涵盖全部七组像素指标、四组检索及归一化类别证据、前景分层、四组救回/误改与 FP 变化、逐事件 IoU、支持库存及跨事件支持量、邻居重合、查询有效性。统计缺项/零分母显示 N/A；完整状态或来源绑定错误时退出并指出原因。

软相容度是分布点积，不是像素准确率；类别软覆盖以对应查询软标签质量为分母。救回率以该类基线错误像素为分母，误改率以基线正确像素为分母，不能直接相减；净 TP 必须结合 FP 才能分析 IoU。原结果没有逐事件 transitions，因此逐事件部分只显示原始 IoU。裸投票不是训练后 Reader 收益。

本工具用于整理已经完成的 E1，不自动选择 key 或启动训练。随机支持/标签打乱对照仍需独立的后续区域证据分析，Reader 则属于 E2。

验证：15项专属标准库测试通过（python -m unittest -q test_analysis.AnalysisTests）；原V3合成450图完整schema兼容与stdin执行通过，4组joint校核一致、结果字节不变。合成验证不是一次真实BRIGHT GPU评估。
