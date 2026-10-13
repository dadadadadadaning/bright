# V3 已保存结果的只读汇总

本工具独立于 V3 源码，只读小型 JSON，stdout 输出，不导入实验代码、torch，不读大 bank/NPZ/数据，不运行模型/GPU，不修改结果。默认运行根目录为 run_20261013_100220_049144_3778665。用于提取终端报告省略的类别分母、逐事件结果，以及两个 Top64 标签投票相对 grid_probq 的纠错/误改统计。

将 SUMMARIZE_MEMORY_REBUILD_V3.py 和同名 .sha256 文件手动下载，放到服务器 /root/shared-nvme/BRIGHT-master/New/，在该目录校验：

```bash
cd /root/shared-nvme/BRIGHT-master/New
sha256sum -c SUMMARIZE_MEMORY_REBUILD_V3.py.sha256
```

校验通过后运行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/SUMMARIZE_MEMORY_REBUILD_V3.py
```

其他根目录添加 --root /实际路径。不要放进 V3 实验源码目录，以免改变原实验的源码哈希。

完成状态、no-training/no-Test、新库来源与小型 JSON 绑定必须一致。缺统计字段显示 N/A，分母为零显示 N/A。这里只汇总已保存记录，没有重新校验大 bank、权重、源代码、全部预测文件或原数据；不自动推荐 key，不宣称训练 Reader 已有效。

标准库测试：python -m unittest -q test_summary。34项测试通过，包括手算转移表、联合计数核对、类别分母、逐事件差值、scope/绑定及只读行为。另通过 V3 合成450图完整schema兼容，已有结果不变；不是一次真实 BRIGHT GPU 验证。服务器这份原始 result.json 的实际输出需用户运行后查看。
