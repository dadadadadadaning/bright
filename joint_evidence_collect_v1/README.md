# BRIGHT 只读证据收集

将单个 `collect_joint_evidence_v1.py` 文件放到服务器的 `New` 目录，用 Python 3.9+ 运行一次：

```bash
python3 /root/shared-nvme/BRIGHT-master/New/collect_joint_evidence_v1.py
```

脚本打印 `REPORT:` 和 `ZIP:` 的完整路径。可直接粘贴 `REPORT.txt`，或上传 ZIP；ZIP 包含原始报告快照与 `inventory.json`，方便检查原始 `metrics`、`per_event`、状态与缺失情况。

目录不同时通过参数指定：

```bash
python3 collect_joint_evidence_v1.py --root /path/to/original/run --diagnostic /path/to/existing/diagnostic --output-parent /path/to/new/outputs
```

默认训练目录为 `SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D`，默认诊断目录为 `SRAC_JOINT_GENERALIZATION_DIAGNOSTIC_V1/full_20261016_090249_389262`；完整默认路径见源码常量。

仅使用 Python 标准库，读取固定白名单中的已有 Train/Val JSON、诊断摘要/traceback 和三个日志尾部。不会扫描源目录，不加载 torch/权重/图像，不运行 CUDA、训练、评估、其它脚本或网络请求，也不会自动重启。输出只写入指定父目录中新建的唯一 `evidence_collect_时间_随机串/` 及同名 ZIP；禁止将输出父目录放入任一源目录。

原报告每个最多 8 MiB；超过上限会记录 `oversize` 并跳过。日志只取最多 128 KiB 的末尾字节。`train/train.jsonl` 流式读取启动时已有内容，最多 128 MiB，每行最多 1 MiB；这些上限、文件变化、重复/残缺行、字段缺失/无效状态都会写入清单。每个 step 只使用首个可解析对象行，后续同 step 行只记为重复。训练 journal 不复制到 ZIP；只汇总指定窗口的 loss/lr/grad_norm 以及已提供的 event/image 计数。JSON 报告不解析为任意代码，也不追踪其中引用的文件路径。用户提供的根目录先解析为实际目录，支持挂载/链接别名；其下报告路径若再通过符号链接跳转则拒绝。

`NA` 表示缺失或不可用，不代表 0。文件缺失不代表从未运行；不完整诊断不会被当作已完成。不同训练窗口采样的图像可能不同，窗口 loss 不可单独用来诊断过拟合。该工具仅汇总文件快照，不验证权重、数据或流水线完成性。

本地验证：

```bash
python3 -m unittest discover -s . -p 'test_collect.py' -v
```
