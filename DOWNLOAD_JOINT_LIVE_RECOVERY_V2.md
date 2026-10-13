# 重新启动三组 Memory 续训：离线入口 V2

此前重复失败的位置是 `Original-semantics CPU pooling reference gate failed`。这份代码先保存实际模型前向同一次调用的 GPU 输出与 CPU 参考，再验证**明确指定的 CPU FP32 区域汇聚后端**；主干和 Reader 继续使用 GPU。它保留原 checkpoint 1500，完成完整 Val450 重评后才更新第 1501–5000 步。

这属于改变汇聚执行设备的修复续训，可能变慢；不是已证明根因解决的原 CUDA 训练。原源码、库和失败目录保留。

## 下载与启动

打开本仓库根目录 `RUN_JOINT_LIVE_RECOVERY_V2.py`，使用文件右上角 **Download raw file** 保存文件。也可下载本分支 ZIP，解压后取根目录同名文件。

将文件上传到：

```
/root/shared-nvme/BRIGHT-master/New/RUN_JOINT_LIVE_RECOVERY_V2.py
```

然后仅运行一次：

```bash
cd /root/shared-nvme/BRIGHT-master/New
if printf '%s  %s\n' \
  2b7d14e291cdba8c6a75871207e16c04872fa9b74014abfd9fe84878bab61f1e \
  RUN_JOINT_LIVE_RECOVERY_V2.py | sha256sum -c -; then
  /root/shared-nvme/conda-envs/bright4090/bin/python RUN_JOINT_LIVE_RECOVERY_V2.py
fi
```

不需要单独 `.sha256` 文件。入口内嵌全部工具，内部再次核对每个文件；**运行时无需网络下载**。提取到新的 `New/joint_live_...` 目录，不覆盖旧工具。

默认后台运行并打印 PID、工具目录、启动日志。不要同时运行旧 `start_recovery.sh`。启动器有新旧共享锁，但锁不代替 GPU 空闲检查。

## 看懂日志顺序

```
live-probe       实际 CUDA 模型前向现场取证（零更新）
live_analysis    当次输出与离线重算对照
cpu-probe        CPU 汇聚的单图完整前向，三组 Reader（零更新）
cpu_analysis     独立 FP64 对照，原容差
train           先 checkpoint_revalidation：1500 步 Val450
                再 Joint recovery fit：1501–5000 步
                最后 final_val 和完整性检查
```

看到 PID、`START train` 或探针 `complete` 不代表训练已经更新。只有 `Joint recovery fit` 的 step≥1501 才是本次新增更新。`report.txt` 和 `train/summary.json` 会区分旧1500步与新增步数。

新结果目录为：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_LIVE_RECOVERY_V2/run_...
```

本工具已通过101项CPU测试、8项离线入口测试及生产规模合成CPU数值检查。真实服务器GPU、checkpoint1500完整Val450和后续训练尚未在云工作区运行。任何检查失败都会停止，不跳图、不扩大容差。

详细说明与全部可审阅源码：[diagnostics/joint_live_recovery_v2](diagnostics/joint_live_recovery_v2/README.md)。
