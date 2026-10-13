# Memory 数值对照：GitHub 下载与运行

这份交付位于独立分支 **`codex/joint-pool-numeric-audit-v2`**，不在 `main` 分支。

## 下载入口

- [打开交付分支](https://github.com/dadadadadadaning/bright/tree/codex/joint-pool-numeric-audit-v2)
- [查看单文件 RUN_POOL_NUMERIC_AUDIT_V2.py](https://github.com/dadadadadadaning/bright/blob/ce549f2d1f99142003b8d7e1a78f5165792e6e17/RUN_POOL_NUMERIC_AUDIT_V2.py)
- [查看完整源码与说明](https://github.com/dadadadadadaning/bright/tree/ce549f2d1f99142003b8d7e1a78f5165792e6e17/joint_pool_numeric_v2)
- [下载固定版本的仓库 ZIP](https://github.com/dadadadadaning/bright/archive/ce549f2d1f99142003b8d7e1a78f5165792e6e17.zip)

如果 Raw 文件地址无法访问，推荐下载仓库 ZIP；也可以打开交付分支，在绿色 **Code → Download ZIP** 下载。解压后，找到仓库顶层的 **RUN_POOL_NUMERIC_AUDIT_V2.py**，把这一个文件上传到服务器：

```
/root/shared-nvme/BRIGHT-master/New/RUN_POOL_NUMERIC_AUDIT_V2.py
```

脚本包含全部运行辅助代码，运行时会自动解压到自身旁边的新目录；不需要上传整个仓库，不需要覆盖 BRIGHT-master，不需要安装包或连接网络。

## 校验后运行

```bash
cd /root/shared-nvme/BRIGHT-master/New

if printf '%s  %s\n' \
  0a89eb4df2e8de2ad098233f8b3c6698ecce58e6d9fbe3d8c8201eb1b1b0305e \
  RUN_POOL_NUMERIC_AUDIT_V2.py | sha256sum -c -; then

  /root/shared-nvme/conda-envs/bright4090/bin/python \
    -u RUN_POOL_NUMERIC_AUDIT_V2.py
fi
```

无需另行下载 `.sha256` 文件；上面的校验值绑定已测试的单文件交付。

## 服务器能够访问 GitHub 时的备用方式

以下命令把交付分支克隆到新的临时目录，不切换现有 BRIGHT 仓库的分支：

```bash
NUMERIC_GH_DIR="$(mktemp -d /root/shared-nvme/BRIGHT-master/New/pool_numeric_github_XXXXXXXX)"

if git clone --depth 1 --single-branch \
  --branch codex/joint-pool-numeric-audit-v2 \
  https://github.com/dadadadadaning/bright.git "$NUMERIC_GH_DIR"; then

  if printf '%s  %s\n' \
    0a89eb4df2e8de2ad098233f8b3c6698ecce58e6d9fbe3d8c8201eb1b1b0305e \
    "$NUMERIC_GH_DIR/RUN_POOL_NUMERIC_AUDIT_V2.py" | sha256sum -c -; then

    /root/shared-nvme/conda-envs/bright4090/bin/python \
      -u "$NUMERIC_GH_DIR/RUN_POOL_NUMERIC_AUDIT_V2.py"
  fi
fi
```

## 它会执行什么

默认只读取本次恢复失败目录：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_RECOVERY_V1/run_20261013_113711_128984_3991264
```

程序从 failure.json 定位当前 call2 的输入，并对比原 CPU FP32、CPU FP64、原 CUDA 与连续索引 CUDA。CUDA 对照会等待 GPU 空闲，每种后端在独立子进程执行。同时只读采集两组 SAR 的状态与失败日志。

结束时打印 `PASTEABLE_REPORT=.../report.txt`。把这个报告内容贴回即可分析。`execution_complete` 仅表示诊断调度结束，不能代替数值检查通过，更不代表训练完成。

**它不训练、不重建 Memory 库、不修改原源码或旧 checkpoint，也不放宽数值门槛。** 实际 GPU 故障尚待服务器对照；CPU 测试通过不等于 GPU 修复成功。
