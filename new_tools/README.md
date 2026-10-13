# BRIGHT New 目录离线工具包

本包同时更新 Memory、PRT initial debug、PRT sequence 和 region-night 的工具路径。安装位置：

```text
/root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/
├── RUN_TOOLS.py                 # 统一入口，默认只显示命令
├── RUN_MEMORY_REPLAY.py         # Memory V2 定位与校验，默认不运行模型
├── AUDIT_MEMORY_REPLAY_V2.py     # 原始 V2 探针，字节不变
├── memory_replay_debug/         # 原始 V1 Memory debug
├── prt_initial_debug/           # PRT 初始化诊断
├── prt_sequence_experiments/    # PRT 顺序诊断
└── region_night/                # 夜间实验调度与 residual 工具
```

你提供的五个旧目录作为独立目录检查，保留原文件：

| 原 New 子目录 | 本包对应工具 |
| --- | --- |
| `BRIGHT-memory-debug_esfI2YOk` | `memory_replay_debug`；V2 另用根目录探针 |
| `BRIGHT-prt-sequence_GgXF1Hmw` | `prt_sequence_experiments` |
| `BRIGHT-prt-debug_s67wQ0Di` | `prt_initial_debug` |
| `BRIGHT-prt-debug_EDXqo29z` | `prt_initial_debug` |
| `BRIGHT-region-night-unpack_HB4jnNVw` | `region_night` |

两个 PRT debug 旧目录统一使用可核验的原始 initial debug 版本。本包不在旧目录内替换文件，也不执行未核验的旧 ZIP。原始版本见 `SOURCE_ORIGINS.json`。

## 1. 下载、上传、安装

把 `bright_new_tools_v1.zip` 和对应的 `.sha256` 文件上传至：

```text
/root/shared-nvme/BRIGHT-master/New/
```

在服务器复制：

```bash
cd /root/shared-nvme/BRIGHT-master/New

if sha256sum -c bright_new_tools_v1.zip.sha256; then
  /root/shared-nvme/conda-envs/bright4090/bin/python -c 'import zipfile; exec(compile(zipfile.ZipFile("/root/shared-nvme/BRIGHT-master/New/bright_new_tools_v1.zip").read("INSTALL_NEW.py"), "INSTALL_NEW.py", "exec"))'
fi
```

安装器只使用 Python 标准库，不访问网络，不调用 `renameat2`，不运行模型。它校验安装器和全部包成员，独占创建新目录并校验写入内容。已有目录仅在全部文件一致时接受；不同内容或未完成标记会停止并保留现场。共享文件系统若不支持目录 `fsync`，允许跳过目录同步，文件写入与 SHA256 校验仍保留。

本次安装不需要 `KEY_OUTPUT_ROOT`、`DIAG_OUTPUT` 或临时解压目录名。

## 2. 检查路径

先查看统一清单：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py
```

默认只验证发布文件并列出路径，不启动任何子进程。下面逐条显示 PRT 与 region-night 的完整运行命令，也不会启动训练：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool prt-debug --plan-only
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool prt-sequence --plan-only
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool region-night --plan-only
```

PRT sequence 的 region-night 工具路径已改为包内同级目录，统一入口也显式传入该目录。

## 3. Memory V2：先定位，再重放

直接运行定位入口，进行真实路径和证据绑定检查：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_MEMORY_REPLAY.py --plan-only
```

这一步只用 CPU 读取已校验的证据元数据及源版本信息，不执行模型、不创建结果目录。它在原 checkpoints 与 `New` 中查找 `keys_smoke/summary.json`，只接受以下固定证据 SHA256：

```text
d05ca254bbdff51e0c7cc08c6dbce1aec1fac8b55a4dadbd8f6d0f7506306213
```

它核对 summary 中的相对证据路径、证据内容、E0 protocol SHA256 和已安装 V2 源码 SHA256。找到多份独立的匹配证据时会停止；用 `--evidence /实际根目录/keys_smoke/replay_evidence/00003.pt` 指定原失败文件。它不会选择“最新”目录或新建空实验根目录。

定位通过后，先执行 CPU 存档身份检查：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_MEMORY_REPLAY.py --run --cpu-only
```

CPU 检查通过、GPU 空闲后，执行原 V2 六次前向诊断：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_MEMORY_REPLAY.py --run
```

使用随包探针，不再触发 urllib 下载。结果写入原 checkpoints 下的新 `SRAC_MEMORY_REPLAY_AUDIT_V2/run_*`，完整路径会打印。此诊断不训练、不修改 bank、不访问 Test；`audit_complete` 仅表示审计完成，不表示检索或完整验证已通过。

V1 的旧失败案例可单独显示命令或执行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool memory-debug --plan-only
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool memory-debug --run
```

## 4. PRT 与 region-night

合成 CPU 检查：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool memory-debug --preflight
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool prt-debug --preflight
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool prt-sequence --preflight
```

真实 PRT 诊断需显式执行，PRT sequence 统一入口始终只运行 checks，不继续训练：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool prt-debug --run
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool prt-sequence --run
```

**下面这条会启动 region-night 原有训练队列**，执行前应先检查其 `--plan-only` 输出；它等待已有 GPU 工作完成，不终止其他作业：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/RUN_TOOLS.py --tool region-night --run
```

PRT 继续训练的条件和开关见 `prt_sequence_experiments/README.md`；各算法、训练预算、评估门限保持原版本。

## 路径与验证范围

默认仓库仍为 `/root/shared-nvme/BRIGHT-master`，Python 仍为 `bright4090`，数据、bank、权重与历史结果仍位于原位置。默认 E0 是：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json
```

默认 J1/J2/J3 来源是：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint
```

移动工具不需要改 protocol 内容或其 SHA。若原证据、模型或数据也移动了，应根据实际位置单独核对；本包不会批量替换 protocol 内记录的路径。统一入口支持 `--repo`、`--checkpoints`、`--tools-root`；Memory 另支持 `--protocol`、`--evidence`。

已验证 CPU 合成测试、离线安装、重复安装与搬移后的入口。真实服务器的 GPU、数据和 shared-nvme 文件系统尚未在本云工作区验证。
