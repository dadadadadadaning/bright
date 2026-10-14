# 固定 Q 的双槽区域与 Memory 联合训练

版本：`SRAC_SOFT_MEMORY_JOINT_V1`，2026-10-14（Asia/Shanghai）。

这是独立的 **A/B/C/D 四组实验工具**，复用服务器已有 E0 协议、V3 FP32 父案例库与 BRIGHT 代码。代码放在 `/root/shared-nvme/BRIGHT-master/New`；不会安装或覆盖原 `bda_benchmark`，不改旧库、旧报告或失败状态。

本修订版修复 smoke 子库的父案例选择：原 V3 排序器仅根据 T128/F256 非零范数判断可检索性，不读取新双槽库的 `written` 标记。现在只在临时检索视图中将未写父行的两种 key 置零，保留原父库、行号以及选后合法性检查。此前失败目录不能续跑，需使用新工具目录和新输出目录重新运行 `check`。

已实现真实数据建库、smoke、训练和 Val 驱动，但云端只有 CPU。发布前的测试结果见 `CPU_VALIDATION.json`：**CPU 合成测试不是 BRIGHT/CUDA 验证，也不是实验涨分结果。** 已知旧 native pooling 越界不在本包中被修复或绕过，发生时会停止并保存现场。

## 现在先做什么

原 R0/RT/RF 尚未正式完成，因此目前使用：

```bash
bash start_server.sh check
```

这只执行预检、4 张 Train 来源图像的双槽小库、4 个头各 20 步真实数据 smoke 及 2 张 Val 检查，不执行 5000 步正式训练。它会占用 GPU；请在旧任务不使用该 GPU 时运行。启动器不会停止或等待其他任务。

后续原三组完成，再使用 `full`。正式入口会只读检查原三组末步报告、5000 步权重、Val450 和同一 E0/父库绑定；没有“跳过失败检查”选项。这个顺序来自本次已确定的实验设计，不是需要额外批准。

## 四组含义

| arm | 双槽规则 | 参考来源 |
|---|---|---|
| a | 固定 4×4 特征格棋盘划分 | 当前图像空间近邻区域 |
| b | `Q*(1-B)` 与 `Q*B` | 当前图像空间近邻区域 |
| c | 固定空间划分 | 跨事件 Train 历史案例 |
| d | soft 建筑概率划分 | 跨事件 Train 历史案例 |

默认用原 T128 key 查找 Top64 父案例。A/B 共用父空间近邻 ID；C/D 共用一次父 key 检索结果。每个父案例展开两个槽，最多 128 个支持槽，仍只有 64 个父案例。F256 key 可另用 `--key-name f256` 做独立同配置实验，不能将不同 key 的零散组拼成四格。

每个模型保留像素 F256、父区域 F256/T128、子槽与 Reader 回投，以及原 `grid_probq`。父 T128 只是父上下文，不是重建的子槽 T128。Reader 可读取两种参考槽，没有 Memory 采纳/拒绝门控。

建筑槽和背景槽并不限制类别；背景槽也可以含 Damaged 标注。槽特征与 Train 软标签同权重，255 不进入标签，按有效标签质量归一化；R0 式图内槽只使用 A1 预测分布，不读取查询真值。两槽各用自己的 route 权重回投，不能都用原 Q。

默认四个头：共同初始化、AdamW(lr=1e-4, weight_decay=.005)、同训练图序、NLL+Lovasz、梯度裁剪 1、FP32、5000 次完整图像更新。无新增类别重加权、数据增强、Q 训练或实例划分。末层零初始化，起点核对同次 grid_probq。主比较是预定末步 Val450 前景三类 mIoU；中间 Val 不作为各组分别选最佳的依据。

## 文件与输入

核心文件：

- `slots.py`：双槽路由、汇总、标签和回投。
- `model.py`：四组同结构的联合分类器，支持表只编码一次，按父查询分块读取。
- `build_values.py`：固定成员的新双槽 value，包含独立 CPU FP64 标签参考。
- `server_runtime.py`：同次冻结前向、原生 B 捕获、父检索、图内预测标签和来源检查。
- `base_runtime.py`：打包的原 V3 只读桥接；仍调用服务器已有 BRIGHT 模块。
- `train.py`：四组训练、末步序列化权重重新加载、预测保存与独立回读。
- `gates.py`：查找、核对旧三组完整结果。
- `run.py`、`start_server.sh`：前台命令与后台启动。

默认输入：

```
Python: /root/shared-nvme/conda-envs/bright4090/bin/python
repo: /root/shared-nvme/BRIGHT-master
V3: /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665
E0: 从 V3 pipeline_state.json 读取；否则使用既有 fresh_e0/protocol.json
```

需要 V3 的完整 `keys/paired_bank.pt`、manifest、summary、protocol 和 images journal；不要求把旧失败的 `diagnose` 流水线伪装为成功。父库本身必须完整、通过已有严格装载。

新 value 保存原父成员 ID 和两个规则的槽内容，不覆盖父 key。完整构建遍历原父成员的 Train 来源图像；1184 来源图像不是全部 Train，正式训练仍覆盖 Train2918。smoke 保留全父行号但仅四张跨事件来源标记 written，未写行不可检索，不能进入正式训练。

## 解压与只读核对

先把修订版 `bright_soft_memory_joint_v1_fix1.zip` 上传到：

```
/root/shared-nvme/BRIGHT-master/New/bright_soft_memory_joint_v1_fix1.zip
```

使用发布消息中的 SHA256 校验，然后解压到新的临时目录，保留稳定的工具路径记录：

```bash
(
  set -euo pipefail
  PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
  ZIP=/root/shared-nvme/BRIGHT-master/New/bright_soft_memory_joint_v1_fix1.zip
  test -s "$ZIP"
  UNPACK="$(mktemp -d /root/shared-nvme/BRIGHT-master/New/soft_memory_XXXXXXXX)"
  "$PYTHON" -m zipfile -e "$ZIP" "$UNPACK"
  TOOL="$UNPACK/soft_memory_joint_v1"
  "$PYTHON" "$TOOL/run.py" plan
  "$PYTHON" "$TOOL/run.py" find-baseline
  printf '%s\n' "$TOOL" > /root/shared-nvme/BRIGHT-master/New/soft_memory_tool_path_fix1.txt
)
```

`plan` 只读展示四组定义，不访问 GPU、不开始建库。启动检查和实际执行入口会验证包内 SHA256。`find-baseline` 只读显示旧三组进度及不符合正式条件的原因；即使旧实验失败也能查看。

## 当前可运行：检查新工具

即使删除原终端，也可从记录文件恢复工具位置，不需再次 `read -p`：

```bash
TOOL="$(cat /root/shared-nvme/BRIGHT-master/New/soft_memory_tool_path_fix1.txt)"
bash "$TOOL/start_server.sh" check
```

流程：独立预检进程 → `values_smoke` → `smoke`。全部通过后 `complete=true` 只代表本次 check 完成，`full_validation_evidence=false`。原三组不完成不会阻止 check。

脚本打印 PID、结果目录、日志及 report 命令。后台 PID 不是训练成功；应查看日志中的建库、初始检查、20 步更新与最终检查。

输出默认：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_JOINT_V1/run_..._check_...
```

新槽增加注意力激活，显存和时间要以真实 smoke 的 `peak_cuda_allocated_bytes`、`elapsed_seconds` 为准，不保证与旧三组相同。分块控制的是临时展开量，不是“完全不增加显存”。

## 原三组正式完成后

自动发现通过核对的旧三组：

```bash
TOOL="$(cat /root/shared-nvme/BRIGHT-master/New/soft_memory_tool_path_fix1.txt)"
bash "$TOOL/run.sh" baseline-check
bash "$TOOL/start_server.sh" full
```

若旧结果放在自定义位置，可显式传入 `--baseline-root /实际完成目录`。默认正式输出是新目录，流程为 check 全部步骤 → 全量 `values` → 四组各 5000 步 → 最终 Val450 与完整性检查。源协议、旧结果、权重、原库、支持来源或包代码发生变化会停止，不恢复、不覆盖失败目录。

`full` 会重新运行自己的小库与 smoke，保持同一流水线的证据完整；不依赖此前 check 的临时状态。

## 常用单步命令

```text
run.py build --output NEW_VALUES --limit-images 4   # 小库，仅 Train
run.py build --output NEW_VALUES --limit-images 0   # 完整父成员的新 value
run.py smoke --values VALUES --allow-smoke-values --output NEW_SMOKE
run.py train --values FULL_VALUES --output NEW_TRAIN --baseline-root COMPLETED_OLD_RUN
run.py evaluate --values SAME_VALUES --checkpoint CHECKPOINT --output NEW_EVAL
run.py report --root RESULT_ROOT
```

smoke 权重再评估还需 `--allow-smoke-values`；评估的 key、TopK、chunk-size、seed 必须与 checkpoint 一致，formal checkpoint 会重新核对其绑定的旧三组证据。所有命令可加 `--repo`、`--v3-root`、`--protocol`。相对命令行请通过明确的 `"$PYTHON" "$TOOL/run.py"` 执行。

后台启动器接受 `SOFT_JOINT_WORKERS`、`SOFT_JOINT_CHUNK_SIZE`、`SOFT_JOINT_RESULT_BASE` 等配置。默认 `CUDA_LAUNCH_BLOCKING=1` 用于定位 CUDA 异步错误，不是旧 pooling 故障的修复；所有阶段保持相同该设置并记录。

## 报告怎么解释

- B-A：图内参考下，soft 相对这项空间规则的变化。
- D-C：历史参考下，soft 相对空间规则的变化。
- C-A、D-B：各自规则下历史标注案例的增量。
- D-C-B+A：交互量，正值支持超加性组合；组合最好不要求它为正。

报告包含按整个 split 累积的混淆矩阵、各类 IoU/P/R、逐事件指标、对 grid_probq 的救回与误伤、父候选/子槽覆盖。结构性空槽与背景标签不同。Case 来源加真值软标签的整体作用不等于单独证明检索排序；排序主张还需匹配训练的随机合法案例对照。

`report` 可读失败或部分结果，不调用 resume gate，也不会修改失败状态。正式成绩要求 5000 步、Train2918 覆盖、Val450、序列化末步权重、预测回读、原生概率与冻结状态全部通过。没有 Test 访问、历史旧库复现或准确率提升宣称。

## 验证与限制

CPU 测试包括：同权标签与 ignore、微小正质量、双槽守恒/回投、父表分块读取、梯度、四组真实优化循环、末步预测回读、同源库与字段完整性、跨事件选择、查询 GT 不影响输入、smoke 与正式证据区分、失败现场保留。完整发布结果在 `CPU_VALIDATION.json`。

原 R0/RT/RF 的 pooling 故障、SAR 的冻结参数故障不由本工具修复。若真实 check 在原冻结前端再次报错，查看新结果目录中的 traceback、pool_debug 和子阶段日志；不能把它解释为 soft/Memory 方法无效，更不能修改 complete 或放宽阈值让训练继续。
