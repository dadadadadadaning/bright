# SRAC Memory Rebuild V3：新建 FP32 配对库与检索验证

本实验对应第一阶段主线：固定同一组 Train 支持成员，在当前一次冻结前向中生成 T128、F256 和标签证据，比较 global Top8/Top64，并进行完整 Val450 诊断。没有 Reader 训练入口，也没有多原型、建筑加权、SAR 新检索特征或新 Q。

旧库只提供固定支持成员的原始行号、Train 图像 ID、事件 ID、region ID。T128、F256、软标签均重新生成并以 FP32 保存。建库逻辑不读取旧 token 或旧标签作为数值验收标准，不需要运行 Memory replay audit。

输入的已通过 E0 用于绑定冻结权重、原始支持身份、Train/Val 划分、原数据路径和运行设置。输出 `keys/protocol.json` 是本实验的新协议。新实验不能宣称复现了历史 FP16 库或历史 FP32 特征。

## 文件与数据流

| 文件 | 作用 |
| --- | --- |
| `paired.py` | 同次前向的实际 FP32 T128、Q 均值 F256；独立 CPU FP64 当前标签参考 |
| `core.py` | 两种 key 的共同有效成员、global cosine、Top64、软标签证据与诊断统计 |
| `shared.py` | 新协议、成员对应关系、数据划分、来源与逐文件校验 |
| `build_keys.py` | Train 支持成员重新建库，每个来源图像一次前向 |
| `diagnose.py` | Val 检索、标签投票、事后标签证据及像素评分 |
| `report.py` | 只读汇总报告 |
| `run_pipeline.py` / `run.sh` | 无临时结果环境变量的分阶段运行入口 |

```text
冻结 front 一次前向 ─┬─ 当前 T128 ──────────────┐
                    ├─ 当前 F256 + 当前 Q 均值 ─┤ 配对 FP32 bank
                    └─ 当前 Train GT + 当前 Q ──┘

冻结 Val front 一次前向 → 配对 query → global Top64 → Top8 前缀
                                    → 软标签证据 → 固定预测
                                                    ↓
                                          Val GT 事后诊断与评分
```

## 固定实验规则

- 正式建库保留全部原始支持身份和原始行顺序；smoke 只取前四个支持来源图像，并明确标记子集。
- T128 是 front 实际返回的 FP32 token；F256 是同次前向中原 A1 分类器实际接收的 FP32 feature，经同一 Q 路由求精确正质量均值。实际 FP16 输出直接拒绝，不通过上转换冒充 FP32。
- Train 标签从当前 GT 的 area fractions 与当前 Q 重新计算；忽略无效 GT，按正的有效标签质量归一化。独立 CPU FP64 路由累计参考验证这些新标签，固定 `atol=rtol=2e-5`。
- 支持行要求有效 region、正 Q 质量及正标签质量。零或极小 key 保留身份，但两种表示共同排除其检索资格；阈值均为两 key 的 L2 范数 `>1e-8`。
- 检索仅采用全局余弦，温度固定 `0.1`，排除同事件和同图像。每种 key 只排名一次 Top64，Top8 是其精确前缀；并列分数优先较小的新库行号。
- 新检索 CUDA 点积使用完整 FP32，局部关闭 TF32并在异常或正常返回时恢复原设置。front 继续使用 E0 绑定的运行设置。CPU 不修改这些开关。
- Val GT 不进入 key、邻居选择、软标签证据或预测；全部预测固定后才计算 Val 标签和诊断指标。

## 离线安装

把 `bright_memory_rebuild_v3.zip` 和对应的 `.sha256` 文件手动上传到：

```text
/root/shared-nvme/BRIGHT-master/New/
```

```bash
cd /root/shared-nvme/BRIGHT-master/New
if sha256sum -c bright_memory_rebuild_v3.zip.sha256; then
  /root/shared-nvme/conda-envs/bright4090/bin/python -c 'import zipfile; exec(compile(zipfile.ZipFile("/root/shared-nvme/BRIGHT-master/New/bright_memory_rebuild_v3.zip").read("INSTALL_MEMORY_REBUILD_V3.py"), "INSTALL_MEMORY_REBUILD_V3.py", "exec"))'
fi
```

安装器只新增：

```text
/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3/
```

安装器完全离线，不运行模型。已有不同文件或未完成标记会停止并保留现场；全部字节一致时允许只读重复安装。它不调用 `renameat2`，并处理共享文件系统不支持目录 `fsync` 的情况；普通文件同步及全量 SHA256 验证仍保留。旧 V1/V2 和 `New` 中的其他工具目录不会被覆盖。

## 安装后的运行顺序

首先运行 CPU 合成验证和命令计划：

```bash
bash /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3/run.sh preflight
bash /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3/run.sh plan
```

单独运行真实 smoke：

```bash
bash /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3/run.sh smoke
```

该入口会打印 `OUTPUT_ROOT=...`，先提取四个 Train 来源图像的新配对库，再诊断前两个 Val 图像并输出报告。smoke 的建库 `complete=true` 表示子集建库完成；smoke 的诊断只能有 `execution_complete=true`，`full_validation_evidence=false`。

smoke 通过后，使用终端打印的实际根目录继续正式实验：

```bash
bash /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3/run.sh full --root /实际打印的OUTPUT_ROOT
```

入口会验证已有 smoke 的源代码、E0、全部产物与日志 SHA，随后创建该根目录下的新 `keys/` 与 `diagnose/`，执行全支持成员建库及完整 Val450。部分失败或已有正式产物不能覆盖续跑。

也可直接让正式入口在新根目录中自动执行 smoke 和完整实验：

```bash
bash /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3/run.sh full
```

该命令的完整顺序是 `keys_smoke → diagnose_smoke → report_smoke → keys → diagnose → report`。任一步退出失败或证据门禁失败都会停止。它不依赖 `KEY_OUTPUT_ROOT`、`DIAG_OUTPUT` 或 `DEVICE`，不会自动选取“最新结果”。

GPU 正在运行其他工作时默认停止且不创建新结果根目录。希望排队等待可显式添加 `--wait-gpu-idle`，必要时添加 `--gpu-wait-timeout 3600`；不会终止其他作业。默认读取 E0 记录的 CUDA device，保留原来的冻结运行设置。

默认来源锚点是：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json
```

如需另一个实际已通过 E0，可在 `smoke/full` 添加 `--protocol /实际路径/protocol.json`。它必须满足原来源协议的要求。这里的输入应是来源 E0；本实验生成的 `keys/protocol.json` 用于新库验证和留档。

## 报告、产物与结论边界

默认结果位于 `BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_*`：

```text
pipeline_state.json         # 参数、E0/代码身份、阶段、退出码、日志与产物 SHA
logs/                       # 每阶段日志，包括可直接阅读的 report.log
keys_smoke/、keys/          # paired_bank.pt、protocol.json、manifest.json、summary.json、images.jsonl
diagnose_smoke/、diagnose/  # result.json、summary.json、images.jsonl、predictions/*.npz
```

每个新库行记录 `member_ids/original_rows/images/events/regions`，以及 FP32 `t128/f256/distribution/mass/label_mass`。每图 journal 绑定同次前向、Q/ID 与 key/label 指纹；读取时重新验证身份、实际 dtype、正质量、标签参考、划分和内容校验。

只读查看结果：

```bash
bash /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3/run.sh report --root /实际打印的OUTPUT_ROOT
```

报告包含：

- `a1_pixel`、`direct_region`、`grid_probq`，以及 T128/F256 的 Top8/Top64 未训练标签投票。
- 全 split confusion 汇总后的 foreground mIoU，即类别 1/2/3 的 IoU 均值；保留每类指标与事件统计。
- 邻居重合、支持覆盖、标签软兼容、dominant 类别覆盖、标签纯度、前景混合/背景稀释分区。
- 当前固定 Train 支持库的类别与事件库存，分别统计全部成员和共同可检索成员。

若 bank 中某类可用支持很少，先检查固定成员的案例覆盖；若库存充足而两个 key 的 Top64 标签覆盖都弱，应进一步检查特征或成员标签纯度；若 F256 与 T128 的邻居和标签证据有清楚差别，则有依据选择第二阶段检索表示。上述是诊断线索，不能仅凭单项指标证明原因。

本轮不量化全 Train 未选候选的筛选损失，不把区域票数解释为建筑数；未训练 label vote 也不代表 Reader 收益。完整 Val450 的新诊断通过后，再设计 `grid_probq＋可拒绝纠错` 与同容量 query-only 对照。

所有读取和写入都保留来源与划分限制。现有 CPU 测试验证算法及接口；真实 GPU、权重和 BRIGHT Val450 的效果须在服务器运行后确认。
