# BRIGHT：单槽／双槽与 Memory 匹配消融

这是已有 **Extended 冻结前端和正式案例库之上的训练扩展**。它复用服务器实际运行版本，不复制另一套 DamageFormer、Q、Reader 或数据加载代码。

本包包含新的训练入口、支持干预、内容诊断和 CPU 测试。**未在本地执行真实 BRIGHT/CUDA 训练，不包含权重、数据或已有实验成绩。** `full` 在服务器先做 CPU 检查、再做独立 CUDA smoke；失败立即停止。

## 1. 本轮回答什么

| 组别 | 区域内容 | 参考来源 | 支持标签 |
|---|---|---|---|
| E | 单槽 | 图内近邻 | 同次前向的预测分布 |
| F | 单槽 | Top64 Train 父案例 | 对应 Train 软标签 |
| A | 固定空间双槽 | 图内近邻 | 同次前向的预测分布 |
| B | 建筑概率 soft 双槽 | 图内近邻 | 同次前向的预测分布 |
| C | 固定空间双槽 | Top64 Train 父案例 | 对应 Train 软标签 |
| D | 建筑概率 soft 双槽 | Top64 Train 父案例 | 对应 Train 软标签 |
| R | 建筑概率 soft 双槽 | 随机合法 Train 父案例 | 对应 Train 软标签 |
| L | 建筑概率 soft 双槽 | 与 D 相同的父案例 | 打乱对应关系的 Train 软标签 |
| N | 建筑概率 soft 双槽 | 与 D 相同的父案例 | 标签嵌入贡献置零 |

所有组保留像素通路、父区域表示、Reader、`grid_probq` 和原四类预测头。图内参考并不等于“没有 Reader”；单槽也不是在两个槽分别编码后简单相加。

主要比较：

- **B−E、D−F**：soft 双槽相对单槽的增量。
- **B−A、D−C**：控制双槽数量后，建筑概率组织相对空间组织的增量。
- **F−E、D−B、C−A**：对应组织条件下，历史支持相对图内支持的增量。这同时改变支持来源和标签来源，不能单独归因于检索排序。
- **D−R**：基于特征选择的支持，是否优于相同预算的随机合法支持。
- **D−L**：特征和标签的正确对应是否有用。
- **D−N**：显式支持标签嵌入是否有用。N 仍使用同一个有监督筛选的库，不能称为完全无标签方法。
- **D−grid_probq**：整套可训练后端是否超过冻结概率共享基线。
- **(D−B)−(F−E)**：soft 双槽与 Memory 的交互项；若有 A/C，还报告 **(D−C)−(B−A)**。D 最好不自动证明正交互。

结果必须结合类别 IoU、Damaged 精确率/召回率、逐事件指标和配对纠错/误改分析。单次微小正差不是稳定收益证明；不能预先承诺涨多少。

## 2. 固定的训练协议

- 从共同的随机初始化开始，各组完整联合模块参数完全相同，优化器存储独立。
- 训练槽编码、Reader 和联合分类器；冻结 A1/Q/像素前端、父 key 和案例库。
- **没有共同训练过的 D500 前缀，也不恢复旧 D 权重。** 各组从第 1 步开始使用自己的输入条件。
- 同一训练样本每步仅提取一次冻结前端特征，各组顺序更新；训练图像次序与随机状态匹配。
- 数据加载器遍历完后按原训练规则使用 `seed+epoch` 开启下一轮，日志记录 epoch 和对应 seed。
- 原 NLL、原 Lovász、全部有效像素和背景监督均保留；**KL=0**。
- AdamW，最大 LR=`1e-4`，weight decay=`0.005`，梯度裁剪=`1.0`。
- 沿用 5000 步调度：前 500 步 warmup；第 501 步起在原调度上乘 `0.2`。终点默认 2500，**不将调度长度改成 2500**。
- 初始及每 500 步完整 Val450，2500 为预定主要终点。初始、末步还对同一固定 Train450 做推理模式诊断。
- 固定 Train450 从完整 Train 名单按 `seed+191` 随机抽样后排序；只用于诊断，并非训练加载器，也不声称等于旧诊断子集。
- 不读取 Test；不按中间最佳步替代预定终点。
- 一个种子是一轮消融诊断。方案确定后做至少三个配对种子，按同一终点报告均值与离散程度。

这轮可以和同轮冻结基线、同轮其他组比较；不能将跨代码、跨案例成员、跨训练起点的旧成绩直接相减当成纯结构收益。

## 3. 干预的实现

### 单槽

复用原两槽接口，但只保留一个真实槽，另一槽为严格无质量、无路由、无注意力参与的填充槽。特征按原槽的特征质量汇总，Train 标签按有效标签质量汇总；忽略标签像素不进入标签分母。单槽回投为原父 Q。

所有组使用同一模型类，**总参数数量相同**；有效内容槽数量不同正是此项消融。角色嵌入的有效使用量会不同，因此不是声称每个参数在所有组中梯度都相同。不得把两个独立编码/读取后的槽相加冒充单槽。

### 随机支持 R

从合法、可用 Train 父案例均匀无放回抽样，排除同图、同事件。每个查询的父案例数量匹配 D 实际非填充数量；D 无支持的查询仍无支持。使用独立 SHA256 派生随机种子，不改变训练全局 RNG。

### 错配标签 L

在**相同父案例事件、相同两槽有效性模式**内，对父案例的成对软标签做固定错排；父 key、支持特征、案例 ID、槽角色、槽质量不变。这样标签来源仍满足查询的跨事件排除。单元素分层无法改变，重复标签也可能使干预较弱；报告实际发生标签变化的比例，不能仅凭“运行了 permutation”宣称完全打乱。

L 是匹配训练对照，不是仅在推理时临时打乱。

### 无显式标签 N

保持同一标签嵌入层和参数，通过仅限本次前向的 hook 将标签嵌入输出置零；`finally` 清除 hook。原模型输入校验仍接收合法归一化分布。没有修改原模型文件。

## 4. 自动内容诊断

每次评估在同次冻结前向之后统计，查询真值仅供离线分析，不进入支持构造或检索。

- 两槽原始特征 cosine、归一化 L2、两槽 GT 软标签 JS；
- 槽质量、质量比例、有效像素量 `(sum w)^2 / sum(w^2)`，同一像素重复 Q 路由先合并；
- 建筑概率的均值、方差与熵；
- 混合区域、内部纯区域、类别边界、低建筑概率位置、Damaged 查询分层；
- D/R 支持标签组成、不同父案例/图像/事件数及有效数量；
- 完好困难负例代理：完好占优且冻结父 key 相似度高；它是诊断代理，不等于已证明的错误案例；
- 每张图最多 32 个区域案例的数值记录。

分层阈值与分母写在诊断输出中。内部纯区域定义为语义纯度代理，不是建筑实例内部；边界以有效 GT 类别边界定义，不是建筑实例边界。D/R 的支持组成加权不是 Reader 学到的注意力。

**冻结原始槽统计在相同图像上本应基本不随训练变化。** 这些统计解释可用内容，不能单独证明训练后的 Reader 利用了内容。互补内容的使用是否有效，还要结合各训练组差值和配对预测变化。包内不自动把“槽相似”判断为故障，也不强制槽正交。

## 5. 依赖与版本

默认来源实验：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_ANCHOR_EXTENDED_V2/run_20261019_053508
```

入口读取其 `train/manifest.json` 的 `arguments.tool_dir` 和 `runtime_config`，定位实际 Extended 包、Q 来源、正式父库和双槽库。需要这些原文件仍在。

`plan` 只做文件/文本/哈希检查，不加载权重。真正的权重、数据、成员、冻结状态、检索与 CUDA 检查由原 Runtime 在 smoke 和正式阶段执行。

- 不要求缺失的历史成员库，不重建已有正式库。
- 不绕过原 Runtime 校验，不伪造完成报告，不修改原包/manifest/source。
- 外层包与 study 子包哈希属于不同范围，分别记录，不错误地要求相等。
- 未上传的服务器实际 Extended 代码由入口现场校验；本包不能在本地保证其所有接口与上传的参考版本完全一致。
- 本包不提供静默续跑。若失败，保存现场与检查点，先分析 traceback；重新 `full` 会产生新实验目录，不能称为接着上次更新。

## 6. 运行

先解压下载的发布 ZIP，再执行以下单行命令；不要把终端提示符复制进命令。

只看启动计划：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/slot_memory_ablation_v1/ablation_runner.py plan --group main
```

推荐六组：E/F/B/D/R/L，回答单槽、Memory、随机检索和标签对应问题：

```bash
bash /root/shared-nvme/BRIGHT-master/New/slot_memory_ablation_v1/start_server.sh full --group main --seeds 3407 --steps 2500
```

如果要把空间双槽和无标签对照一次补齐，用下行**替换**上一条，而非同时运行：

```bash
bash /root/shared-nvme/BRIGHT-master/New/slot_memory_ablation_v1/start_server.sh full --group all --seeds 3407 --steps 2500
```

组集：

| 参数 | 训练组 |
|---|---|
| `--group core`（默认） | E/F/B/D |
| `--group main` | E/F/B/D/R/L |
| `--group spatial` | A/B/C/D |
| `--group memory` | D/R/L/N |
| `--group all` | E/F/A/B/C/D/R/L/N |
| `--arms e,b` | 明确指定任意不重复子集 |

`full` 已包括真实 CUDA smoke；不需要先单独启动一轮正式程序再启动另一轮。只检查可使用 `smoke` 替换 `full`。smoke 复用正式库，但训练预算和报告明确为 smoke。

后续配对种子可以用 `--seeds 3407,3408,3409`，种子逐个顺序运行。不要未经比较更换所有组的学习率、损失和结构。

检查学习率时可单独使用 `--arms d --post500-lr 1` 或 `.1`；它们均仅改变第 500 步之后的倍率。必须与相同库、初始化、完整数据次序和预算的 D 对照比较。

来源路径不同可追加 `--source-run /真实结果目录`；不支持用别的包代替 manifest 中记录的实际运行包。

## 7. 看日志和结果

读取最新总报告：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/slot_memory_ablation_v1/ablation_runner.py report
```

查看最新启动/流水线日志：

```bash
tail -n 80 -f "$(cat /root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_V1/latest_launcher_log.txt)"
```

流水线日志的 `START 3407_train log ...` 给出训练日志。各阶段使用独立子进程；训练进度在对应阶段日志里。启动器 PID、`START`、Val450 已处理完均不单独代表最终完成。

结果布局：

```text
SRAC_SLOT_MEMORY_ABLATION_V1/
  latest.txt
  run_.../
    plan.json
    workflow_state.json
    logs/
    seed_3407/
      smoke/summary.json
      train/
        manifest.json
        summary.json
        train.jsonl
        support_provenance.json
        input_fingerprints.json
        checkpoints/checkpoint_000500.pt ...
        initial_val/result.json
        val_step_000500/result.json ...
        fixed_train_initial/result.json
        fixed_train_endpoint/result.json
```

每次评估目录还有 `images.jsonl`。逐图机制统计位于其 `mechanism_audit` 字段，流式重读后的汇总位于 `result.json` 的同名字段。结果包含总体与逐事件混淆、类别精确率/召回率、共同原损失、相同像素配对纠错/误改、各消融差值与交互项。

最终仅在冻结文件、模型、源码、逐图记录和 Runtime 退出检查通过后，才设置 `complete=true`、`full_validation_evidence=true` 和正式终点标记。不要手动修改这些字段。

## 8. 验证范围与后续

本地测试覆盖真实原模型的九组前向、反向、更新，支持/标签干预、源版本绑定、评估模式/RNG恢复、错误路径和重新读取日志的一致性。实际 CUDA/数据数值与性能仍须服务器 smoke/完整 Val 验证。

训练后先提供：

1. `report` 输出；
2. 同一 seed 的末步 `result.json` 与 `support_provenance.json`；
3. 末步内容诊断汇总与逐事件差值；
4. 有失败时对应阶段日志末尾和 `traceback.txt`。

不需要上传数据集或模型权重来做第一轮结果分析。当前包没有自动生成论文图片，没有实现槽交换/屏蔽推理干预，也没有宣称新结构或检索已经有效。
