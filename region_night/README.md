# BRIGHT 区域读出：P/R/T 与区域辅助修正的夜间队列

这份代码是可训练、可审核的候选实现，尚无服务器真实成绩。默认串行执行：

1. 核对已完成的 J1/J2/J3 来源、代码指纹及 CPU 合约。
2. 等待目标 GPU 上现有计算进程结束（显式使用 `--wait-gpu-idle`）。
3. 从原 J 实验的权重/数据/缓存生成一份新的完整 E0：legacy/native/all、Val450、原五路径 1e-5 复现门。
4. 原 P/R/T 分类头各 5000 次更新、初始和末步 Val450、独立 Val450、报告。
5. 本包 local_delta 与 region_delta 各 5000 次更新、初始和末步 Val450、独立 Val450、报告。

旧 `protocol.json` 不存在时，第三步生成的是新复现协议。它不是恢复原文件或原 SHA；不能拿新协议直接冒用历史 Q 实验的 checkpoint 身份。新 E0 若复现失败，队列保存失败并停止，不进入训练。

## 训练在比较什么

| 组 | 特征和输出 | 判读 |
| --- | --- | --- |
| P | 固定原 F256，训练 A1 初始化的像素分类头 | 像素头适配对照 |
| R | 固定原 Q 对 F256 精确均值，训练同初始化/同容量的分类头 | 与 P 比较区域均值路径 |
| T | 固定原复杂 T128，训练原区域头的独立副本 | 与原 region 比较；T 不是 P/R 等容量的第三臂 |
| local_delta | 同 grid_probq 基础概率，修正头输入 F256+F256 | 相同架构的本地修正控制 |
| region_delta | 同基础概率，修正头输入 F256+按 Q 汇总回填的 F256 | 测试额外区域均值上下文 |

后两臂都是 512→64→4 的 1×1 卷积/GELU 网络，各 33,092 个参数，同初始化、共享 Train 图像顺序。末层零初始化，修正为 `2*tanh(raw/2)`，概率为 `base_prob*exp(delta)` 再按四类归一化；先在同一特征网格修正，再上采样概率。

初始化在 FP32 容差内复现 grid_probq；它不是位级绝对相同的保证。两个输入空间的相关性/信息秩不同，因此只称同架构、同参数数、同预算，不能称有效自由度严格相同。local_delta 的基础概率本身也使用 Q；它只是没有额外的区域均值特征输入，并非完全无 Q 的模型。

这是新的像素/区域辅助读出，而非修复原 T128 的已验证方案。本轮不训练 Q、A1、原 embedding、旧 J 头或 Memory。新输出臂没有读取 Memory；历史 J 路径只用于复现与参照，不能把新候选涨分称为 Memory 收益。

## 获取代码

代码放在 GitHub 分支 `codex/region-night-20261012` 的 `region_night/`，独立于旧实验目录。可在浏览器登录有访问权限的 GitHub 账号，查看该目录；不要将新分支覆盖到正在使用的 BRIGHT-master。

推荐使用随发布说明提供的精确 commit 下载命令：先下载 `install_night.py`，再带 `--ref` 精确提交执行。安装器只下载这里的十一个代码/说明文件，逐个检查 SHA256；拒绝覆盖不同的已存在目录。它不修改原工程，不运行训练，不下载权重或数据。

也可将整个仓库新分支克隆到另一个新目录，只使用其中 `region_night/`；`--repo` 仍指向训练原权重时的 BRIGHT-master，保留原源码指纹。

## 服务器启动

默认路径已经根据本次对话填写：

- 原工程 `/root/shared-nvme/BRIGHT-master`。
- Python `/root/shared-nvme/conda-envs/bright4090/bin/python`。
- J 来源 `/root/shared-nvme/BRIGHT-checkpoints/SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint`，不使用 20 步 probe。
- 工具目录 `/root/shared-nvme/BRIGHT-region-night-tools_20261012`。
- 新结果父目录 `/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1`。

先看计划，不使用 GPU、不读取 Train/Val 图像：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-region-night-tools_20261012/night_run.py --plan-only
```

终端出现正常 shell 的 `#` 提示符后，逐条执行；这些命令没有 `read` 输入步骤：

```bash
export NIGHT_TOOLS=/root/shared-nvme/BRIGHT-region-night-tools_20261012
export NIGHT_LOG=/root/shared-nvme/bright_region_night_$(date +%Y%m%d_%H%M%S).log
nohup /root/shared-nvme/conda-envs/bright4090/bin/python -u "$NIGHT_TOOLS/night_run.py" --wait-gpu-idle > "$NIGHT_LOG" 2>&1 &
printf 'PID=%s\n日志=%s\n' "$!" "$NIGHT_LOG"
tail -n 60 -f "$NIGHT_LOG"
```

退出查看日志时按 Ctrl+C；停止的是 tail，后台队列继续。不要重复启动相同队列。脚本使用文件锁防止同结果父目录下的重复队列；它只能防止同一 launcher 的重复运行，不能阻止其他独立实验后来占用 GPU。

`--wait-gpu-idle` 会每 30 秒检查 GPU 计算进程。如果 Memory Val450 仍在跑，日志显示 Waiting，等结束后再开始。本脚本不会终止其他实验。无超时默认持续等待；可使用 `--gpu-wait-timeout 秒数`。输出脚本不会主动启用 GPU，正式运算在子进程进行；用户不要在队列启动后另外开同 GPU 训练。

只跑 P/R/T：给 night_run 加 `--prt-only`。已有准确且通过的 E0 文件：加 `--protocol /实际文件/protocol.json`；必须与指定 J 来源一致。默认 fresh E0 从 J manifest 读取原 A1、front、cache、metric、数据列表、seed/device/workers，文件搬迁或来源不完整会停止并报告。

## 输出与结果判断

日志会打印 `OUTPUT_ROOT=.../run_时间_UUID`。该根目录包含：

```text
workflow_state.json
logs/各阶段.log
fresh_e0/protocol.json          # 没有显式原协议时生成
prt_fit/summary.json
prt_independent_val/result.json
residual_fit/summary.json
residual_independent_val/result.json
```

每个新 fit 包括训练 journal、manifest、末步权重、initial_val/final_val。没有任何自动失败续训、跳图、降低容差、挑 best checkpoint 或读 Test 的步骤。

完整成绩只使用通过全部检查的 Val450。固定前景 mIoU=(Intact+Damaged+Destroyed IoU)/3，背景不计入平均。两图 smoke、训练 step 输出、失败验证不能当正式结果。

重点比较：

- T−原 direct_region：原复杂摘要的分类头是否可以适配。
- R−P，以及 R−region_mean 与 P−pixel_grid：区域均值价值与普遍的分类头训练效应。
- region_delta−local_delta：区域均值上下文相对本地修正控制的增量。
- region_delta−grid_probq：候选整体是否超过未修改基础路径。
- Damaged 的 IoU、Precision/Recall、GT×旧预测×新预测救回/伤害/FP，以及事件差异。

两新臂的结果存有原五路径、native A1 回放、Q合法性、源码/权重/运行时/训练来源哈希、逐图预测和 journal。全 450 图与历史五路径及新 E0 参考复现均通过才标 full_validation_evidence=true。若最终 Val 失败，已完成训练权重保留 complete=false，不会作为正式 fit 自动继续后续阶段。

运行时间取决于整图计算与多次 Val450；不承诺一晚跑完或一定涨分。明早先看 workflow_state 的 complete/current_stage，失败时发该文件及对应日志末尾；成功时发两个 independent_val/result.json 和两个 fit/summary.json。

## CPU 验证

```bash
PYTHONPATH=/root/shared-nvme/BRIGHT-master /root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-region-night-tools_20261012/region_residual.py preflight
```

合约覆盖零初始化、冻结输入、有效梯度、有界概率、ignore255、双臂失败不部分更新、合成 smoke 训练/保存、同次 evaluation、正式门、来源与文件哈希、队列顺序、锁、GPU等待、日志和停止逻辑。CPU通过不代表真实 BRIGHT/CUDA 实验通过。
