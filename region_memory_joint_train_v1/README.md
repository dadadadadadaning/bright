# BRIGHT 像素＋超像素＋Memory 联合训练工具包

2026-10-13。此包补齐之前只有model.py原型的训练入口：严格读取现有V3 FP32库、真实20步smoke、R0/RT/RF独立匹配训练、末步Val450、保存权重和只读报告。没有拒绝门控。云端只进行了CPU测试；真实BRIGHT与RTX4090的运行结果由服务器smoke及正式报告确认。

## 从下载到运行

把ZIP上传到服务器 `/root/shared-nvme/bright_region_memory_joint_train_v1.zip`。注意：电脑下载目录、云工作区、研究服务器是三个位置，必须上传到服务器后再执行。

```bash
(
  set -euo pipefail
  PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
  ZIP=/root/shared-nvme/bright_region_memory_joint_train_v1.zip
  test -s "$ZIP" || { echo "服务器没有找到ZIP：$ZIP"; exit 1; }
  mkdir -p /root/shared-nvme/BRIGHT-master/New
  UNPACK="$(mktemp -d /root/shared-nvme/BRIGHT-master/New/joint_train_XXXXXXXX)"
  "$PYTHON" -m zipfile -e "$ZIP" "$UNPACK"
  TOOL="$UNPACK/region_memory_joint_train_v1"
  test -s "$TOOL/run.py"
  bash "$TOOL/start_server.sh"
)
```

启动脚本打印后台PID、实际工具路径、结果目录和日志路径。复制打印的tail命令看日志，Ctrl+C退出tail不会停止nohup任务。不要在每次查看日志时重复启动。所有输出进入新的 `SRAC_REGION_MEMORY_JOINT_V1/run_...` 目录；不改旧权重、库、协议和失败现场。

本包不向原experiments目录安装任何文件，也不联网下载、不要求改仓库公开性，因此没有此前原子安装rename失败或GitHub连接超时的安装步骤。

默认配置：

```
Python: /root/shared-nvme/conda-envs/bright4090/bin/python
仓库: /root/shared-nvme/BRIGHT-master
V3结果: /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665
读取库: 上述目录/keys/paired_bank.pt
GPU: cuda:0（须与源E0协议绑定一致）
```

如果路径改变，启动前在同一终端设置 `JOINT_REPO`、`JOINT_V3_ROOT`。可以设置 `JOINT_WORKERS=0` 减少DataLoader资源、`JOINT_CHUNK_SIZE=32`减小Reader临时编码块。不要修改原协议来改变运行时；分块不保证总反传显存按比例下降。第一轮Top64；Top8用单独新目录，完整重新训练。

## 自动执行什么

1. 验证本工具文件SHA256、Python/CUDA、已有V3源码接口和正式库文件；执行CPU合约。
2. 严格调用已安装V3的load_bundle/load_paired_bank，核对源E0、FP32库、成员对应、Train标签参考、文件hash、划分和运行时。不会跳过V3原bank门槛。
3. 从4个不同事件选4张Train，从不同事件选2张Val，均要求存在合法跨事件支持。每组20次更新，验证Reader梯度、同次P0初始化、有限损失、来源冻结、保存重载和预测计数；保存显存峰值。
4. smoke通过后，重新初始化三组头，开始完整Train2918，每组5000次优化更新。20步smoke权重不用于正式训练。
5. 正式0步完整Val450，每500步完整中间Val，5000步保存last并重载，完成末步Val450，重开保存预测累计混淆矩阵并核对冻结来源。

5000是优化更新次数，不是5000轮。前端每个查询只算一次，三组头依次反传，尽量减少同时驻留的训练激活。完整验证需时间，日志会显示`Joint Val images processed ... final checks pending`；450/450后仍有产物核验，不能只看进度宣布成功。

## 三组的精确定义

共享像素F256、Q均值F256、T128和同次grid_probq；三组同模型、参数数、初始权重、图序、loss、lr和更新预算：

| 组 | Reader支持来源 |
|---|---|
| r0 | 当前图像Q质心空间近邻，**包括自身**，用当前A1预测概率的Q均值作软标签；不读查询真值 |
| rt | 原T128余弦检索合法Train Top64；读取该行配对F256/T128与Train软标签 |
| rf | 原F256余弦检索合法Train Top64；读取相同格式的配对F256/T128与Train软标签 |

两个Memory组只改候选ID。r0/Memory对照比较的是“图内预测参考”与“跨图训练标注案例”两种上下文来源；不能把全部差异单独归因于排序器。相同bank key保持冻结，新Reader内部编码器可训练，避免静态库与变化key不一致。

原A1/Q/T生成器/原检索key/库冻结；仅训练新Encoder、Reader和非线性联合像素头。log(grid_probq)+新头输出用于初始化，新头末层为零；没有alpha、拒绝阈值、困难像素限定或强制修改幅度上界。

初始新概率按预定atol=rtol=2e-5对同次P0检查；log/softmax重表达可能导致极近tie处argmax舍入变化，会记录，不把它冒充旧原生分类器重放。原生A1同次门槛仍调用原native helper。首次更新仅末层收到梯度是零初始化的正常结果，随后检查Reader确有梯度。

沿用旧训练预算：AdamW lr1e-4、weight_decay5e-3、500步warmup+cosine、梯度裁剪1；旧region_nll+Lovasz(classes=present,ignore=255)；FP32、batch1、1024完整图无随机增强。源legacy/TF32配置保持协议值，不声称跨机器bitwise确定。

## 查看与解释结果

```bash
"/root/shared-nvme/conda-envs/bright4090/bin/python" \
  /实际工具目录/region_memory_joint_train_v1/run.py report \
  --root /实际结果目录/run_...
```

`report`只读取，允许查看失败和部分记录，不调用旧“失败阶段不能resume”的report门槛。

```
结果根目录/
  workflow_state.json         整体队列状态
  preflight.json
  logs/smoke.log
  logs/train.log
  smoke/summary.json          仅小样本执行检查，不是Val450效果
  train/manifest.json         固定方案、源协议、库和源码绑定
  train/train.jsonl           三组逐步loss、grad、支持覆盖、显存峰值
  train/summary.json          更新数与最终验证状态
  train/checkpoints/checkpoint_last.pt
  train/checkpoints/checkpoint_last.pt.json  权重校验记录
  train/initial_val/result.json
  train/final_val/result.json
  train/final_val/predictions/*.npz
```

前三个差值：r0−grid_probq=新图内联合头变化；rt−r0或rf−r0=历史案例相对同容量图内参考的变化；rf−rt=检索key差异。主指标是完整Val450累计confusion的前景三类mIoU，不是四类Test。末步同时报告每类IoU/P/R、逐事件和基线正确→错误/错误→正确统计。候选出现后再重复种子，不按不同最佳中间步挑对照。

没有Test入口，没有改训练图数/源bank，也没有宣称旧diag流水线成功：旧诊断状态单独观察。只要正式新bank与源协议严格有效，可以启动独立联合训练；本包不修改旧完成标记。

## 失败处理与独立评估

任何预检/smoke/训练错误会停后续阶段，记录traceback与当前summary，保留已保存权重。先读当前日志末尾，不自动降低数值门槛、重建库、覆盖目录或启动重复任务。`training_complete=true`与完整Val450通过是两个状态，训练完但最终核验失败时权重仍保留。

不提供精确resume；checkpoint有optimizer/RNG供取证但不承诺恢复DataLoader位置。可用同版本工具独立重评一个已保存checkpoint：

```bash
"/root/shared-nvme/conda-envs/bright4090/bin/python" /实际工具目录/run.py evaluate \
  --checkpoint /实际结果目录/train/checkpoints/checkpoint_last.pt \
  --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_MEMORY_JOINT_V1/reval_新的名字
```

种子/TopK/chunk若改过，评估也传相同值。smoke权重只评smoke，不能冒充正式5000步结果。中间权重完整Val结果会标明step，不能声称末步正式实验完成。

如显存不足，保留失败目录；先看进程/实际峰值，调整分块或worker后开新目录。工具不自动结束其他任务，也不保证4090在任何并行负载下都能容纳训练。训练耗时以真实smoke与阶段日志为准。

## 本地测试边界

CPU测试包括联合模型、接线目标无泄漏、R0质心/候选mask、真实接口stub、三头优化/保存重载/离线混淆复算、失败停止与源码/权重绑定。CPU合成driver使用FakeRuntime，不代表跑过真实Train/Val或CUDA。最终测试数量与结果见RELEASE_CHECKS.json。

可单独运行：`python -m unittest discover -s /实际工具目录 -p 'test_*.py' -v`。本包依赖服务器已有可运行的V3及BRIGHT环境，不包含数据、权重或旧项目源码副本。
