# Memory联合训练：独立索引布局候选修复与1500步后继续训练

针对指定失败运行：三组r0/rt/rf各更新1500次，1500步checkpoint已保存，随后Val450在`la_palma-volcano_00000586`的512通道区域统计汇聚中触发CUDA索引断言。保存的CPU输入索引0…1023合法、内容有限。候选只克隆整数索引并将expand的广播索引物化为连续张量；数学路由、权重、重复贡献及mass.clamp_min(1e-6)保持原定义。

**这是待服务器GPU验证的候选修复，尚未证明CUDA故障的唯一根因，也不保证长时运行永不出错。** 原生与候选重放在不同新进程里执行；不在CUDA已出错的进程里重试。

## 固定输入

- 原训练工具：`/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1`
- 失败根目录：`/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_MEMORY_JOINT_SYNC_V1/run_20261013_073724_212223_3841256`
- 权重：上面目录的`train/checkpoints/checkpoint_001500.pt`
- 输入快照：`train/pool_debug/input_009005.pth`
- 先等待SAR四组：`/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_Q_MATCHED_V1/run_20261013_082551_3863093`
- 再等待SAR＋soft：`/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_SOFT_CROSS_V1/run_20261013_092358_3887540`

需要原文件、SHA256 sidecar、完整1500步train.jsonl与原训练包。新工具读取并校验它们，不覆盖、不重新标记旧运行，不重建库或改原源码。修复仅在自己的Python进程内安装作用域受限的汇聚包装器，退出/失败后恢复引用。原协议的数值配置保留；`CUDA_LAUNCH_BLOCKING=1`作为明确记录的同步诊断设置。

## 从ZIP到启动

手动把ZIP上传到：

```
/root/shared-nvme/BRIGHT-master/New/bright_joint_pool_recovery_v1.zip
```

按发布消息中的SHA256核对后，解压到New里的新目录。一次启动即可：

```bash
cd /root/shared-nvme/BRIGHT-master/New
PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
UNPACK="$(mktemp -d /root/shared-nvme/BRIGHT-master/New/memory_recovery_XXXXXXXX)"
"$PYTHON" -m zipfile -e bright_joint_pool_recovery_v1.zip "$UNPACK"
bash "$UNPACK/joint_pool_recovery_v1/start_recovery.sh"
```

启动器打印PID、工具路径和launcher日志。日志里的`OUTPUT_ROOT=`为实际新结果根目录。重复启动会被恢复工具的文件锁拒绝；该锁仅防止重复Memory恢复入口，不是整台GPU的预约锁。

## 自动顺序与门槛

1. 校验原训练包、旧模型/协议/库绑定、checkpoint sidecar、输入快照及完整训练日志；CPU重放一次。CPU阶段不创建CUDA上下文。
2. 等上述**两项**SAR任务都明确结束，再连续两次观察目标GPU没有计算PID且利用率≤5%。不会终止、重启或改写SAR任务。静默中断但没写完成/失败状态的SAR任务仍会阻塞，需检查其进程和状态，不能伪造完成标记。
3. 分别新进程重放原生和contiguous候选，每组3次；保留原记录的核心数值配置与补充运行环境/版本/GPU检查。原生重放失败作为诊断保存；候选必须全部通过固定atol=rtol=2e-5的CPU参考和重复性检查。单次通过不等于长时稳定。
4. 校验三组权重、AdamW、RNG、完整配置和数据文件哈希；用候选独立完成1500步checkpoint的Val450并重新读取保存预测。此结果标记为中间checkpoint，不是5000步完成。
5. 在CPU重建训练数据迭代器，逐条核对原1500步图像与epoch顺序；恢复三组优化器和随机状态，使用原5000步学习率计划，从1501继续到5000。重建进度每250条打印一次。
6. 沿用原中间验证间隔；各新checkpoint带权重哈希和恢复状态。末步重载保存权重，完整Val450、重读预测、核对冻结模型/库/数据/源码及作用域恢复。

每种新输入布局首次出现时，包装器还必须独立对照原CPU汇聚参考。任一必要门槛失败即停止，保留新的失败证据与已保存checkpoint。包不放宽容差、不clamp区域ID、不跳过失败图片或验证。

## 结果应该怎样表述

新运行是**原生执行的1500步匹配前缀＋候选修复后的3500步继续训练**。三组最终总预算相同，都是5000次更新，但不能宣称精确复现原始不中断运行。`exact_historical_resume_claim=false`、`historical_reproduction_claim=false`。

旧原包没有resume接口；本工具实现了新的有校验继续训练引擎。CPU合成对照检验了恢复后的权重、优化器、RNG与数据顺序；真实BRIGHT和RTX4090结果必须由本次服务器记录证明。

原三组定义与模型保持原工具设计：r0为当前图像预测上下文，rt/rf为历史Train标签上下文。**该联合训练包没有显式拒绝门控**；此次处理不增加门控，也不把它改称可拒绝Reader。

## 查看状态

```bash
"$PYTHON" /实际解压目录/joint_pool_recovery_v1/recovery.py \
  --stage report --root /日志中的OUTPUT_ROOT
```

状态含义：

- `waiting_declared_sar`：正常等两项SAR任务；不是卡住。
- `waiting_gpu`：等待GPU空闲。
- `replay_contiguous`：检查候选；通过前不更新模型。
- `checkpoint_revalidation`：子训练报告中正在重新评价1500步权重。
- `reconstructing_training_iterator`：CPU核对历史顺序。
- `training_continuation`：新更新已经开始，报告分开显示prefix/new/total。
- `complete`与`full_validation_evidence=true`：新修复后继续训练的全部证据通过；不改变旧失败状态。

原生重放的`diagnostic_only=true`表示对照结果；它失败但候选通过时允许继续。真正失败看顶层`failure`及相应阶段日志。

只读检查或仅CPU重放：

```bash
"$PYTHON" /实际解压目录/joint_pool_recovery_v1/recovery.py --stage inspect
"$PYTHON" /实际解压目录/joint_pool_recovery_v1/recovery.py --stage cpu
```

CPU单位测试无需GPU：在BRIGHT仓库中运行，并保留完整原训练工具：

```bash
cd /root/shared-nvme/BRIGHT-master
JOINT_ORIGINAL_TOOL=/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1 \
PYTHONPATH=/root/shared-nvme/BRIGHT-master \
"$PYTHON" -m unittest discover -s /实际解压目录/joint_pool_recovery_v1 -p 'test_*.py' -q
```

测试细节与范围见`VALIDATION.md`。任何CPU通过结果都不代表服务器GPU已修复。
