# SAR A1 冻结状态取证

这是定位工具，还不是已经在4090上验证的修复。保留原训练器、原SHA校验和原实验结果；不修改冻结标志，不把旧`last.pth`认证为成功，不访问Test。

## 已有现场结论

SAR Q baseline完成2000步/32000裁块呈现及450张内部Val，打印region=61.5050%、A1=61.6867%。随后严格冻结A1检查失败。最后权重文件存在，约18 MB，保存的是Q等adapter和优化器，并没有故障时完整A1快照。其他三组只有smoke通过，尚未正式训练。SAR＋soft的stages为空，是上游失败后停止等待，没有训练。

原训练器只在初始Val后及全部训练后比较整体SHA。因而现存失败记录无法追溯变化的张量和发生时刻。不能据此判定BatchNorm、AMP、显卡或检验误报，也不能直接把旧权重设为`run_complete=true`。

## 使用

将单个文件`SAR_FREEZE_WITNESS_V1.py`放到服务器`/root/shared-nvme/BRIGHT-master/New/`。每条命令都是一行，不使用heredoc。

只读检查旧结果、checkpoint元数据和GPU/相关进程，不创建CUDA上下文或训练：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/SAR_FREEZE_WITNESS_V1.py inspect
```

短程定位：从相同E0新建baseline，使用原BF16训练数学、batch2×accum8，仅20步及2张smoke验证；使用原GPU队列等待两次空闲。此命令在前台运行，GPU忙时会继续等待。不要重复启动；停止本工具用Ctrl+C，工具不停止其他任务。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -u /root/shared-nvme/BRIGHT-master/New/SAR_FREEZE_WITNESS_V1.py probe
```

它会打印新的`OUTPUT_ROOT`，证据放在这个目录下：

- `run.log`：完整子程序输出。
- `witness/summary.json`：本次取证是否完成。
- `witness/*_initial_tensors.json`：初始每个参数和buffer的名称、形状及哈希。
- `witness/failure.json`：发生变化的分组和具体张量、字节差异、最大数值差、检测步骤及上一个通过的步骤。
- `witness/frozen_change.pth`：变化张量的CPU前后快照；只有失败时生成。

检查第1步、每20步和验证前后。报告的step是**发现变化的检查点**，不宣称精确的首次写入时间。每次前向还检查冻结模块的模式和requires_grad。

`probe`通过只能证明本次20步通过，不能证明原2000步失败已修复。它不会自动启动正式四组，也不会运行SAR＋soft。若再次失败，请读取新目录里的failure.json；若没有此文件，查看run.log或witness/traceback.txt，可能是其他异常。

## 后续正式任务的条件

先定位并处理变化来源；如果真实冻结状态受到修改，旧baseline不满足原对照条件，需要在修复后重新训练该组。若能证明是校验实现问题，必须保留证据并独立复核权重，而不是补写成功标志。其余三组正式训练尚未开始，没有正式成果要丢弃。新的SAR四组完整验证通过后，再把SAR＋soft明确绑定到该次有效来源，使用新的结果根目录。

不能把Memory的CPU池化替代直接用于SAR Q训练：后者需要Q汇聚路径的梯度。

## 本地验证

16项CPU测试通过：字节哈希与原函数一致、参数及buffer变化、signed zero/NaN、state_dict顺序变化、训练模式与梯度泄漏、CLI参数、默认只读行为、监视前后优化器更新相同、恢复协议环境及拒绝重复排队。

另用实际QVariantFront的随机初始化模型及64×64合成输入运行FP32和CPU-BF16各一步，冻结状态监视通过。这不代表服务器权重、4090、真实BRIGHT数据或2000步训练已通过。
