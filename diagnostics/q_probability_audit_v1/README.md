# Q概率汇总：只读Val450配对诊断

本工具解释已保存的 `q_a1_pixel`、`q_pixel_grid`、`q_grid_probq` 预测之间的救回与误改。
默认读取已完成的 SAR Q baseline Val450：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_FULL_CHAIN_V1/restored_20261014_134825_xno2hi6o/sar_q/formal/val_baseline
```

程序使用CPU及NumPy，不导入Torch、不加载模型、不执行CUDA前向、不训练、不访问Test。
它需要原仓库的 `damageformer_srac_q_factorial_v1/compare.py`，沿用其正式Val校验入口；额外核对源报告、协议、预测文件SHA256，以及逐图、逐事件、全局的整数混淆矩阵和IoU。

## 运行

下载单文件 `RUN_Q_PROBABILITY_AUDIT_V1.py` 到服务器：

```
/root/shared-nvme/BRIGHT-master/New/RUN_Q_PROBABILITY_AUDIT_V1.py
```

执行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -u /root/shared-nvme/BRIGHT-master/New/RUN_Q_PROBABILITY_AUDIT_V1.py
```

它在后台运行，打印PID、结果目录、日志和报告命令。无需设置旧的环境变量。
默认结果父目录为 `/root/shared-nvme/BRIGHT-checkpoints/SRAC_Q_PROBABILITY_AUDIT_V1`；不会覆盖旧目录。同一来源已有诊断运行时拒绝重复启动。
默认单线程CPU统计，顺序读取约2G现有存档；运行时长取决于服务器的共享盘速度和CPU负载。

选项：

- `--source`：另一个合格的同格式Val450存档目录。
- `--repo`：BRIGHT仓库，默认 `/root/shared-nvme/BRIGHT-master`。
- `--output-parent`：新分析结果的父目录。
- `--boundary-radius`：建筑边界扩展半径，默认2个原图像素。
- `--limit 2`：仅处理前2张；这是不完整诊断，不能视为Val450分析。默认0表示全部450张。
- `--report /结果目录`：读取已完成报告，不重新分析。

程序需要读取原 `predictions/*.npz`；分析完成前保留该目录。它不删除源文件，也不会把源预测复制到结果目录。

## 分析内容

三组配对关系：

| 配对 | 含义 |
|---|---|
| a1_to_pixel | A1到无Q的网格概率解码路径 |
| pixel_to_q | 相同网格概率路径上增加Q汇总 |
| a1_to_q | 相对原始A1的最终净变化 |

每个配对保存 `GT × 旧预测 × 新预测` 的4×4×4整数计数。
救回是旧错新对；误改是旧对新错；二者之差等于该真实类别正确像素数的变化。
另外统计每类误报的新增、移除及净变化，避免把“更多TP”直接解释为更高IoU。

逐图结果累积后，必须与原报告的三个预测分支逐事件和全局混淆矩阵一致。完整450张通过之后才会记录 `diagnostic_complete=true`。
这是对源预测的复核与事后诊断，不是一项新模型效果验证。

## 边缘与Q混合分组

这些分组都在原模型预测完成后用Val真值计算，仅用于解释，不送入模型。

- 建筑边缘：GT建筑/背景的四邻接边界，标记两侧像素，再做默认半径2的方形扩展；图像外部不当作背景。
- 255忽略标签附近：独立分组，避免把未标注边缘当作真实建筑边界。
- 背景主导Q区域：前景比例小于0.2。
- 背景/建筑混合Q区域：前景比例至少0.2且小于0.8。
- 前景类别混合Q区域：前景比例至少0.8，前景内部最大类比例小于0.8。
- 前景纯净Q区域：前景比例至少0.8，前景内部最大类比例至少0.8。

区域标签按保存的当前 `q_new` 和 `region_ids_new` 加权汇总GT面积比例，排除255。
要求原图与Q网格可作整数倍面积下采样；不满足时停止，不切换近似插值。
像素归组使用其特征网格中最大Q权重对应的区域，并作整数倍最近展开；并列时取首个路由。
**这只是诊断归组规则，不等于原模型的完整软Q及双线性解码依赖范围。** 原预测始终直接读取，不重建、不修改。

所有分组都保留类别分母与误改率，不能只比较组内误改总数。
区域不是建筑实例；没有逐栋标签时，不声称统计了建筑数量或逐栋一致性。

## 输出

- `report.txt`：主要分数、每类救回/误改、误改去向、分组重点和逐事件差值。
- `result.json`：完整配对计数、混淆矩阵、precision/recall/IoU、来源SHA256及定义。
- `groups.csv`：类别×边缘/Q混合分组的救回、误改、净TP/FP及相应比例。
- `events.csv`：逐事件配对分数与损毁类别救回/误改。
- `images.jsonl`：逐图转移计数和分组像素数。
- `summary.json`：阶段、处理图片数、完整诊断标志及异常。

只保存小体积统计文件，不输出新的NPZ预测数组。
没有 `diagnostic_complete=true`、`source_confusions_reproduced=true` 时，不能宣称完整复核了Val450。
GT分组是描述性证据，不能独立证明因果、统计显著性或Test泛化。
A1到pixel_grid的路径变化可能同时包含额外分类器调用的数值差异，不据此声称已分离纯解码顺序因果效应。

## 验证范围

CPU测试使用合成图像及完整450条存档格式夹具，覆盖原formal入口、输入哈希、矩阵复核、Q加权和边缘规则、后台启动、报告读取及源文件不被修改。
测试结果见 `CPU_VALIDATION.json`。真实服务器预测存档尚需在服务器实际运行验证。
