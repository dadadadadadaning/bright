# SAR 辅助 Q：四组匹配训练与完整 Val450

本工具复用服务器中已有的 `bda_benchmark/experiments/damageformer_srac_q_factorial_v1`，不替换其模型、损失、Q 池化或验证实现。代码放在 `/root/shared-nvme/BRIGHT-master/New`；新结果放在 `/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_Q_MATCHED_V1`。不更新旧 Memory 库，不训练 Reader，不访问 Test。

## 先理解 soft 与 SAR→Q

原 Q 给出位置 i 对区域 r 的分配权重。soft 路由将每个区域分成背景槽与建筑槽：

```
q_background(i,r) = Q(i,r) × (1 − B(i))
q_building(i,r)   = Q(i,r) × B(i)
```

B 是冻结 A1 预测的建筑概率，非真值。默认不加 floor。Q=0.8、B=0.7 时，该位置对两个槽分别贡献0.24、0.56。原区域内重新池化，再以相同权重回投；不是在原混合 token 生成后切开。hard 将 B 先以0.5阈值二值化。soft 没有 Memory 接受/拒绝门控，也不能自动分开建筑内部的 Intact/Damaged。

旧路由实验自身匹配结果：unsplit61.4665%，capacity_control61.4512%，hard61.5463%，soft61.7976%。soft 相对 unsplit +0.3311个百分点。这是其自身训练设置的结果，不能拿来直接加到其他 baseline 的分数上。

SAR→Q 改的是“位置应分给哪个区域”。A1 原本就使用 RGB+SAR，本轮新增的是 SAR 对区域构造的影响，不是给整个模型第一次加入 SAR。

## 已有实现如何加入 SAR

实际实现：`damageformer_srac_q_factorial_v1/q_variants.py::CuePartition`。

1. 取同次冻结 A1 前向的 RGB256 与 SAR256 特征；沿用既有归一化与特征，不另对 SAR 随意 log/归一化。
2. 拼接为512维，插值到 Q 网格，经可训练1×1投影并归一化，得到线索 c。
3. 使用当前 Q 求区域线索均值，并在原9个局部候选分数上加入 `0.1 × cosine(c_i,c_r)`，再 softmax。原结构相似度、空间距离、建筑相容度、候选范围和迭代次数保留。

```
score_new(i,r) = score_original(i,r) + 0.1 × cosine(c_i, c_r)
Q_new(i,·) = softmax(score_new(i,·))
```

SAR可能帮助区分 RGB 结构相似、灾后状态不同的位置，但斑点、叠掩、阴影、尺度和配准残差也可能干扰区域构造。这个公式是待检验方案，不是已证明的改进。

## 本轮四组

|组|新增 Q cue 输入|回答的问题|
|---|---|---|
|baseline|没有新增 cue；原 Q 继续训练|匹配2000步训练本身的效果|
|rgb_capacity|concat(RGB, RGB)|新增投影参数/路径能否解释收益|
|shifted_sar|concat(RGB, roll(SAR))|对齐空间线索是否重要|
|sar_cue|concat(RGB, SAR)|额外 SAR 对 Q 是否有价值|

三个 cue 组的投影形状、参数量和初始权重相同。baseline 不增加 cue，因此需要 rgb_capacity 控制新增参数。错位使用 SAR **特征图**高度/宽度各半幅的循环平移；只影响新增 cue，不移动 A1 主干原有 SAR。这个对照检验大幅空间破坏，不等价于真实小配准误差鲁棒性，且仍保留事件纹理。

固定同一个 passed E0、同一初始化种子和训练采样配置。仅 Q decoder/partition 及新增 cue 投影训练；A1、原区域 embedding/classifier 冻结。四组均 Train2918，2000次更新，batch2×累积8=每次16个随机512裁块；32,000次裁块呈现不是32,000张不同图像。学习率1e-4、AdamW、默认BF16、cue_gain0.1。前三组不使用建筑/语义重建辅助损失，SAR组也不使用；保留参数 aux_weight0.1 是原训练器固定配置，此四组实际没有对应重建项。

所有组使用 CUDA_LAUNCH_BLOCKING=1，保留 E0 的 CUBLAS、TF32等数值配置。该设置曾通过相关冻结流程重放，不能保证所有未来 CUDA 路径永不出错；出错保留现场并停止后续组，不自动重试、切CPU或放松门槛。

## 运行顺序

1. 只读验证 E0/输入文件并记录绑定；通过 nvidia-smi 等待目标 GPU 连续两次空闲。父进程不导入Torch、不先占CUDA。排队只观察空闲，不会杀掉已有任务，也不能阻止其他人随后启动任务。
2. 原 Q 包 CPU preflight。
3. 四组各2次更新的真实 smoke，再各2张独立验证。四组全部通过才继续。
4. 四组顺序各2000次正式更新，每组使用自己的全新模型；每组末步 `last.pth` 完整独立 Val450。
5. 检查四组来源/训练配置一致，逐图核对保存预测、真值、混淆矩阵与指标。
6. SAR与baseline保存预测配对分析；两个Q的事后语义混合诊断。

原训练器内部也会做初始/末步 Val450，独立评价还会重放原E0路径，所以总耗时不能只按8000步估算。完整过程顺序跑，不与当前 Memory 训练争抢 GPU。需要日志中的实际步耗时才能估算剩余时间。

## 安装与启动

下载发布的 **bright_sar_q_matched_v1.zip**，上传为：

```
/root/shared-nvme/BRIGHT-master/New/bright_sar_q_matched_v1.zip
```

使用发布说明中的ZIP SHA256核对后，解压到 New 下新目录。解压内部为 `sar_q_matched_v1/`：

```bash
PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
UNPACK="$(mktemp -d /root/shared-nvme/BRIGHT-master/New/sar_q_XXXXXXXX)"
"$PYTHON" -m zipfile -e /root/shared-nvme/BRIGHT-master/New/bright_sar_q_matched_v1.zip "$UNPACK"
bash "$UNPACK/sar_q_matched_v1/start_server.sh"
```

`start_server.sh`先验证工具哈希、跑合成CPU测试，再创建后台进程，打印实际日志与结果路径。不要重复启动同一任务。缺文件时先核对上传文件名，不要直接改成不明ZIP。

默认使用已通过的源协议：
`/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json`。
必要时可通过 `SAR_PROTOCOL` 指定另一个完整通过、legacy/native/all 的E0；四组必须同源。

只查看计划，不训练：

```bash
"$PYTHON" "$UNPACK/sar_q_matched_v1/pipeline.py" plan
```

报告命令由启动器按实际路径打印；`pipeline.py report --root 结果目录` 是只读操作，失败/未完成也可以读，不做 resume。`waiting_gpu` 是排队；`smoke_*` 只查流程；`formal_train_*` 是训练；`formal_val_*` 是验证。进度450/450和进程PID都不等于正式通过。最终看 `workflow_state.json`、`summary.json` 的 `complete` 与 `full_validation_evidence`，以及 `sar_verification.json` 的全量存档检查。

## 如何解释结果

主指标为 `q_region` 的 **Val前景三类mIoU**，因为本轮Q训练损失经过冻结区域分类头。另记录 `q_grid_probq`，它回答未来像素/区域联合设计关心的概率共享路径是否受益。两条都预先报告，不事后只挑上涨者；背景IoU单列，不和Test四类平均混用。

- SAR > baseline：当前设置下的候选收益。
- SAR > RGB：支持收益不只是多加投影参数。
- SAR > shifted-SAR：支持空间对应线索的贡献；仍不是物理因果机制完整证明。
- 一次小幅优势需要匹配重复种子才能讨论稳定性。库中的原训练器种子绑定E0，不能仅改一个壳变量假称不同seed；后续应建立对应可核验的重复协议。
- 正式完成但SAR没涨分也是有效结果。若新增cue分数远小于原结构分数，先看记录的 `cue_score_mean_abs/base_score_mean_abs` 与Q变化，再设独立gain试验；不要同时改gain、重建损失和库。
- 若仅q_region改善、q_grid_probq不改善，可能只是适配冻结区域头，不应直接宣称有利于联合Memory路径。

同一套验证还提供每类与逐事件指标、像素救回/误改、Q混合/边界事后统计。Val真值仅用于评价与事后诊断，不进入Q或检索输入。单次val比较不能证明整个超像素路线已找到唯一根因。

## 和 soft / Memory 后续如何结合

先完成当前四组。本轮不把soft和语义重建一起加入SAR组。如果SAR有稳定收益，后续再做“原Q/SAR-Q × unsplit/soft”的2×2匹配实验，四组重新训练相同下游头，以检验叠加/冲突。已有 `building_route_v1` 的候选Q加载器仅接受特定重建来源，不应直接把SAR checkpoint塞进去绕过检查。

Memory联合方向仍然是像素、区域、历史案例共同分类，无拒绝门控。若选用新Q，区域身份、特征与软标签随之变化：支持库与查询必须用同一新Q及汇总规则重新生成，再比较有/无Memory的同容量联合头。不能把新Q查询直接接旧Q库后宣称是公平组合。

## 文献与官方实现

已核对官方README/源码；本环境未读取被拒绝访问的论文正文，不将下列工作当作本项目的成绩保证。

- SSN，ECCV2018，[Superpixel Sampling Networks官方代码](https://github.com/NVlabs/ssn_superpixels)：可微区域分配；软区域不必等于独栋建筑，也不自动保证连通。
- SpixelFCN，CVPR2020，[论文](https://arxiv.org/abs/2003.12929)、[官方代码](https://github.com/fuy34/superpixel_fcn)、[语义/位置重建实现](https://github.com/fuy34/superpixel_fcn/blob/master/loss.py)：Train监督可帮助Q保留类别边界，语义重建应单独设实验因素。
- BRIGHT，ESSD2025，[官方仓库](https://github.com/ChenHongruixuan/BRIGHT)、[跨模态配准说明](https://github.com/ChenHongruixuan/BRIGHT/blob/master/umim_benchmark/README.md)：配准评价独立于损毁分类，支持明确设置空间扰动对照。
- CFOG，TGRS2019，[多模态遥感配准官方实现](https://github.com/yeyuanxin110/CFOG)：跨模态结构描述有方法依据，但不是可直接证明SAR-Q增益的分类模型。

本包测试为CPU合成流程/存档契约检查。真实BRIGHT、CUDA、训练成绩需在研究服务器运行后验证。
