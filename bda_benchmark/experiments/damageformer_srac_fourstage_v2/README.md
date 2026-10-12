# DamageFormer SRAC 四阶段改编 V2

本目录独立于 damageformer_srac_swin_v1；没有修改旧实验文件，不自动停止旧进程。
目标是尽量实现用户提供的SRAC架构图中的四阶段流程，**不是已验证的官方论文复现**。
没有获得原论文完整代码/配置，下面明确区分图示对应机制与本实现选择，不承诺75.65%或超越A1。

## 四阶段与图示的对应

1. RGB Swin多尺度特征 → 轻量UNet++式嵌套解码器 → 三轮可微局部中心更新，得到软归属Q。
   分区是SSN思路的局部聚类，不是官方VMambaSSN。每位置至多9个附近种子，不是SAR位移搜索。
2. 原RGB编码得到预测建筑概率B，Q的argmax给出硬边界E。RGB反归一化至原0..255数值域，先按0.1+0.9B衰减，再以0.15E向白色突出边界，重新归一化并送回共享RGB Swin。
   SAR影像不作亮度重写，保留原SAR Swin特征，与第二次RGB编码融合。不是V1的融合特征层卷积调制。
   B/E在调制处detach，RGB编码器两次前向共享权重；这是控制成本的实现选择，图示不能证明原文也共享。
3. 融合特征 → 注意力加权软Pooling、mean/std旁路 → 128维Token → 区域四分类 → Q概率混合回投。
4. 冻结最佳前端，Train全图提取Token及预测区域几何，按事件×多数类别每桶最多64个真实样本构库。
   预计算Train查询的Top8邻居ID，保存neighbors.npy内存映射表。分类器读取ID再Gather支持特征和元信息，Cross-Attention+FFN分类，再回投全图。

### 元信息的真实含义

- Label：仅历史Train支持样本的四类软标签分布，不是查询GT。
- Disaster：由Train事件名后缀映射到固定灾害类型词表，未知为unknown。
- Region：Train事件ID嵌入，**仅为区域/来源代理，不是真实经纬度或正式地理区域编码**。事件与地理可能混杂，不能声称已消除风格偏差。
- Geometry：由预测Q计算的归一化面积、y/x质心、y/x标准差，共5维，不使用GT实例。
- 查询只提供Token；支持元信息加入attention Value，Key仍为支持Token。排除同图、同事件，无合格邻居时使用自身Token回退。

图中未明确而本版自行设定：调制系数、3轮聚类、Token宽度、均衡抽样预算、检索度量、排除策略、soft标签、训练损失与阶段划分。不存在官方一致性保证。
没有独立紧致度损失、连通性后处理、真实地理字段、多原型、对比学习。硬区域不保证连通；不存在建筑/背景硬分组，跨边界混合仍需评估。

## 梯度与训练

前端从已核验A1任务权重完整strict加载。A1 lr1e-5，新模块1e-4，AdamW wd5e-3，前1000步warmup再cosine，EBS16，50,000次新更新。
因为RGB第二次编码增加显存，脚本默认batch2×accum8（仍为16）；不保证任意GPU一定足够，可batch1×accum16，但需重新冒烟。
训练随机crop512，完整Val450、1024输入，每1000步验证。正式训练前先复现原像素分支0.6168659209491562，失败停止。
区域损失通过Q的Pooling和回投路径反传到分区和编码器。B与硬边界不经调制路径反传；定位头有单独监督。
总损失=定位CE+Lovasz + 区域NLL+Lovasz + 0.25×像素辅助CE+Lovasz。
保留二维像素表损失、CPU标签检查、CPU独立int64→同步GPU搬运逐值校验；没有修改GT类别。
区域输出不是原像素logits加残差。随机新增区域头开始时不等于A1成绩。

后续前端冻结，Train缓存建立后再训练两个同参数量分类头，各5000步，每500步完整Val选最佳。
nomemory用可学习槽位，不读取历史样本；memory使用邻居特征和元信息。两者计算量不严格相同。
标签监督使用区域软类别分布、特征网格有效质量加权，不是精确原图每像素CE等价替代。
检索损失不更新前端，避免库向量过期；这不是四阶段全程联合训练。

## 邻居表与缓存保护

bank仅来自Train2918。构库时允许Train标签用于监督/均衡抽样，不允许Query标签影响检索。
Train查询邻居预计算只读取特征、图像ID、事件ID，不读取查询标签。neighbors.npy是int32 ID表，无合格项=-1。
大表通过numpy mmap读，不复制全表；较小bank仍装入内存/GPU，不能宣称全部特征都采用官方MMAP实现。
Val特征在线生成、在线检索同一个Train库，不将Val加入库。CPU离线检索与GPU在线检索可能在浮点近似并列邻居时出现差异。
前端、bank、分片、邻居表绑定hash；禁止旧V1缓存/权重混用。源码哈希覆盖本目录py文件及主要依赖，但不是所有第三方依赖。
无Test入口。若需要最终Test，应先冻结配置，再单独实现明确授权的评估入口。

## 安装与启动

上传完整目录到：
`/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_fourstage_v2`
依赖现有BRIGHT仓库的DamageFormer与旧区域基础算子，不需要新CUDA扩展。不在运行中修改代码。

```bash
cd /root/shared-nvme/BRIGHT-master
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export BATCH=2
export ACCUM=8
export AMP=bf16
export BASE_LR=1e-5
export NEW_LR=1e-4
unset DIAGNOSTIC_STEPS CUDA_LAUNCH_BLOCKING TORCH_SHOW_CPP_STACKTRACES

conda run --no-capture-output -p /root/shared-nvme/conda-envs/bright4090 \
 bash bda_benchmark/experiments/damageformer_srac_fourstage_v2/run_a1.sh preflight

export RUN_ROOT=/root/shared-nvme/BRIGHT-checkpoints/DAMAGEFORMER_SRAC_FOURSTAGE_V2/smoke_seed3407
conda run --no-capture-output -p /root/shared-nvme/conda-envs/bright4090 \
 bash bda_benchmark/experiments/damageformer_srac_fourstage_v2/run_a1.sh smoke_all
```

冒烟：前端20步，4张跨事件Train图构库，邻居表，两个头各20步，再完成多种Val干预。Val仅前2张，不能评价精度。
这版结构与V1不同，V1冒烟通过不代表V2通过。输出已存在时换新目录，不删除旧结果。

冒烟通过后，可先运行500步诊断（正式LR计划，自动提前退出，不保存诊断模型）：

```bash
export RUN_ROOT=/root/shared-nvme/BRIGHT-checkpoints/DAMAGEFORMER_SRAC_FOURSTAGE_V2/diag500_seed3407
export DIAGNOSTIC_STEPS=500
conda run --no-capture-output -p /root/shared-nvme/conda-envs/bright4090 \
 bash bda_benchmark/experiments/damageformer_srac_fourstage_v2/run_a1.sh regions
unset DIAGNOSTIC_STEPS
```

正式全流程（确保GPU空闲、目录全新）：

```bash
export RUN_ROOT=/root/shared-nvme/BRIGHT-checkpoints/DAMAGEFORMER_SRAC_FOURSTAGE_V2/formal_seed3407
mkdir -p /root/shared-nvme/BRIGHT-train-logs/DAMAGEFORMER_SRAC_FOURSTAGE_V2
nohup conda run --no-capture-output -p /root/shared-nvme/conda-envs/bright4090 \
 bash bda_benchmark/experiments/damageformer_srac_fourstage_v2/run_a1.sh pipeline \
 > /root/shared-nvme/BRIGHT-train-logs/DAMAGEFORMER_SRAC_FOURSTAGE_V2/formal_seed3407.log 2>&1 &
echo "PID=$!"
```

`pipeline`依次regions→build_memory→build_neighbors→train_heads→evaluate，任一步错误即停止。
也可分别执行这五个命令，前端阶段不满意时不必自动继续。Memory确实在后两阶段启用，regions没有Memory。
不支持阶段内恢复训练；不得把diag结果当成正式完成。只保存各训练阶段best/last，不每步保存。
磁盘粗估正式新增约3–4GiB，建议至少6GB空闲；还需计算保留的旧V1与冒烟权重空间。邻居表约91MiB（2918×1024×8×4字节）。
没有自动删除旧文件或停止任何进程；不要同时在同一GPU启动旧V1和新V2。

## 必须报告的对照

1. 原A1历史参照与当前共训练pixel分支要区分。
2. region_only：前端原区域分类头。
3. nomemory：同参数量但无历史支持的头。
4. memory_normal：真实历史特征、标签和元信息。
5. memory_random / shuffle_labels / off / no_metadata：固定模型推理干预。
   no_metadata保留支持标签，只去掉灾害、事件代理和几何；不能作为独立训练消融或证明泛化。

全部全图四类统计，输出前景mIoU、每类指标及每事件结果；不使用GT前景筛选查询像素。
不把训练loss下降当作超像素有效证据，不把元信息收益当作跨地区泛化证据。
建议确认memory相对nomemory正增益后重复种子与随机干预，再做正式消融。
