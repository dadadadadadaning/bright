# SRAC Support Ablation V1

本机CPU测试：6项unittest通过，含合成Train/Val的E0–E5端到端训练与干预。
未在服务器真实缓存或CUDA上执行。所需依赖为PyTorch及Python标准库。

独立目录，原模型与历史缓存只读。连续执行E0/E1/E2/E3训练，再运行E4/E5评估干预。
只保存各训练分支best和last。best按Val区域Macro-F1选择，验证集用于模型选择，非独立Test证据。

## 输入和适用范围

复用 `spi_srac_mechanism_ablation_v1` 或 `changemamba_deep_region_retrieval_v1` 的采样Token缓存：
`tokens[source]: [N,P,C]`、`token_mask: [N,P]`、`labels: [N]`（1/2/3）、
`events`、`images`、`split`（train/validation）、`test_accessed=false`。
不接受只有mean的flat记录或建筑级聚合缓存。来源由manifest记录；ChangeMamba缓存不能被称为A1特征实验。
默认source为base_change,explicit_diff。缓存缺少字段时显式失败，不猜测特征来源。

注意：缓存采样点不是完整区域像素，mean/std只表示采样特征统计；本实验验证缓存表示，不能宣称完整像素池化。
现有缓存可能按GT有效前景筛除区域，因此结果是缓存覆盖区域上的三类分类，不能外推到漏检建筑或背景。

## 实验定义

- E0：采样特征projection、masked mean、分类；这是新训练的mean基线，不冒充旧checkpoint的复现。
- E1：同源采样特征attention pooling，加原始mean/std可学习旁路；E1-E0同时改变池化及部分参数量。
- E2：冻结E1，query加三个可学习类槽，经过attention+FFN分类；无历史记录。
- E3：冻结相同E1，使用真实Train邻居特征作为K，特征加支持标签embedding作为V。
  E2/E3使用相同模块、参数数目、初始化、采样和预算；所有参数参与训练。计算量因support长度不同不相等。
- E4：复用E3 best，在同样跨事件/跨图约束下随机抽取支持条目，标签随条目同行。
- E5：复用E3 best，固定真实邻居特征和索引，对整个训练bank的标签随机置换；保持全局类别计数。
  随机置换不是强制每个标签错误，输出changed_label_fraction。E4/E5默认各5次，输出均值和标准差。

Memory按Train类别×事件等上限抽样，固定E1后建立，cosine top-k。
每个Train或Val query都排除同事件及同图像，候选不足报错。事件ID不进入网络。
E1训练见过Train本身，故这不是整条特征链的严格OOF；若研究未见事件泛化需另做事件留出训练。
E4/E5是评估期破坏实验，存在分布变化，不能单独证明泛化；以E3-E2为主要增量对照。

## Linux运行

```bash
cd /root/data/BRIGHT-master
conda activate bright4090
bash bda_benchmark/experiments/srac_support_ablation_v1/run.sh preflight
export TRAIN_RECORDS=/root/data/BRIGHT-checkpoints/CHANGEMAMBA_DEEP_REGION_RETRIEVAL_V1/slico_train_records.pt
export VAL_RECORDS=/root/data/BRIGHT-checkpoints/CHANGEMAMBA_DEEP_REGION_RETRIEVAL_V1/slico_validation_records.pt
export OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/SRAC_SUPPORT_ABLATION_V1/cm_seed3407
mkdir -p /root/data/BRIGHT-train-logs/SRAC_SUPPORT_ABLATION_V1
nohup bash bda_benchmark/experiments/srac_support_ablation_v1/run.sh all > /root/data/BRIGHT-train-logs/SRAC_SUPPORT_ABLATION_V1/cm_seed3407.log 2>&1 &
```

先用 `STEPS=20 EVAL_EVERY=20 REPEATS=2 OUTPUT_ROOT=另一个新目录` 做完整小步数检查。
每次输出目录必须是新目录，RUN_STARTED.json通过排他创建防止并发覆盖。失败后保留证据并选择新目录重跑。
脚本不生成原图缓存、不加载骨干、不访问Test、不自动启动服务器作业，不含像素回写。

## 输出与判断

summary.json：区域Accuracy、Macro-F1、三类P/R/F1、每事件指标、E1-E0、E2-E1、E3-E2及随机干预均值。
若缓存含基线区域预测，还输出rescue/harm/net count；这仍是区域数量，不是像素收益。
neighbor_ids.pt、train_memory.pt、各分支predictions.pt用于回溯。
manifest.json包含输入SHA256和缓存元信息。后续固定配置重复3个种子，再做事件分层不确定性评估。
本实验不输出pixel mIoU；最终改善还需另行使用一致的像素读出与Val全图评估证明。
