# V2区域下降诊断（独立目录）

不修改旧V2源码、原权重或正在运行的训练。依赖服务器现有V2及其原始依赖；无需新CUDA扩展。
只允许标准Train2918与Val450，禁止Test。使用可信本地checkpoint；禁止覆盖输出。

## A/B：只读诊断

设置FRONT_CKPT为固定V2前端best.pth，OUTPUT_ROOT为新目录。
run.sh preflight；run.sh smoke_audit；换新OUTPUT_ROOT后run.sh audit。
smoke只用2批Train裁剪和2张Val，不用于效果结论。

完整审计对比original、shared、split三种统计。仅fusion_layer_1/3的4个BN参与校准，
其余层保持eval。校准使用固定随机种子的Train512裁剪、batch2，默认128批、2遍；
第二遍使用第一遍统计，降低上游统计改变对下游输入的影响，但不能保证已经完全收敛。
使用FP32、逐通道全像素加权矩估计；不是EMA继续训练，也不是权重优化。
shared两次调用合并，split按调用次序分别统计。严格断言每层每次前向恰好调用两次。
两种方法同数据同预算；Train标签不参与统计计算。输出样本名和校准配置。
该结果是对归一化敏感性的诊断，不独立证明BN是训练退化的唯一原因。
若改善很小，不排除更大校准集或分辨率差异的影响；禁止用Val统计更新BN。

每种统计下评估pixel、resize_only、projection、region。
projection对pixel概率下采样后用Q做无注意力加权的软汇总再回投，不经过区域分类器。
不是region真实特征路径的完整复刻，也不是IoU上限。
记录全图累计混淆矩阵、分类别指标、逐事件指标、GT×pixel预测×region预测转移表。
没有实现边界/建筑尺寸统计，请勿误称已有。GT仅用于评估。
checkpoint步数和SHA记录在summary.json；最佳权重不一定来自最新训练步数。
开始和加载后检查文件SHA；若训练仍在覆盖best，建议先复制固定版本再运行。
不会保存完整模型副本，仅小型BN缓冲区及JSON；不会更新Memory。

## C：训练对照（先诊断，后按需要启动）

run.sh train a1：A1单独微调，L=定位CE+Lovasz+像素CE+Lovasz。
run.sh train joint：当前V2目标，L=定位+区域+0.25像素。
run.sh train joint_frozen_bn：仅在joint基础上冻结fusion1/3的BN运行统计，仿射参数仍更新。
固定的BN统计来自同一个A1初始化，不是从当前微调权重强行替换。
三个模式都从原A1开始，不从V2继续、不加载旧优化器。可分别使用smoke_train。
默认5000次更新，但LR沿用50000步计划和1000步预热，不能把5k短程改成5k余弦结束。
batch2×accum8=EBS16；AdamW、WD0.005、梯度裁剪1；只保存best和结束last。
先构造相同完整模型保证共享初始化一致；a1不执行新区域模块并冻结其参数。
同种子和数据顺序，但不同结构消耗随机数不同，不保证逐算子随机性相同。
每次验证完整Val450，初始像素指标必须复现原A1；smoke仅2张不作该断言。
日志grad_norm是裁剪前值。
该a1对照同时改变结构和目标，不能单独证明梯度冲突；joint与joint_frozen_bn为更窄对照。
输出采用REGION_DIAG_CONTROL_V1格式，不可直接喂给V2 Memory或V2原评估入口。
无resume。需要更改方案时使用独立输出，保留已有证据。不要自动终止已有GPU任务。

## 判读

- shared改善：重新估计统计可能有帮助，不证明一定需要分路BN。
- split额外改善：支持两种输入分布分开归一化的价值，仍需训练对照。
- resize_only到projection下降大：当前Q汇总/回投丢信息。
- projection接近pixel而region差：重点检查调制、区域表示和分类训练。
- A1继续微调也下降：基础微调过程需要检查。
- joint_frozen_bn优于joint：冻结这些统计在该预算下有效，不等于解决全部问题。

审计中全部模型权重校验不变；BN缓冲区用完恢复。前端源哈希仍按原V2校验，
新增目录不会改变V2源哈希。运行期间不要编辑V2及其依赖文件。
