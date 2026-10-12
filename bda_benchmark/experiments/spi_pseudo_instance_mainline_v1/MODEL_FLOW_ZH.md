# P3与伪实例语义主线：模型流程、因果关系与后续设计

## 1. 论文真正研究的问题

核心问题不是“能否把语义连通域切成漂亮的实例”，而是：

> 不使用官方实例参与训练或选择，仅从模型预测得到伪建筑结构，能否安全地
> 修正P3像素损毁判断，并在标准语义分割协议上超过60.5020%。

最终任务仍是四类像素语义分割：Background、Intact、Damaged、Destroyed。
官方实例只能用于独立诊断几何上限或建筑级指标，不能替代语义TIFF主指标。

## 2. P3底层数据流

```text
灾前RGB ──> RGB Encoder ──> F_rgb ──> Localization Head ──> Z_loc ──> P_B
                  │
                  └────────────────┐
                                   ├─> Cross-modal Fusion ─> F_fused
灾后SAR ──> SAR Encoder ──> F_sar ┘                         │
                                                             × W_B
                                                               │
                                                           F_fused_roi
                                                               │
                                                        Pixel Damage Head
                                                               │
                                                    Z0: 4类像素logits
```

建筑概率及ROI权重：

\[
P_B(x)=\operatorname{Softmax}(Z_{loc}(x))_{building}
\]

\[
W_B(x)=\alpha+(1-\alpha)\operatorname{stopgrad}(P_B(x)),\quad \alpha=0.1
\]

\[
F_{roi}(x)=F_{fused}(x)W_B(x),\qquad Z_0(x)=\operatorname{Conv}_{1\times1}(F_{roi}(x))
\]

`stopgrad`意味着损毁损失默认不能反向改变定位分支；`alpha=0.1`保证定位漏检
时仍保留少量损毁特征。

## 3. P3原有伪Bag为何不等于实例语义推理

训练阶段把语义GT按I/D/X分别做连通组件：

\[
\widetilde R_{c,k}=CC_4(\mathbf 1[Y=c]),\quad c\in\{I,D,X\}
\]

每个组件作为MICB/DSMIL的Bag，给共享特征施加辅助损失。但锁定P3的：

```text
mil_residual_scale = 0
prototype_bypassed = True
```

因此推理输出仍然完全来自像素头`Z0`。MICB从未被训练成安全的像素纠错器，
这解释了将MICB Bag标签硬广播后mIoU下降到57.5036%的现象。

## 4. 当前无需训练的伪结构反馈

### 4.1 LocCC区域

从P3定位预测生成建筑支持域：

\[
M_B(x)=\mathbf 1[\arg\max Z_{loc}(x)>0]
\]

\[
\mathcal R=CC_4(M_B),\quad |R_j|\ge32
\]

它是“预测建筑区域”，不是严格实例。同类相邻建筑仍可能处于同一组件。

### 4.2 区域证据与A6门控

区域多数类别：

\[
\hat c_j=\arg\max_{c\in\{I,D,X\}}
\sum_{x\in R_j}\mathbf1[\hat y_0(x)=c]
\]

区域纯度把整个定位区域作为分母，背景混入会降低可信度：

\[
\rho_j=\frac{\sum_{x\in R_j}\mathbf1[\hat y_0(x)=\hat c_j]}{|R_j|}
\]

平均概率与类别间隔：

\[
q_j(c)=\frac1{|R_j|}\sum_{x\in R_j}p_0(c|x)
\]

\[
m_j=q_j(\hat c_j)-\max_{c\ne\hat c_j}q_j(c)
\]

锁定门控：

\[
g_j=\mathbf1[\rho_j\ge0.70\land m_j\ge0.10]
\]

### 4.3 S1 Hard反馈

\[
\hat y_{S1}(x)=
\begin{cases}
\hat c_j,&x\in R_j,g_j=1\\
\hat y_0(x),&\text{otherwise}
\end{cases}
\]

它已经得到61.5399%，是之后所有模型创新必须超过的简单强基线。

### 4.4 S2 Soft Context

先归一化区域前景分布，再与像素概率融合：

\[
\bar q_j(c)=\frac{q_j(c)}{\sum_{r\in\{I,D,X\}}q_j(r)}
\]

\[
p_{S2}(x)=(1-\beta)p_0(x)+\beta\bar q_j,\quad \beta=0.75
\]

S2得到61.5716%，但只比S1高0.0317 pp。没有S2-S1置信区间前，不能
把Soft融合包装成已经成立的独立创新点。

### 4.5 S3不确定内部反馈

归一化熵：

\[
H(x)=-\frac{\sum_c p_0(c|x)\log p_0(c|x)}{\log4}
\]

不确定性权重：

\[
u(x)=\sigma\left(\frac{H(x)-\tau_u}{T}\right)
\]

内部权重`d_j(x)`在边界为0.2，向核心逐渐增加到1：

\[
\alpha_j(x)=g_j\,\beta\,u(x)d_j(x)
\]

\[
p_{S3}(x)=(1-\alpha_j(x))p_0(x)+\alpha_j(x)\bar q_j
\]

S3改善11/14事件，但Damaged F1只提高0.5936 pp，没有达到预注册1 pp。
固定门控提高了覆盖稳定性，却漏掉了大量可以安全纠正的像素。

### 4.6 S4局部对照

S4用相同的不确定性和内部权重，但把区域分布替换成匹配的5×5局部上下文。
它只提高0.0060 pp，而S3提高0.6424 pp，且结构特异性Bootstrap下界为正。
因此增益来自建筑范围结构，而不是任意局部平滑。

## 5. RGID在当前系统中的准确位置

```text
P3定位支持域
   │
   ├─ LocCC ───────────────────────────────> R_j
   │
   └─ Frozen P3 fused_roi ─> RGID Head
                              ├─ Center
                              └─ Boundary
                                   │
                          Watershed + Gate + Rollback
                                   │
                               R_{j,k}
```

当前实际RGID只有Center和Boundary，没有Offset。它是一种区域划分器，不是损毁
分类器。几何因果实验中必须保持P3概率与A6语义决策不变，只比较：

\[
\Delta_{geometry}=mIoU(RGID+A6)-mIoU(LocCC+A6)
\]

现有结果中RGID改变了分区，但没有超过LocCC，因此尚不能声称拆分带来额外
语义收益。其监督由Binary Localization CC生成，一个粘连CC只有一个Center且
没有内部边界，这与推理时希望它拆出多个建筑存在监督矛盾。

## 6. 当前证据链

```text
Stage-0真实实例池化上限明显高于P3
        │ 说明正确建筑归属有价值，但不证明可部署方法有效
        ▼
T1组件均衡训练未超过P3
        │ 排除“简单重加权就是答案”
        ▼
MICB广播显著下降
        │ 排除“Bag标签直接覆盖像素”
        ▼
LocCC + Majority超过P3
        │ 证明预测区域上下文具备纠错潜力
        ▼
S4局部对照近似零、结构CI为正
        │ 证明收益具有结构特异性
        ▼
S2最高但≈S1；S3更稳定但欠纠错
        │ 暴露固定融合强度与固定门控的局限
        ▼
下一步：可学习、风险感知、参数匹配的区域残差适配器
```

## 7. 推荐模型：PI-SRA

伪实例仅提供上下文与路由，不向整栋建筑广播硬标签。

### 7.1 可靠区域Token

\[
v_j^m=\frac{\sum_{x\in R_j}w_j(x)F_m(x)}
{\sum_{x\in R_j}w_j(x)},\quad m\in\{rgb,sar,fused\}
\]

\[
w_j(x)=C_{P3}(x)d_j(x)
\]

其中`C_P3`降低不可信像素贡献，`d_j`降低边界贡献。

### 7.2 像素—区域关系残差

\[
h_j(x)=[F(x),v_j,F(x)-v_j,F(x)\odot v_j]
\]

\[
\Delta Z_j(x)=f_{res}(h_j(x))
\]

区域Token不直接成为标签，而是帮助像素判断自己是否偏离建筑整体状态。

### 7.3 可学习风险门控

\[
z_j(x)=[H(p_0(x)),\rho_j,m_j,d_j(x),C_{P3}(x),r_j^{rgb},r_j^{sar}]
\]

\[
g_j(x)=\sigma(f_{gate}(z_j(x)))
\]

\[
Z(x)=Z_0(x)+g_j(x)\Delta Z_j(x)
\]

这统一了S2和S3：可靠区域允许充分纠错，跨域或不可信区域自动接近原始P3。

### 7.4 安全训练目标

第一阶段冻结P3，仅训练Adapter与Gate：

\[
L=L_{sem}+\lambda_gL_{gate}+\lambda_aL_{anchor}
\]

P3错误且区域建议正确的位置作为“可纠正错误”：

\[
e^+(x)=\mathbf1[\hat y_0(x)\ne Y(x)\land\hat y_R(x)=Y(x)]
\]

高置信、原本正确或区域不可靠位置使用P3锚定：

\[
L_{anchor}=\frac1{|\Omega_{safe}|}
\sum_{x\in\Omega_{safe}}KL(p_0(x)\Vert p(x))
\]

与T1不同，这不是给每个组件相同权重；与MICB广播不同，这不是把一个区域
标签强加给所有像素。

## 8. 下一阶段最小可证伪实验

| 编号 | 模型 | 唯一变化 | 回答的问题 |
|---|---|---|---|
| M0 | Frozen P3 | 无 | 60.5020%基线 |
| M1 | P3普通微调 | 训练步数 | 排除额外训练收益 |
| M2 | 参数匹配Local Adapter | 5×5上下文 | 排除参数量/局部上下文 |
| M3 | LocCC Region Adapter | 换成区域Token | 结构Token是否有效 |
| M4 | M3 + learned gate | 增加风险门控 | 是否减少错误修改 |
| M5 | M4 + core/boundary | 增加几何可靠性 | 边界保护是否有效 |
| M6 | M5 + modality routing | RGB/SAR可靠性 | 是否抑制屋顶纹理和SAR强散射假证据 |
| M7 | M6 + prototype contrast | 困难像素原型 | 是否改善D/X判别 |
| M8 | M7中LocCC换RGID | 仅换分区 | 拆分是否提供额外价值 |

最重要的因果比较：

\[
M3-M2,\qquad M4-M3,\qquad M8-M7
\]

先做M2-M4的500步单seed Pilot。通过条件应在训练前写入新协议：M3显著优于
M2、M4优于M3、M4超过61.5399%、Damaged F1不下降、任一前景IoU下降不
超过0.5 pp。通过后才扩展到3000步和三seed。

## 9. 速度与实现边界

当前主要瓶颈不是4090，而是CPU对大量小连通域反复创建全图Mask、整图融合与
腐蚀。等价优化必须满足：

1. 连通域先统计面积，再为保留组件创建Mask；
2. 每个区域只在自身像素或包围盒计算；
3. 变化统计只对区域内部执行argmax；
4. 优化前后区域数量、混淆矩阵和changed/corrected/harmed严格一致；
5. `verify_stack.py`确认服务器导入的函数确实支持`minimum_area`。

`NUM_WORKERS`只影响数据加载，无法解决主进程的区域循环。先用性能Smoke判断
`data_wait/model_forward/connected_regions/four_refinements`各自耗时，再决定
Worker数量。

## 10. 论文定位

当前最稳妥的主张是：

> 冻结语义模型的预测建筑结构可以提供超越局部上下文的语义纠错信号；然而，
> 固定区域广播在总体收益和跨事件稳定性之间存在冲突，因此需要风险感知的
> 像素选择性区域残差适配。

RGID应作为可替换区域生成器和几何消融，而不是论文唯一支柱。即使RGID继续
不优于LocCC，PI-SRA的核心命题仍可通过M3-M2和M4-M3独立成立。

