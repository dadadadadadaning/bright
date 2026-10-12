# S8-P6 A4 Soft-Critical MICB阶段汇报

## 1. 研究背景与问题

当前P3在Validation上取得60.5019 Valid mIoU、69.5772 mIoU4和90.57
Localization F1。定位与Intact识别稳定，主要错误是弱Damaged未进入Affected，
Validation Damaged gate-miss约33.09%。

已有受控实验给出两个局部正向信号：A4原生单通道SAR输入适配在Step 1000
达到60.5751 Valid mIoU；关键实例监督是P3相对像素基线的重要训练机制。
N1-shared Cross-Attention虽然相对N0提高Valid和Destroyed，但增加Gate-miss并
降低Event worst，因此不进入本轮组合。

## 2. 模型总览

```mermaid
flowchart LR
    RGB["灾前RGB 3×H×W"] --> RE["P3 RGB Swin-T\nStage1-4"]
    SAR["灾后原生SAR 1×H×W"] --> A4["A4函数保持PatchEmbed\n1→96, k4/s4"]
    A4 --> SE["P3 SAR Swin-T\nStage1-4"]

    RE --> RF["RGB多尺度融合\n1440→256"]
    SE --> SF["SAR多尺度融合\n1440→256"]
    RF --> LOC["RGB定位头\n256→2"]
    LOC --> PB["建筑概率 P_B\n停止梯度"]
    RF --> CAT["Concat RGB/SAR"]
    SF --> CAT
    CAT --> F3["fusion_layer_3\n512→256"]
    PB --> ROI["W_B=0.1+0.9P_B"]
    F3 --> ROI
    ROI --> CLF["P3四类像素头\n256→4"]
    CLF --> OUT["Background/Intact/\nDamaged/Destroyed"]

    RF -.训练期.-> TOKEN["[R,S,|R-S|,R×S,F]\n建筑内Token"]
    SF -.训练期.-> TOKEN
    ROI -.训练期.-> TOKEN
    TOKEN --> SC["Soft-Critical\n多关键实例聚合"]
    SC --> HIER["Occurrence + Severity\n建筑级辅助损失"]
    HIER -.辅助梯度.-> F3
```

正式推理保持`mil_residual_scale=0`，Soft-Critical不直接广播到像素logits。

## 3. Soft-Critical底层计算

```mermaid
flowchart TB
    BAG["一个训练建筑Bag\nN个跨模态Token"] --> SCORE["local_exist: 每个Token得到a_i"]
    SCORE --> HARD["P3 Hard Critical\nargmax(a_i), 单点选择"]
    SCORE --> SOFT["P6 Softmax(a_i/τ)\nτ=0.5"]
    SOFT --> WEIGHT["w_i, 所有高响应位置获得权重"]
    WEIGHT --> CRIT["t_soft=Σw_i t_i"]
    SCORE --> LSE["归一化LogSumExp\n消除Bag大小偏置"]
    CRIT --> QUERY["作为DSMIL Attention Query"]
    QUERY --> ATT["聚合相似局部损毁位置"]
    LSE --> OCC["Intact/Affected辅助头"]
    ATT --> OCC
    OCC --> GRAD["梯度回到融合表示\n不直接修改测试预测"]
```

该修改不增加可训练参数，P3 checkpoint参数命名和严格加载协议保持不变。

## 4. 两阶段4000步训练

```mermaid
flowchart LR
    S0["Step 0\nA4函数保持初始化\n精确复现P3"] --> S1["Step 1-1000\n仅训练A4 SAR PatchEmbed\nLR=1e-4\nMICB warm-up"]
    S1 --> LOCK["Step 1000锁定A4输入头\n防止输入表示继续漂移"]
    LOCK --> S2["Step 1001-4000\n仅训练fusion_layer_3+clf_2\nLR=1e-6\n启用Soft-Critical Bag损失"]
    S2 --> V["每500步完整Validation"]
    V --> G["Valid/Damaged/Recall/\nGate-miss/Event mean/worst"]
    G -->|连续两次低于P3 0.3pp| STOP["提前停止"]
    G -->|4000步完成| GATE["严格Advancement Gate"]
    GATE -->|通过| REP["补充多种子复现\n仍不立即Test"]
    GATE -->|失败| END["停止P6\n不加像素Occurrence头"]
```

## 5. 相对已有模型的受控关系

| 模型 | 本轮采用 | 原因 |
|---|---|---|
| P3 RGB定位/ROI/像素头 | 是 | 已在Validation和Test成立 |
| A4函数保持Native-SAR输入 | 是 | Step 1000有安全微增益，修正伪RGB输入 |
| Soft-Critical MICB | 是，待验证 | 直接针对单点Hard Argmax和弱Damaged梯度 |
| N1-shared Cross-Attention | 否 | Gate-miss +2.80 pp、Event worst -0.29 pp |
| 像素Occurrence新头 | 否 | 只有Soft-Critical通过后才允许增加 |
| MIL像素残差广播 | 否 | 避免建筑级证据直接误伤Intact |

## 6. 准入标准

相对P3必须同时满足：Valid不下降、Damaged IoU至少+0.30 pp、Affected Recall
至少+0.50 pp、Damaged gate-miss至少下降0.50 pp、Event mean/worst不下降，
且至少10/14事件不下降。

## 7. 当前状态与边界

目前完成的是代码实现与本地语法验证，不是已经获得提升。PyTorch单元测试、
Exact Guard、Smoke和4000步Validation需在服务器bright5090环境运行。只有Gate
通过并完成额外种子复现后，才讨论一次固定Test。

