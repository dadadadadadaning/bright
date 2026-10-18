# Q＋双槽＋Memory论文图表设计包

- paper_tables_figures_design.docx/PDF：15页图表手册，含6张结果表、6幅图版、各图表的问题与所需数据。
- paper_tables_blank.xlsx：六表空白数值、自动差值、三种子均值/样本SD、全部事件与运行登记。黄色为待填实测，绿色为待有输入后的计算；—和空白不等于0。
- paper_figures_editable.pptx：6页原生可编辑形状/文字。图1—2为方法示意，图3—6为结果占位版面，不能作为已完成实验直接投稿。
- figures/：同图PNG/PDF/SVG，支持排版与进一步修改。没有生成假预测、假曲线或假热力图，没有修改用户原论文图片。
- captions_and_text.md：完整表题、图注与结果分析句型，兼容无收益和负结果。
- experiment_contract.md：比较条件、复用、混合度/边缘错率、支持标签混杂与统计规则。
- experiment_registry.csv：每种配置的全称、复用表号和身份字段。正文不再仅用A/B/C/D编号。

主表使用标准Test四类mIoU；机制表使用同一Val前景三类mIoU，所有配置保留完整混淆矩阵。支持内容重训与固定权重干预单列。具体数值须来自相同协议的真实结果。
