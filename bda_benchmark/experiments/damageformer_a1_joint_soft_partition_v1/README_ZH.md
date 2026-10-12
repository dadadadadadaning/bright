# A1 + Soft Partition 联合训练 20k

这版不是冻结 A1。

- A1 best checkpoint 只作为初始化。
- A1 参数参与反向传播。
- Soft Partition 参数同时参与反向传播。
- A1 LR=2e-5，Partition LR=1e-4。
- 总损失 = location + damage + partition regularization。
- 训练 20000 optimizer steps，完整 Validation 每 500 steps。
- Step 0 会先做一次 A1-equivalent Validation。
- Test 禁止访问。

如果你说的“不是在 A1 基础上”是指**完全不加载 A1 checkpoint、从随机初始化开始**，那是另一项 from-scratch 实验，本包不是那个版本。
