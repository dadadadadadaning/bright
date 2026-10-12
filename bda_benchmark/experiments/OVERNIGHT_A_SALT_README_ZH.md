# A系列与SALT-D系列整夜顺序调度

`run_overnight_a_salt_v1.sh`只使用一张GPU，禁止访问official test。它会等待当前D0父模型；若D0意外停止且尚未达到50000步，会从其`latest_resume.pth`继续。父模型完成和身份验证后，顺序运行A0-A3各20000步，再把已有SALT D0/D1/D2/D2R从8000步续训到20000步。

按当前实验要求，A0-A3全部使用同一个正式DamageFormer父权重并独立训练20000步，以确保相邻比较具有完全相同的预算。已有5000步probe仍保留为历史筛查，不会被当成20000步结果。

推荐在会话管理器中启动。SSH/Xshell断开不会终止其中的程序：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
mkdir -p /root/data/BRIGHT-train-logs/OVERNIGHT_A_SALT_V1
tmux new-session -d -s bright_a_salt \
  "bash experiments/run_overnight_a_salt_v1.sh > /root/data/BRIGHT-train-logs/OVERNIGHT_A_SALT_V1/master_seed3407.log 2>&1"
```

查看：

```bash
tail -f /root/data/BRIGHT-train-logs/OVERNIGHT_A_SALT_V1/master_seed3407.log
```

查看会话：`tmux ls`。进入实时会话：`tmux attach -t bright_a_salt`。退出显示但保持后台运行：按`Ctrl+B`，松开后按`D`。

只跑A而跳过SALT：`RUN_SALT_SHORT=0`。只续训SALT而跳过A：`RUN_A_CONFIRM=0`。
