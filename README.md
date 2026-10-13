# BRIGHT：原三组联合训练的同步启动器

[下载 ZIP](https://github.com/dadadadadadaning/bright/raw/refs/heads/codex/joint-sync-launcher-v1/bright_joint_sync_launcher_v1.zip) · [使用说明](joint_sync_launcher_v1/README.md)

将代码放在 `/root/shared-nvme/BRIGHT-master/New`。校验五项已通过的诊断后，开启同步执行，调用原三组训练包完成 smoke、各5000步和末步Val450。保留原native池化，不把诊断通过声称为根因解决。启动记录与旧失败目录分开保存。

13项CPU合成测试通过，真实训练需要服务器执行。
