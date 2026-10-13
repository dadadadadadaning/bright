# BRIGHT Memory联合训练：独立候选修复和1500步后继续训练

代码与说明：[joint_pool_recovery_v1/README.md](joint_pool_recovery_v1/README.md)。

下载：bright_joint_pool_recovery_v1.zip。仅新增独立工具目录；保留原模型、库、失败运行和SAR队列。候选GPU修复尚需服务器验证。

ZIP SHA256：`34018a9be0a52a144667cf55eed0fb8fb0c3f5e046971136c695228041c09041`

60项CPU测试与独立CPU重放通过。新入口先等待指定SAR四组与SAR＋soft完成，再验证候选、重评1500步权重，并在新目录继续原三组预算到5000。新证据明确标记修复后继续训练，不声称精确历史复现。
