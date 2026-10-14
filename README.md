# BRIGHT 实验记录检测器 V1

这是一个 **Python 标准库单文件工具**。它扫描已有 JSON 报告，分类列出区域读出、soft/hard 路由、P/R/T、Memory 建库/诊断、R0/RT/RF、新 soft＋Memory 四组、SAR-Q 四组和 SAR-Q×路由实验。

它不导入 PyTorch，不加载 checkpoint，不访问 BRIGHT 图像，不调用 CUDA，不联网、不启动/停止/恢复训练，也不修改原报告。只读取保存报告不等于重新认证实验；正式类别统一称“报告声明正式 Val450 完成”。部分小型 result.json 哈希绑定和指标算术会复核，模型/输入/预测数组不会重新验证。

## 最简单用法

下载单文件 `BRIGHT_EVIDENCE_AUDIT_V1.py`，上传到：

```text
/root/shared-nvme/BRIGHT-master/New/BRIGHT_EVIDENCE_AUDIT_V1.py
```

执行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python \
  /root/shared-nvme/BRIGHT-master/New/BRIGHT_EVIDENCE_AUDIT_V1.py
```

不需要解压、安装依赖、恢复上一个终端的环境变量，也不需要停止正在运行的实验。

默认读取 `/root/shared-nvme/BRIGHT-checkpoints` 中 SRAC/JOINT 实验目录。每次在以下位置生成新的独立记录：

```text
/root/shared-nvme/BRIGHT-master/New/evidence_records/record_北京时间_进程号/
```

- `SUMMARY.md`：中文简表；每类最多最近6条带指标记录，不挑最佳。
- `REPORT.md`：完整阶段记录、失败信息、指标、同报告差值与原文件路径。
- `metrics.csv`：Excel可读，指标统一为百分比，差值解释使用百分点。
- `evidence.json`：结构化状态、原小报告 SHA256、扫描提醒，便于以后对比。

屏幕会打印实际路径。把 `SUMMARY.md` 的内容发回即可开始分析；若某项“待核对”，同时提供对应 `REPORT.md` 段落。记录数不是独立实验数，训练/结果/流程摘要可能分别列出。

## 只查某次实验

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python \
  /root/shared-nvme/BRIGHT-master/New/BRIGHT_EVIDENCE_AUDIT_V1.py \
  --base /root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_LIVE_RECOVERY_V2/run_20261013_131051_937354_4035676
```

只打印、不保存：追加 `--stdout-only`。默认全量报告较长，通常建议保存文件后读取简表。

## 本工具如何区别证据

| 类型 | 记录方式 |
|---|---|
| 1500步恢复后重新评估 | 完整Val450断点重新验证；不当5000步终点 |
| 2000/2500等中间450图 | 完整Val450中间评估；pending不是自动失败 |
| 原三组/新四组末步 | 同时检查伴随训练summary与final_val、各组5000步、完整性字段及指标一致 |
| SAR训练结束 | 训练及终检完成；单列训练内末次Val；不使用best_valid_miou替代末步 |
| SAR独立Val | 根据完整独立报告；若相邻训练明确失败，降级待核对 |
| SAR四组完成声明 | 独立于baseline；四组跑完也不自动证明SAR有效 |
| Memory label vote | 诊断结果；不是Reader训练效果 |
| 建库完成 | 阶段完成；不是分类成绩 |
| smoke/初始检查/失败 | 单独标记，不提升成正式成绩 |
| Test四类或非Val450 | 单独记录，不进入前景Val对比 |

所有差值只在**同一个报告内部**计算；不会把不同run的baseline与sar_cue拼成增益。单次RT−R0或soft−对照为正，不会自动写成“多种子稳定有效”或“唯一根因已定位”。

本次恢复的CPU汇总后端会标明，不宣称精确复现未中断的原GPU实验。父流水线的旧状态不覆盖子训练的当前步数；父失败和已完成子阶段分别记录。

## 范围与限制

- 无需GPU，但读取报告仍有少量CPU/磁盘开销。
- 默认最多6000个JSON、深度9、每个16 MiB；超限或读取时改变会明确提示，不静默判断成功。
- 不读取大的预测/权重目录；不跟随报告和子目录符号链接。
- 只通过日志最后完整行报告进度，不把文件存在、mtime或PID当作“正在健康运行”的证明。
- 未找到报告只说明扫描范围内未检出；移动目录或新格式需要显式 `--base` 或人工核对。
- 没有服务器实际报告的云端运行不能给出当前服务器实验清单。

## 验证

```bash
python -m unittest discover -s . -p test_audit.py -q
```

测试使用合成报告和用户提供的已知数值，覆盖中间/末步、错误记录、单位、Test隔离、文件写入边界等；不代表BRIGHT模型或CUDA验证。发布结果见 `VALIDATION.json`。
