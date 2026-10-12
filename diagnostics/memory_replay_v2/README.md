## 不依赖shell变量的启动方式

`RUN_REPLAY_AUDIT_V2.py` 按本次失败证据的固定SHA查找唯一结果目录，验证真实证据文件内容，再调用同一个已校验的重放探针。启动器无需 KEY_OUTPUT_ROOT、PYTHON 或 E0_PROTOCOL 环境变量。找不到或存在多个匹配就停止，不猜测最近一次运行。原探针的源码与数值协议保持不变。

用服务器bright4090解释器运行该启动器；它会打印定位目录和新诊断summary路径，并在身份检查通过后执行六次同图前向。已有文件不同则保留并停止。查找逻辑已用缺失、唯一匹配、无关目录及重复匹配进行本地验证；没有在云端调用服务器GPU。

# V2 Train 支持特征重放探针

这是独立诊断工具，放在工程目录外运行。它不安装或修改 V1/V2 实验，不更新权重，不重建原库，不放宽门禁，不运行训练、Val 或 Test。使用实际通过的 E0，以及本次失败保存的精确证据文件。

此次失败图为 `bata-explosion_00000000`：所选 T128 共256个元素，即两个128维支持成员。在归档FP16舍入区间与固定 `atol=rtol=2e-5` 之外仍有21个坐标；另有233个存储不确定、2个确定容差内。不能再用纯舍入解释这次失败，也不能据此判断 Memory 方法是否有效。前三张支持源图已完成；失败图的软标签门禁尚未执行。

## 探针核对什么

CPU模式先验证 E0、证据、bank、cache manifest 和对应 Train shard 的 SHA，核对图名、Train索引、区域ID，以及 bank/shard/证据中的归档token是否逐位一致。CPU阶段不初始化CUDA。

GPU模式固定同一份当前 Train pre/post 输入，比较三种条件，各两次前向：

1. native、直接 `inference_mode`：接近旧建库方式。
2. native、直接 `no_grad`。
3. `no_grad` 加 native PoolGuard：接近 V2 方式。

记录 A1 F256、Q/ID、adapted content、main/stats 池化输入、mass、geometry、最终 T128 的重复差异，并比较当前所选区域标签/mass/geometry与旧 shard。运行配置、模型、原始算子与库不修改。Hook、同步及执行顺序可能影响时序，因此条件间差异仍是观察证据，不能单独确定因果。

旧归档缺少历史 Train 图片内容哈希和完整 CUDA/cuDNN/TF32 记录。当前输入哈希只能证明探针用了什么，不能证明历史图片字节相同。Val450 E0通过也不证明每个历史Train token坐标一致。

## 服务器准备

下载脚本到 `/root/shared-nvme/AUDIT_MEMORY_REPLAY_V2.py`，先按附带 SHA256 验证。沿用本次失败的 `KEY_OUTPUT_ROOT`，不要为其重新设置一个空根目录。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export E0_PROTOCOL=/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json
: "${KEY_OUTPUT_ROOT:?请设置刚才失败的V2结果根目录}"
```

先做CPU核对：

```bash
"$PYTHON" /root/shared-nvme/AUDIT_MEMORY_REPLAY_V2.py \
  --protocol "$E0_PROTOCOL" \
  --evidence "$KEY_OUTPUT_ROOT/keys_smoke/replay_evidence/00003.pt" \
  --evidence-sha256 d05ca254bbdff51e0c7cc08c6dbce1aec1fac8b55a4dadbd8f6d0f7506306213 \
  --output "$KEY_OUTPUT_ROOT/replay_identity_$(date +%Y%m%d_%H%M%S)_$$" \
  --cpu-only
```

CPU核对成功、GPU可用后，单独执行六次同图前向：

```bash
"$PYTHON" /root/shared-nvme/AUDIT_MEMORY_REPLAY_V2.py \
  --protocol "$E0_PROTOCOL" \
  --evidence "$KEY_OUTPUT_ROOT/keys_smoke/replay_evidence/00003.pt" \
  --evidence-sha256 d05ca254bbdff51e0c7cc08c6dbce1aec1fac8b55a4dadbd8f6d0f7506306213 \
  --output "$KEY_OUTPUT_ROOT/replay_forward_$(date +%Y%m%d_%H%M%S)_$$"
```

每次输出目录必须是新的，失败目录保留。若CUDA上下文报错，先保留首个错误与CPU证据，在新进程中诊断，不沿用故障进程。

在对应输出目录的 `summary.json` 中查看 `audit_complete` 和 `archive_identity`。`complete` 始终为false，因为本工具只完成诊断，不会将失败建库提升为正式完成。

## 结果怎样解释

- bank/shard身份不一致：先解决归档对应问题。
- 固定输入重复前向仍波动：优先定位当前数值不稳定的首个阶段。
- 重复稳定但持续偏离归档：进一步检查历史输入、环境或执行流程差异。

报告中的诊断完成不等于建库门禁通过，不是正式Val结果。失败V2保持未完成状态；没有任何分支会自动提高容差、跳过成员、替换原库或启动训练。

## 验证边界

本地28项CPU合成测试通过，包含档案篡改拒绝、输入张量不变核验及三种上下文共六次的CPU模拟调度。真实服务器权重、CUDA与BRIGHT图片尚未运行。测试编排读取已安装V2的replay数学；可用 `BRIGHT_REPO` 指向相应工程，默认本地 `/workspace/bright`。生产脚本的CPU-only模式不依赖工程导入。
