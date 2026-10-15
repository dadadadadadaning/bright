"""Small offline launcher; run only the new evaluation, never training."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D"
TOOL = "/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2"
BASE = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_TEST_V2"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-root", default=ROOT)
    p.add_argument("--tool", default=TOOL)
    p.add_argument("--step", type=int, default=3000)
    p.add_argument("--output-parent", default=BASE)
    p.add_argument("--workers", type=int, default=1)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--report", metavar="RESULT_DIRECTORY")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--output", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.workers < 0 or not 1 <= a.step <= 5000:
        p.error("Require nonnegative workers, step in1..5000")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    here = Path(__file__).resolve().parent
    command = [sys.executable, "-u", str(here / "evaluate_test.py")]
    if a.report:
        return subprocess.call(command + ["report", "--output", a.report])
    params = ["--train-root", a.train_root, "--tool", a.tool, "--step", str(a.step),
              "--workers", str(a.workers)]
    if a.plan:
        return subprocess.call(command + ["plan"] + params)
    checkpoint = Path(a.train_root) / "train/checkpoints" / (
        "checkpoint_last.pt" if a.step == 5000 else "checkpoint_%06d.pt" % a.step)
    if not checkpoint.is_file() or not checkpoint.with_suffix(".pt.json").is_file():
        raise SystemExit("指定检查点或其JSON不存在；不会自动改用别的步数：" + str(checkpoint))
    base = Path(a.output_parent).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256((str(checkpoint.resolve())).encode()).hexdigest()[:24]
    lock_path = base / ("active_" + identity + ".lock")
    if a.worker:
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("同一检查点的Test已在运行，本次不重复启动。")
            # Foreground child lifetime is enclosed by the per-checkpoint lock.
            return subprocess.call(command + ["run"] + params + ["--output", a.output])
    # This early check improves feedback; the worker acquires its own lock to
    # cover launch races without relying on stale PIDs.
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("同一检查点的Test已在运行，本次不重复启动。")
    fd, log_name = tempfile.mkstemp(prefix="test_fix1_step_%06d_%s_" % (a.step, datetime.now().strftime("%Y%m%d_%H%M%S")),
                                     suffix=".log", dir=base)
    output = log_name[:-4]
    child = [sys.executable, "-u", str(here / "start.py"), "--worker", *params,
             "--output-parent", str(base), "--output", output]
    with os.fdopen(fd, "w") as log:
        process = subprocess.Popen(child, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=here, start_new_session=True)
    record = dict(pid=process.pid, checkpoint=str(checkpoint), checkpoint_step=a.step,
                  tool=str(here), output=output, log=log_name,
                  mode="exploratory_Test878", version="V2_FIX1", started_is_not_passed=True)
    Path(output + ".launcher.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("已后台启动指定检查点Test；先核验来源，再推理878张。无训练、无重新建库。")
    print("FIX1：从绑定协议恢复完整线程与精度设置；原始数值校验保留。")
    print("本次直接使用GPU；已有任务继续运行，不自动排队或停止其他实验。")
    print("日志：tail -n 40 -f '" + log_name + "'")
    print("报告：" + sys.executable + " '" + str(here / "start.py") + "' --report '" + output + "'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
