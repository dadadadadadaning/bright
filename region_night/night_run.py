#!/usr/bin/env python3
"""Serial, fail-closed BRIGHT overnight launcher. Never edits archived sources."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

BASE = "bda_benchmark.experiments."
PRT = BASE + "damageformer_srac_region_head_adapt_v1"
E0 = BASE + "damageformer_srac_hardquery_memory_v1.protocol"
DEFAULT_JOINT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint"
SOURCE_CHECK = '''import importlib,json,sys
from pathlib import Path
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import validate_joint_header,FORMATS
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import code_hash
from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.core import checked_protocol
p=Path(sys.argv[1]); m=json.loads((p/'manifest.json').read_text()); s=json.loads((p/'summary.json').read_text())
validate_joint_header(m,s)
c=importlib.import_module('bda_benchmark.experiments.'+FORMATS[m['format']]+'.common')
if m.get('source_hash') != c.source_hash(): raise ValueError('Joint source changed; restore matching archived code, never weaken gate')
if sys.argv[2]:
    q,r=checked_protocol(sys.argv[2])
    if Path(r['arguments']['joint_run']).resolve()!=p.resolve(): raise ValueError('Explicit E0 belongs to another Joint source')
    if r.get('source_sha256')!=code_hash(): raise ValueError('Explicit E0 executable source differs from current code')
print('CPU source header/source-hash checks passed. Full artifact/runtime checks remain in original E0/PRT loaders.')
'''


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def source_arguments(joint, repo=None):
    joint = Path(joint).resolve()
    manifest, summary = read_json(joint / "manifest.json"), read_json(joint / "summary.json")
    if (manifest.get("format") not in ("SRAC_JOINT123_V1", "SRAC_SNAPSHOT_JOINT123_V1")
            or any(r.get(k) is not False for r in (manifest, summary)
                   for k in ("smoke", "probe", "test_accessed"))
            or summary.get("complete") is not True or summary.get("run_complete") is not True):
        raise ValueError("Need completed formal J1/J2/J3, not probe/smoke/failed source")
    args = manifest["arguments"]
    if (args.get("steps") != 5000 or summary.get("frozen_fingerprints_unchanged") is not True
            or any(summary.get("train_image_counts", {}).get(a) != 5000 for a in ("j1", "j2", "j3"))):
        raise ValueError("Joint source lacks equal completed 5000 updates and immutable evidence")
    values = {"joint_run": str(joint)}
    for key in ("a1_checkpoint", "checkpoint", "cache", "metric_checkpoint", "dataset", "train_list", "val_list"):
        value = args.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError("Missing original Joint argument: " + key)
        path = Path(value)
        path = (Path(repo or Path.cwd()) / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.exists():
            raise FileNotFoundError("Missing original " + key + ": " + str(path))
        values[key] = str(path)
    for name in ("j1_last.pth", "j2_last.pth", "j3_last.pth", "validation.jsonl"):
        if not (joint / name).is_file():
            raise FileNotFoundError(str(joint / name))
    values.update(seed=int(args.get("seed", 3407)), device=str(args.get("device", "cuda:0")),
                  workers=int(args.get("workers", 4)))
    if values["workers"] < 0:
        raise ValueError("Negative original worker count")
    return values


def build_plan(args, root, source):
    py, root = str(args.python), Path(root)
    residual = Path(__file__).with_name("region_residual.py")
    protocol = str(Path(args.protocol).resolve()) if args.protocol else str(root / "fresh_e0" / "protocol.json")
    fit, val = root / "prt_fit", root / "prt_independent_val"
    plan = [("source_check", [py, "-c", SOURCE_CHECK, source["joint_run"], str(args.protocol or "")]),
            ("prt_preflight", ["bash", str(Path(args.repo) / PRT.replace(".", "/") / "run.sh"), "preflight"])]
    if not args.prt_only:
        if not residual.is_file():
            raise FileNotFoundError("Missing companion region_residual.py: " + str(residual))
        plan.append(("residual_preflight", [py, str(residual), "preflight"]))
    if not args.protocol:
        cmd = [py, "-u", "-m", E0]
        for key, value in source.items():
            cmd += ["--" + key, str(value)]
        cmd += ["--output", str(root / "fresh_e0"), "--profile", "legacy", "--pool_mode", "native", "--scope", "all"]
        plan.append(("fresh_e0", cmd))
    plan += [("prt_train", [py, "-u", "-m", PRT + ".train", "--protocol", protocol, "--output", str(fit), "--steps", "5000"]),
             ("prt_evaluate", [py, "-u", "-m", PRT + ".evaluate", "--protocol", protocol, "--fit", str(fit), "--output", str(val)]),
             ("prt_report", [py, "-u", "-m", PRT + ".report", "--output", str(val)])]
    if not args.prt_only:
        fit, val = root / "residual_fit", root / "residual_independent_val"
        plan += [("residual_train", [py, "-u", str(residual), "train", "--protocol", protocol, "--output", str(fit), "--steps", "5000"]),
                 ("residual_evaluate", [py, "-u", str(residual), "evaluate", "--protocol", protocol, "--fit", str(fit), "--output", str(val)]),
                 ("residual_report", [py, "-u", str(residual), "report", "--output", str(val)])]
    return plan, protocol


@contextmanager
def exclusive_lock(parent):
    path = Path(parent) / ".region_night.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another night launcher owns this output parent; no duplicate run: " + str(path)) from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def gpu_selector(device):
    if device != "cuda" and not device.startswith("cuda:"):
        raise ValueError("--wait-gpu-idle requires a CUDA device")
    index = 0 if device == "cuda" else int(device.split(":", 1)[1])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(index)
    ids = [s.strip() for s in visible.split(",")]
    if index < 0 or index >= len(ids) or not ids[index] or ids[index] == "-1":
        raise ValueError("Cannot map CUDA_VISIBLE_DEVICES to target GPU")
    return ids[index]


def wait_gpu_idle(device, timeout=0, query=None, pause=time.sleep, now=time.monotonic):
    selector, started = gpu_selector(device), now()
    def actual_query():
        return subprocess.run(["nvidia-smi", "--id=" + selector, "--query-compute-apps=pid",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              check=True, timeout=15).stdout
    query = query or actual_query
    while True:
        lines = [s.strip() for s in query().splitlines() if s.strip()]
        if any(not s.isdigit() for s in lines):
            raise ValueError("Unrecognized nvidia-smi process output; cannot assume idle: " + repr(lines))
        if not lines:
            print("Target GPU has no compute process; beginning serial experiments.", flush=True)
            return
        print("Waiting for target GPU " + selector + "; existing compute PIDs: " + ", ".join(lines), flush=True)
        if timeout and now() - started >= timeout:
            raise TimeoutError("GPU wait expired; other GPU processes were not interrupted")
        pause(30)


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def run_logged(command, logfile, cwd, env):
    child = None
    with Path(logfile).open("w", encoding="utf-8") as log:
        log.write("Command: " + shlex.join(command) + "\n")
        log.flush()
        try:
            child = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return child.wait()
        finally:
            stop_child(child)
            if child is not None and child.stdout is not None:
                child.stdout.close()


def save_state(root, state):
    temp = Path(root) / ".workflow_state.tmp"
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(Path(root) / "workflow_state.json")


def execute_plan(plan, root, repo, env, state, runner=run_logged, gpu_wait=None):
    root = Path(root)
    for name, command in plan:
        if gpu_wait is not None and name in ("fresh_e0", "prt_train"):
            state.update(current_stage="waiting_gpu")
            save_state(root, state)
            gpu_wait()
            gpu_wait = None
        row = {"name": name, "command": command, "status": "running", "returncode": None,
               "log": str(root / "logs" / (name + ".log"))}
        state["stages"].append(row)
        state.update(current_stage=name)
        save_state(root, state)
        print("\nStarting " + name, flush=True)
        rc = runner(command, row["log"], repo, env)
        row.update(returncode=rc, status="passed" if rc == 0 else "failed")
        save_state(root, state)
        if rc != 0:
            raise RuntimeError(f"Stage {name} failed with returncode {rc}; stopped, preserve logs. No resume or relaxed gates.")
        if name == "fresh_e0":
            state["protocol_sha256"] = digest(state["protocol"])
            save_state(root, state)
    state.update(complete=True, current_stage="complete")
    save_state(root, state)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="/root/shared-nvme/BRIGHT-master")
    p.add_argument("--python", default="/root/shared-nvme/conda-envs/bright4090/bin/python")
    p.add_argument("--joint-run", default=DEFAULT_JOINT)
    p.add_argument("--output-parent", default="/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1")
    p.add_argument("--protocol", help="Explicit passed E0 file/directory; otherwise generate a fresh E0, not restore old SHA")
    p.add_argument("--prt-only", action="store_true")
    p.add_argument("--plan-only", action="store_true", help="Print commands only; no Train/Val/GPU access")
    p.add_argument("--wait-gpu-idle", action="store_true", help="Wait for existing target-GPU compute processes, never stop them")
    p.add_argument("--gpu-wait-timeout", type=int, default=0, help="Seconds; zero waits indefinitely")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    repo, parent = Path(args.repo).resolve(), Path(args.output_parent).resolve()
    args.repo, args.python = str(repo), str(Path(args.python).resolve())
    if args.gpu_wait_timeout < 0:
        raise ValueError("GPU wait timeout must be nonnegative")
    if not repo.is_dir() or not Path(args.python).is_file():
        raise FileNotFoundError("Repository or Python executable does not exist")
    if parent == repo or repo in parent.parents:
        raise ValueError("Keep new experiment output outside the archived repository")
    source = source_arguments(args.joint_run, repo)
    for name in ("joint_run", "cache", "dataset"):
        immutable = Path(source[name])
        if parent == immutable or immutable in parent.parents:
            raise ValueError("Output parent would enter immutable source: " + name)
    if args.protocol:
        protocol = Path(args.protocol).resolve()
        protocol = protocol / "protocol.json" if protocol.is_dir() else protocol
        if not protocol.is_file():
            raise FileNotFoundError(str(protocol))
        args.protocol = str(protocol)
    if args.wait_gpu_idle:
        gpu_selector(source["device"])
    root = parent / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    plan, protocol = build_plan(args, root, source)
    if args.plan_only:
        print("PLAN ONLY: no tests, Train/Val/GPU access, training or output creation. Original strict gates execute in the real run.")
        print("Protocol provenance: " + ("explicit_existing" if args.protocol else "fresh_e0_not_original_protocol_restoration"))
        for name, cmd in plan:
            print(name + ": " + shlex.join(cmd))
        return 0
    parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHON=str(args.python), PYTHONUNBUFFERED="1",
               PYTHONPATH=str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with exclusive_lock(parent):
        root.mkdir(exist_ok=False)
        (root / "logs").mkdir()
        state = dict(format="BRIGHT_REGION_NIGHT_V1", complete=False, root=str(root), source=source,
                     protocol=protocol, protocol_sha256=digest(protocol) if args.protocol else None,
                     protocol_provenance="explicit_existing" if args.protocol else "fresh_e0_not_original_protocol_restoration",
                     launcher_sha256=digest(__file__), test_accessed=False, stages=[], current_stage="initializing")
        save_state(root, state)
        print("OUTPUT_ROOT=" + str(root), flush=True)
        old_handler = signal.getsignal(signal.SIGTERM)
        def interrupt(_signum, _frame):
            raise KeyboardInterrupt("Launcher terminated; only its own child process group is stopped")
        signal.signal(signal.SIGTERM, interrupt)
        try:
            wait = (lambda: wait_gpu_idle(source["device"], args.gpu_wait_timeout)) if args.wait_gpu_idle else None
            execute_plan(plan, root, repo, env, state, gpu_wait=wait)
        except BaseException as error:
            state.update(complete=False, failure=type(error).__name__ + ": " + str(error))
            if state["stages"] and state["stages"][-1]["status"] == "running":
                state["stages"][-1]["status"] = "interrupted"
            save_state(root, state)
            raise
        finally:
            signal.signal(signal.SIGTERM, old_handler)
        print("NIGHT RUN COMPLETE. Formal evidence is determined by each original report; no Test accessed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt as error:
        print(str(error), file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
