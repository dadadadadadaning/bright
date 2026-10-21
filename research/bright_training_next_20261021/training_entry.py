#!/usr/bin/env python3
"""Fixed-parent six-arm continuation and separate fresh D/H training."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from types import SimpleNamespace

import resume_train
from resume_context import prepare as original_prepare, inactive_parent
from probe import json_value
from training_common import certificate, observe_bindings, DEFAULT_ROOT
from evaluate_checkpoint import write_json
from grid_probe import read_json, sha256

PAIR_BASE = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_D_H_NO_SUPPORT_READ_V1"


def prepared(args):
    print("PREFLIGHT: checking original source, checkpoint1500 and completed Val450 evidence", flush=True)
    context = certificate(original_prepare(args))
    # Include the pinned extension itself in the immutable worker identity.
    artifact = Path(sys.argv[0]).resolve()
    if artifact.is_file() and artifact.suffix == ".pyz":
        # Stable digest rather than launch path: the worker uses its own copy.
        context["extension_sha256"] = sha256(artifact)
        if getattr(args, "command", None) == "worker":
            request = read_json(Path(args.output) / "request.json")
            if context["extension_sha256"] != request["trainer_sha256"]:
                raise ValueError("Pinned continuation extension differs from launch request")
    return context


def observed_run(context, output):
    artifact = Path(sys.argv[0]).resolve()
    if artifact.suffix == ".pyz":
        context["evidence_files"][str(artifact)] = context["extension_sha256"]
    write_json(Path(output) / "training_extension.json", dict(
        format="BRIGHT_CONTINUATION_OBSERVER_V1", checkpoint_step=1500,
        endpoint_step=2500, successful_prior_evaluation=context["standalone_evaluation"],
        extension_sha256=context.get("extension_sha256"),
        loss_changed=False, probability_validator_changed=False,
        original_training_complete=False, checkpoint_saves_every=50,
        initial_validation_repeated=True, test_accessed=False))
    with observe_bindings(context, output):
        return ORIGINAL_RUN(context, output)


ORIGINAL_RUN = resume_train.run_training
ORIGINAL_BINDING = resume_train.imported_file


def described_binding(function):
    original = inspect.unwrap(function)
    description = ORIGINAL_BINDING(original)
    if original is not function:
        description["observation_wrapper"] = dict(module="training_common",
            artifact=str(Path(sys.argv[0]).resolve()), artifact_sha256=sha256(sys.argv[0]),
            probability_validator_changed=False)
    return description


def resume(command, remaining):
    if command in ("resume", "worker") and "--parent-root" not in remaining:
        remaining += ["--parent-root", DEFAULT_ROOT]
    resume_train.prepare = prepared
    resume_train.run_training = observed_run
    resume_train.imported_file = described_binding
    sys.argv[1:] = ["start" if command == "resume" else command, *remaining]
    return resume_train.main()


def pair_context():
    return prepared(SimpleNamespace(original_run=resume_train.DEFAULT_RUN, seed=3407,
                                   checkpoint_every=50, addon_dir=None, parent_root=DEFAULT_ROOT))


def pair_identity(context):
    return dict(original=resume_train.request_identity(context),
                extension_sha256=context.get("extension_sha256"),
                initialization="fresh seed3407, no old head/optimizer weights")


def pair_launch():
    context = pair_context()
    base = Path(PAIR_BASE)
    base.mkdir(parents=True, exist_ok=True)
    # Both experiments share the continuation lease, preventing concurrent starts.
    with (Path(resume_train.DEFAULT_OUTPUT) / "active.lock").open("a+") as shared:
        fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (base / "active.lock").open("a+") as own:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            artifact = Path(sys.argv[0]).resolve()
            if not artifact.is_file() or artifact.suffix != ".pyz":
                raise ValueError("Start with the standalone .pyz artifact")
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            output = base / ("pair_%s_%s" % (stamp, os.getpid()))
            output.mkdir()
            pinned = output / "trainer.pyz"
            shutil.copyfile(artifact, pinned)
            write_json(output / "request.json", dict(identity=pair_identity(context),
                       trainer_sha256=sha256(pinned)))
            write_json(output / "summary.json", dict(phase="queued", complete=False,
                       format="BRIGHT_D_H_NO_SUPPORT_READ_V1", output=str(output)))
            log = output / "train.log"
            command = [sys.executable, "-u", str(pinned), "baseline-worker", "--output", str(output)]
            with log.open("x") as stream:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(shared.fileno(), own.fileno()),
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                             BRIGHT_PAIR_FDS="%s,%s" % (shared.fileno(), own.fileno())))
            write_json(output / "launcher.json", dict(pid=process.pid, command=command, log=str(log)))
            (base / "latest.txt").write_text(str(output) + "\n")
            print("D/H 从相同初始化训练 0→2500；后台启动不代表完成。", flush=True)
            print("PID=%s\n结果目录=%s\n日志=%s" % (process.pid, output, log), flush=True)
            print('查看日志：%s %s baseline-logs' % (sys.executable, pinned), flush=True)
    return 0


def pair_worker(output):
    from no_read_baseline import run_pair
    output = Path(output).resolve()
    if output.parent != Path(PAIR_BASE) or not output.name.startswith("pair_"):
        raise ValueError("Require a newly allocated D/H directory")
    try:
        with contextlib.ExitStack() as stack:
            raw = os.environ.pop("BRIGHT_PAIR_FDS", None)
            if raw:
                for fd in raw.split(","):
                    stream = stack.enter_context(os.fdopen(int(fd), "a+"))
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                for folder in (Path(resume_train.DEFAULT_OUTPUT), Path(PAIR_BASE)):
                    stream = stack.enter_context((folder / "active.lock").open("a+"))
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            own = stack.enter_context((output / "worker.lock").open("a+"))
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            context = pair_context()
            request = read_json(output / "request.json")
            if (pair_identity(context) != request["identity"]
                    or sha256(output / "trainer.pyz") != request["trainer_sha256"]):
                raise ValueError("D/H source or artifact changed after launcher preflight")
            stack.enter_context(inactive_parent(DEFAULT_ROOT))
            context["evidence_files"][str(output / "trainer.pyz")] = request["trainer_sha256"]
            resume_train.verify_files(context["evidence_files"])
            return run_pair(context, output)
    except BaseException as exc:
        write_json(output / "summary.json", dict(format="BRIGHT_D_H_NO_SUPPORT_READ_V1",
                   complete=False, phase="failed", failure_phase="worker_preflight",
                   error=type(exc).__name__ + ": " + str(exc), output=str(output)))
        (output / "traceback.txt").write_text(traceback.format_exc())
        raise


def latest(base):
    root = Path((Path(base) / "latest.txt").read_text().strip()).resolve()
    if root.parent != Path(base).resolve():
        raise ValueError("Unexpected latest output location")
    return root


def main():
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("resume", "worker", "status", "logs",
        "baseline", "baseline-worker", "baseline-status", "baseline-logs", "check-evaluation"))
    args, rest = parser.parse_known_args()
    try:
        if args.command == "baseline-worker":
            worker_parser = argparse.ArgumentParser()
            worker_parser.add_argument("--output", required=True)
            return pair_worker(worker_parser.parse_args(rest).output)
        if args.command in ("logs", "baseline-logs", "baseline-status", "baseline") and rest:
            parser.error("This command takes no additional arguments")
        if args.command == "baseline":
            return pair_launch()
        if args.command in ("logs", "baseline-logs", "baseline-status"):
            base = PAIR_BASE if args.command.startswith("baseline") else resume_train.DEFAULT_OUTPUT
            root = latest(base)
            if args.command.endswith("logs"):
                print("当前日志：" + str(root / "train.log"), flush=True)
                os.execvp("tail", ["tail", "-n", "60", "-F", str(root / "train.log")])
            return resume_train.show_status(SimpleNamespace(root=str(root), output_base=base))
        return resume(args.command, rest)
    except BlockingIOError:
        print("已有续训或 D/H 进程持有运行锁；本次没有重复启动。", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print("STOP: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
