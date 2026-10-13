#!/usr/bin/env python3
"""Offline entry points for BRIGHT tools stored together under New."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = "/root/shared-nvme/BRIGHT-master"
CHECKPOINTS = "/root/shared-nvme/BRIGHT-checkpoints"
FAILED_RUN = "SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34"
JOINT_RUN = "SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint"
DIRECTORIES = ("BRIGHT-memory-debug_esfI2YOk", "BRIGHT-prt-sequence_GgXF1Hmw",
               "BRIGHT-prt-debug_s67wQ0Di", "BRIGHT-prt-debug_EDXqo29z",
               "BRIGHT-region-night-unpack_HB4jnNVw")
TOOLS = ("inventory", "memory", "memory-debug", "prt-debug", "prt-sequence", "region-night")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_release(root):
    hashes = json.loads((root / "RELEASE_SHA256.json").read_text(encoding="utf-8"))
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Invalid release checksum manifest")
    for name, expected in hashes.items():
        if not isinstance(name, str):
            raise ValueError("Invalid release checksum path")
        parts = PurePosixPath(name).parts
        if ("\\" in name or not parts
                or PurePosixPath(name).is_absolute() or any(p in (".", "..") for p in parts)
                or name != PurePosixPath(name).as_posix()
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError("Invalid release checksum entry")
        path = root / name
        if any((root.joinpath(*parts[:i])).is_symlink() for i in range(1, len(parts) + 1)):
            raise ValueError("Release entry is a symbolic link: " + name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Release checksum mismatch: " + name)
    return len(hashes)


def build_command(cli, root=ROOT):
    repo, checkpoints = Path(cli.repo).resolve(), Path(cli.checkpoints).resolve()
    tools_root = Path(cli.tools_root).resolve() if cli.tools_root else repo / "New"
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHON=sys.executable,
                       REPO=str(repo), FAILED_RUN=str(checkpoints / FAILED_RUN))
    environment["PYTHONPATH"] = str(repo) + (os.pathsep + environment["PYTHONPATH"]
                                             if environment.get("PYTHONPATH") else "")
    forward = []
    if cli.protocol:
        forward = ["--protocol", str(Path(cli.protocol).resolve())]
    if cli.preflight:
        if cli.run or cli.tool in ("inventory", "region-night", "memory"):
            raise ValueError("Use --preflight only for memory-debug, prt-debug or prt-sequence")
        if cli.tool == "memory-debug":
            return [sys.executable, "-m", "unittest", "discover", "-s",
                    str(root / "memory_replay_debug"), "-p", "test_*.py", "-q"], environment
        folder = "prt_initial_debug" if cli.tool == "prt-debug" else "prt_sequence_experiments"
        return ["bash", str(root / folder / "run.sh"), "--preflight"], environment
    if cli.cpu_only and cli.tool != "memory":
        raise ValueError("--cpu-only is an archive identity action for --tool memory")
    if cli.tool == "inventory":
        if cli.run:
            raise ValueError("Inventory does not have a --run action")
        return [], environment
    if cli.tool == "memory":
        command = [sys.executable, str(root / "RUN_MEMORY_REPLAY.py"), "--repo", str(repo),
                   "--tools-root", str(tools_root), "--checkpoints", str(checkpoints), *forward]
        if cli.evidence:
            command += ["--evidence", str(Path(cli.evidence).resolve())]
        command += ["--run" if cli.run else "--plan-only"]
        if cli.cpu_only:
            command.append("--cpu-only")
        return command, environment
    if cli.evidence:
        raise ValueError("--evidence belongs to --tool memory")
    if cli.tool == "memory-debug":
        command = [sys.executable, str(root / "memory_replay_debug/run_debug.py"), "--repo", str(repo),
                   "--results-base", str(checkpoints / "SRAC_MEMORY_KEY_COMPARE_V1")]
        command += forward or ["--protocol", str(checkpoints / FAILED_RUN / "fresh_e0/protocol.json")]
        if not cli.run:
            command.append("--plan-only")
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            command += ["--output", str(checkpoints / "SRAC_MEMORY_REPLAY_DEBUG_V1" /
                                          ("run_" + stamp + "_" + str(os.getpid())))]
        return command, environment
    if cli.tool == "prt-debug":
        if cli.protocol:
            raise ValueError("PRT debug binds the original failed run; --protocol is not an override")
        environment["OUTPUT_PARENT"] = str(checkpoints / "SRAC_PRT_INIT_DIAGNOSTIC_V1")
        command = ["bash", str(root / "prt_initial_debug/run.sh")]
        if not cli.run:
            command.append("--plan-only")
        return command, environment
    if cli.tool == "prt-sequence":
        if cli.protocol:
            raise ValueError("PRT sequence binds the original failed run")
        command = [sys.executable, str(root / "prt_sequence_experiments/run_sequence.py"),
                   "--repo", str(repo), "--python", sys.executable,
                   "--failed-run", str(checkpoints / FAILED_RUN),
                   "--night-tools", str(root / "region_night"),
                   "--output-parent", str(checkpoints / "SRAC_PRT_SEQUENCE_EXPERIMENTS_V1"),
                   "--checks-only"]
        command.append("--wait-gpu-idle" if cli.run else "--plan-only")
        return command, environment
    command = [sys.executable, str(root / "region_night/night_run.py"), "--repo", str(repo),
               "--python", sys.executable, "--joint-run", str(checkpoints / JOINT_RUN),
               "--output-parent", str(checkpoints / "SRAC_REGION_NIGHT_V1"), *forward]
    command.append("--wait-gpu-idle" if cli.run else "--plan-only")
    return command, environment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tool", choices=TOOLS, default="inventory")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Execute the selected tool; region-night starts its training queue")
    mode.add_argument("--plan-only", action="store_true", help="Print the exact command only; default")
    mode.add_argument("--preflight", action="store_true", help="Run the selected tool's CPU synthetic tests")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--checkpoints", default=CHECKPOINTS)
    p.add_argument("--tools-root")
    p.add_argument("--protocol")
    p.add_argument("--evidence")
    cli = p.parse_args(argv)
    count = verify_release(ROOT)
    command, environment = build_command(cli, ROOT)
    tools_root = Path(cli.tools_root).resolve() if cli.tools_root else Path(cli.repo).resolve() / "New"
    plan = dict(format="BRIGHT_NEW_TOOLS_V1", report_kind="dispatch_plan", package=str(ROOT), tools_root=str(tools_root),
                repo=str(Path(cli.repo).resolve()), checkpoints=str(Path(cli.checkpoints).resolve()),
                checked_release_files=count, tool=cli.tool,
                action="run" if cli.run else ("cpu_preflight" if cli.preflight else "plan_only"),
                execution_pending=bool(command and (cli.run or cli.preflight)),
                offline=True, gpu_ran=False, model_forward_count=0,
                training_requested=bool(cli.run and cli.tool == "region-night"),
                test_accessed=False, command=command,
                original_tool_directories=[dict(path=str(tools_root / name), exists=(tools_root / name).is_dir())
                                           for name in DIRECTORIES],
                zip_files=[str(path) for path in sorted(tools_root.glob("*.zip"))] if tools_root.is_dir() else [])
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if command and (cli.run or cli.preflight):
        return subprocess.call(command, cwd=str(ROOT), env=environment)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("New tools stopped: " + str(error), file=sys.stderr)
        raise SystemExit(1)
