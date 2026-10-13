#!/usr/bin/env python3
"""Run a fresh, matched four-arm SAR-Q experiment using the installed Q code.

This orchestrator imports no torch and changes no repository implementation.
Every model job runs in a fresh interpreter after the GPU queue has cleared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback


FORMAT = "BRIGHT_SAR_Q_MATCHED_V1"
ARMS = ("baseline", "rgb_capacity", "shifted_sar", "sar_cue")
MODULE = "bda_benchmark.experiments.damageformer_srac_q_factorial_v1"
DEFAULT_REPO = "/root/shared-nvme/BRIGHT-master"
DEFAULT_PROTOCOL = (
    "/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/"
    "run_20261012_161704_5dd0ec34/fresh_e0/protocol.json"
)
HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def write_json(path, value):
    # Only new experiment metadata is written; original evidence is read-only.
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def protocol_file(value):
    path = Path(value).resolve()
    return path / "protocol.json" if path.is_dir() else path


def _module_command(args, name, *values):
    return [str(args.python), "-u", "-m", MODULE + "." + name, *map(str, values)]


def build_plan(args):
    """Pure command construction: no imports, model loads, or filesystem writes."""
    root = Path(args.output)
    protocol = str(args.protocol)
    repo = Path(args.repo_root)
    stages = [dict(name="preflight", kind="preflight", command=[
        "bash", str(repo / Path(*MODULE.split(".")) / "run.sh"), "preflight"])]
    for smoke in (True, False):
        scope = "smoke" if smoke else "formal"
        steps = 2 if smoke else 2000
        for arm in ARMS:
            fit = root / scope / ("train_" + arm)
            val = root / scope / ("val_" + arm)
            command = _module_command(
                args, "train_q", "--protocol", protocol, "--arm", arm,
                "--output", fit, "--steps", steps, "--eval_every", steps,
                "--batch", 2, "--accum", 8, "--workers", args.workers,
                "--lr", "0.0001", "--aux_weight", "0.1", "--cue_gain", "0.1",
                "--amp", args.amp)
            if smoke:
                command.append("--smoke")
            stages.append(dict(name=f"{scope}_train_{arm}", kind="train", arm=arm,
                               smoke=smoke, command=command, output=str(fit)))
            command = _module_command(args, "evaluate_q", "--protocol", protocol,
                                      "--checkpoint", fit / "last.pth", "--output", val)
            if smoke:
                command.extend(("--limit", "2"))
            stages.append(dict(name=f"{scope}_val_{arm}", kind="evaluate", arm=arm,
                               smoke=smoke, command=command, output=str(val)))
    verification = root / "sar_verification.json"
    command = [str(args.python), "-u", str(HERE / "verify_sar.py"),
               "--repo-root", str(repo), "--output", str(verification)]
    for option, arm in (("baseline", "baseline"), ("sar", "sar_cue"),
                        ("rgb", "rgb_capacity"), ("shifted", "shifted_sar")):
        command.extend(("--" + option, str(root / "formal" / ("val_" + arm))))
    stages.append(dict(name="verify_four_arm_artifacts", kind="verify", command=command,
                       output=str(verification)))
    paired = root / "paired_sar_vs_baseline"
    stages.append(dict(name="paired_sar_vs_baseline", kind="paired", output=str(paired),
                       command=_module_command(
                           args, "paired_prediction", "--baseline", root / "formal/val_baseline",
                           "--candidate", root / "formal/val_sar_cue", "--output", paired)))
    for arm in ("baseline", "sar_cue"):
        output = root / ("q_audit_" + arm)
        stages.append(dict(name="q_audit_" + arm, kind="audit", output=str(output),
                           command=_module_command(args, "audit_saved_q", "--samepass",
                                                   root / "formal" / ("val_" + arm),
                                                   "--output", output)))
    return stages


def _all_true(value):
    return isinstance(value, dict) and bool(value) and all(v is True for v in value.values())


def validate_stage(stage):
    """Small fail-closed gates; final verify_sar also uses original formal gates."""
    kind = stage["kind"]
    if kind == "preflight":
        return {"passed": True, "scope": "CPU contracts and CLI only"}
    output = Path(stage["output"])
    if kind in ("train", "evaluate"):
        path = output / "summary.json"
        data = read_json(path)
        smoke = stage["smoke"]
        if (data.get("complete") is not True or data.get("smoke") is not smoke
                or data.get("test_accessed") is not False or data.get("arm") != stage["arm"]
                or not _all_true(data.get("immutable"))):
            raise ValueError("Incomplete or incompatible stage: " + str(path))
        if kind == "train":
            steps = 2 if smoke else 2000
            if (data.get("format") != "SRAC_Q_FACTORIAL_V1" or data.get("steps") != steps
                    or data.get("a1_unchanged") is not True
                    or data.get("downstream_unchanged") is not True
                    or data.get("full_validation_evidence") is not (not smoke)):
                raise ValueError("Training completion/frozen-model gate failed: " + str(path))
            checkpoint = output / "last.pth"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise ValueError("Completed last.pth missing: " + str(checkpoint))
            return dict(passed=True, summary_sha256=digest(path),
                        checkpoint_sha256=digest(checkpoint), steps=steps,
                        independent_formal_validation=False)
        if (data.get("format") != "SRAC_Q_FACTORIAL_EVAL_V1"
                or data.get("image_count") != (2 if smoke else 450)
                or data.get("checkpoint_steps") != (2 if smoke else 2000)
                or data.get("native_a1_replay_ok") is not True
                or data.get("full_validation_evidence") is not (not smoke)):
            raise ValueError("Independent evaluation gate failed: " + str(path))
        if not smoke and (data.get("reproduction", {}).get("passed") is not True
                          or data.get("q_a1_reproduction", {}).get("passed") is not True):
            raise ValueError("Formal E0/A1 reproduction gate failed: " + str(path))
        return dict(passed=True, summary_sha256=digest(path),
                    formal_report_passed=not smoke,
                    four_arm_artifact_verification_pending=True)
    if kind == "verify":
        data = read_json(output)
        if (data.get("format") != "BRIGHT_EXTERNAL_SAR_VALIDATION_V1"
                or data.get("complete") is not True or data.get("reports_only") is not False
                or data.get("test_accessed") is not False or data.get("training") is not False):
            raise ValueError("Four-arm archived prediction verification failed")
        return dict(passed=True, report_sha256=digest(output))
    if kind in ("paired", "audit"):
        path = output / "report.json"
        data = read_json(path)
        if data.get("complete") is not True or data.get("test_accessed") is not False:
            raise ValueError("Offline diagnosis did not complete: " + str(path))
        return dict(passed=True, report_sha256=digest(path),
                    scope="offline saved-artifact diagnosis, not a new predictor")
    raise ValueError("Unknown stage kind: " + kind)


def snapshot_sources(args, *, include_inputs=True):
    """Record exact local code/protocol. Original workers also verify E0 dependencies."""
    repo = Path(args.repo_root).resolve()
    package = repo / Path(*MODULE.split("."))
    if not package.is_dir():
        raise ValueError("Installed Q factorial package missing: " + str(package))
    paths = [p for p in package.iterdir() if p.is_file()
             and (p.suffix == ".py" or p.name == "run.sh") and not p.name.startswith("test_")]
    paths += [p for p in HERE.iterdir() if p.is_file()
              and p.suffix in (".py", ".sh") and not p.name.startswith("test_")]
    protocol = protocol_file(args.protocol)
    paths.append(protocol)
    if include_inputs:
        record = read_json(protocol)
        for key in ("front", "a1", "train_list", "val_list"):
            source = Path(record["input_paths"][key])
            expected = record["input_fingerprints"][key]
            actual = digest(source)
            if actual != expected:
                raise ValueError("E0 input fingerprint differs: " + key)
            paths.append(source)
    return {str(p.resolve()): digest(p) for p in sorted(set(paths))}


def _validate_protocol(path):
    record = read_json(path)
    if (record.get("format") != "SRAC_HARDQUERY_PROTOCOL_V1"
            or record.get("scope") != "all" or record.get("complete") is not True
            or record.get("smoke") is not False or record.get("image_count") != 450
            or record.get("full_validation_evidence") is not True
            or record.get("reproduction", {}).get("passed") is not True
            or record.get("test_accessed") is not False
            or record.get("profile") != "legacy" or record.get("pool_mode") != "native"):
        raise ValueError("Requires a passed formal legacy/native all-path E0 protocol")
    device = record.get("arguments", {}).get("device")
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("E0 must declare an explicit CUDA device, e.g. cuda:0")
    return record, int(device[5:])


def run_child(stage, repo, env, log):
    print("\nSTAGE=" + stage["name"], flush=True)
    print("COMMAND=" + shlex.join(stage["command"]), flush=True)
    print("STAGE_LOG=" + str(log), flush=True)
    with Path(log).open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(stage["command"], cwd=repo, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def summarize(root):
    """Read partial/failed runs without a resume gate and without loading models."""
    root = Path(root)
    state_path = root / "workflow_state.json"
    state = read_json(state_path) if state_path.is_file() else {}
    report = dict(format=FORMAT, output_root=str(root), complete=state.get("complete") is True,
                  full_validation_evidence=state.get("full_validation_evidence") is True,
                  current_stage=state.get("current_stage"), failure=state.get("failure"),
                  metrics_percentage={}, missing_or_unreadable=[], test_accessed=False)
    for arm in ARMS:
        path = root / "formal" / ("val_" + arm) / "summary.json"
        try:
            data = read_json(path)
            row = dict(report_complete=data.get("complete"), image_count=data.get("image_count"),
                       formal_report=data.get("full_validation_evidence"))
            for readout in ("q_region", "q_grid_probq"):
                values = data.get("metrics", {}).get(readout)
                if values:
                    row[readout] = dict(
                        foreground_miou=100.0 * values["valid_miou"],
                        iou=[None if x is None else 100.0 * x for x in values["iou"]])
            report["metrics_percentage"][arm] = row
        except (OSError, ValueError, TypeError, KeyError) as error:
            report["missing_or_unreadable"].append(dict(path=str(path), error=str(error)))
    verification = root / "sar_verification.json"
    if verification.is_file():
        try:
            verified = read_json(verification)
            report["artifact_verification_complete"] = verified.get("complete")
            report["direct_differences_pp"] = verified.get("direct_differences_pp")
            report["match_errors"] = verified.get("match_errors")
        except (ValueError, OSError) as error:
            report["verification_read_error"] = str(error)
    report["interpretation"] = (
        "Only complete/full_validation_evidence=true certifies this matched pipeline. "
        "Scores are Val foreground classes 1/2/3 mIoU; complete does not mean SAR gain. "
        "No Memory training or historical result reproduction claim.")
    return report


def run(args):
    root = Path(args.output).resolve()
    repo = Path(args.repo_root).resolve()
    if root.exists() or root.is_relative_to(repo) or root.is_relative_to(HERE):
        raise ValueError("Use a fresh checkpoint output outside repository and tool directories")
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    args = argparse.Namespace(**vars(args))
    args.output, args.repo_root = str(root), str(repo)
    args.protocol = str(protocol_file(args.protocol))
    interpreter = shutil.which(str(args.python))
    if interpreter is None:
        raise ValueError("Python interpreter unavailable: " + str(args.python))
    # Preserve the venv/conda entry point: resolving its symlink can select the
    # base interpreter and silently lose the environment's installed packages.
    args.python = os.path.abspath(interpreter)
    protocol, device_index = _validate_protocol(protocol_file(args.protocol))
    bindings = snapshot_sources(args)
    light_bindings = snapshot_sources(args, include_inputs=False)
    for name in ("verify_sar.py", "audit_predictions.py", "gpu_queue.py"):
        if not (HERE / name).is_file():
            raise ValueError("Incomplete external tool package: " + name)
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", PYTHON=str(args.python))
    workspace = protocol["runtime_settings"].get("cublas_workspace_config")
    if workspace is None:
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    elif isinstance(workspace, str):
        env["CUBLAS_WORKSPACE_CONFIG"] = workspace
    else:
        raise ValueError("Invalid E0 cublas_workspace_config")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    plan = build_plan(args)
    manifest = dict(format=FORMAT, arguments=vars(args), stages=plan, source_bindings=bindings,
                    protocol_sha256=digest(protocol_file(args.protocol)),
                    runtime_environment={key: env.get(key) for key in (
                        "CUDA_LAUNCH_BLOCKING", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
                        "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                    source_runtime_settings=protocol["runtime_settings"], test_accessed=False,
                    unchanged_original_code=True, historical_result_reuse=False,
                    training_budget="2000 updates per arm; batch2 accumulation8, random512 crops",
                    metric="Val450 foreground IoU mean of classes 1/2/3; background reported separately")
    write_json(root / "manifest.json", manifest)
    state = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                 current_stage="waiting_gpu", stages=[], test_accessed=False,
                 manifest_sha256=digest(root / "manifest.json"), started_at=time.time())
    write_json(root / "workflow_state.json", state)
    print("OUTPUT_ROOT=" + str(root), flush=True)
    try:
        from gpu_queue import wait_for_idle
        state["gpu_queue"] = wait_for_idle(device_index, log=lambda value: print(value, flush=True))
        write_json(root / "workflow_state.json", state)
        for stage in plan:
            state["current_stage"] = stage["name"]
            row = dict(name=stage["name"], started_at=time.time(), complete=False,
                       log=str(root / "logs" / (stage["name"] + ".log")))
            state["stages"].append(row)
            write_json(root / "workflow_state.json", state)
            current = snapshot_sources(args, include_inputs=False)
            if current != light_bindings:
                raise RuntimeError("Protocol or source changed while queued/running")
            code = run_child(stage, repo, env, row["log"])
            row.update(returncode=code, finished_at=time.time(), log_sha256=digest(row["log"]))
            if code != 0:
                raise RuntimeError(f"Stage {stage['name']} failed with returncode {code}; later stages stopped")
            row["gate"] = validate_stage(stage)
            row["complete"] = True
            write_json(root / "workflow_state.json", state)
        if snapshot_sources(args) != bindings:
            raise RuntimeError("Final code/protocol/source-file integrity failed")
        if digest(root / "manifest.json") != state["manifest_sha256"]:
            raise RuntimeError("Pipeline manifest changed")
        state.update(complete=True, full_validation_evidence=True, current_stage="complete",
                     finished_at=time.time(), immutable=True)
        write_json(root / "workflow_state.json", state)
        result = summarize(root)
        write_json(root / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    except BaseException as error:
        state.update(complete=False, full_validation_evidence=False, finished_at=time.time(),
                     failure=f"{type(error).__name__}: {error}")
        write_json(root / "workflow_state.json", state)
        (root / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(root / "summary.json", summarize(root))
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("plan", "run", "report"))
    result.add_argument("--repo-root", default=DEFAULT_REPO)
    result.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    result.add_argument("--output", "--root", dest="output")
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    result.add_argument("--wait-gpu-idle", action="store_true", default=True,
                        help="Always wait for an idle GPU before model jobs; retained for explicit launch commands")
    return result


def main():
    arguments = parser().parse_args()
    if not arguments.output:
        if arguments.action != "plan":
            raise SystemExit("--output/--root is required for run or report")
        arguments.output = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_Q_MATCHED_V1/NEW_RUN"
    if arguments.action == "plan":
        print(json.dumps(dict(format=FORMAT, stages=build_plan(arguments),
                              no_training_started=True), ensure_ascii=False, indent=2))
    elif arguments.action == "report":
        print(json.dumps(summarize(arguments.output), ensure_ascii=False, indent=2))
    else:
        run(arguments)


if __name__ == "__main__":
    main()
