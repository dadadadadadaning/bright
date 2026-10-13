"""Explicit, resumable V3 new-bank pipeline; no shell result variables required.

Plan and report are read-only. Smoke/full use a new result root, preserve an
artifact ledger and stop on evidence failures even if a child exits with zero.
This pipeline has no training stage and never invokes historical token replay.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

PACKAGE = "bda_benchmark.experiments.damageformer_srac_memory_rebuild_v3"
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
FORMAT = "SRAC_MEMORY_REBUILD_PIPELINE_V3"
BANK_FORMAT = "SRAC_MEMORY_REBUILD_BANK_V3"
DIAG_FORMAT = "SRAC_MEMORY_REBUILD_DIAGNOSE_V3"
PROTOCOL_FORMAT = "SRAC_MEMORY_REBUILD_PROTOCOL_V3"
DEFAULT_PROTOCOL = Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json")
DEFAULT_OUTPUT_PARENT = Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3")
CHECKS = ("protocol", "bank", "split", "labels", "reference", "native", "pool", "artifacts", "immutable")
SMOKE_STAGES = ("keys_smoke", "diagnose_smoke", "report_smoke")
FULL_STAGES = ("keys", "diagnose", "report")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "plan", "smoke", "full", "report"), default="plan")
    p.add_argument("--root", type=Path, help="Explicit result root; existing roots accepted only for verified smoke-to-full continuation")
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    p.add_argument("--wait-gpu-idle", action="store_true")
    p.add_argument("--gpu-wait-timeout", type=float, default=0, help="Seconds; 0 waits without a deadline only when --wait-gpu-idle is explicit")
    return p


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def code_hash():
    from .shared import source_hash
    return source_hash()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def write_state(root, state):
    # Ordinary replace within this newly-owned directory; no renameat2 flags.
    temporary = root / (".pipeline_state_" + str(os.getpid()) + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(root / "pipeline_state.json")


def proposed_root(cli):
    if cli.root is not None:
        return cli.root.resolve()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return cli.output_parent.resolve() / ("run_" + stamp + "_" + str(os.getpid()))


def command(stage, root, protocol):
    base = [sys.executable, "-m"]
    if stage.startswith("keys"):
        result = base + [PACKAGE + ".build_keys", "--protocol", str(protocol), "--output", str(root / stage)]
        return result + (["--limit-images", "4"] if stage.endswith("_smoke") else [])
    if stage.startswith("diagnose"):
        keys = "keys_smoke" if stage.endswith("_smoke") else "keys"
        result = base + [PACKAGE + ".diagnose", "--protocol", str(protocol), "--keys", str(root / keys), "--output", str(root / stage)]
        return result + (["--limit", "2"] if stage.endswith("_smoke") else [])
    diagnosis = "diagnose_smoke" if stage == "report_smoke" else "diagnose"
    return base + [PACKAGE + ".report", "--input", str(root / diagnosis)]


def stage_artifacts(root, stage):
    folder = root / stage
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Missing or symbolic stage directory: " + str(folder))
    ledger = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symbolic artifact is refused: " + str(path))
        if path.is_file():
            ledger[path.relative_to(root).as_posix()] = digest(path)
    if not ledger:
        raise ValueError("Stage produced no saved evidence: " + stage)
    return ledger


def all_true(value):
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def stage_gate(root, stage, state):
    smoke = stage.endswith("_smoke")
    folder = root / stage
    summary = read_json(folder / "summary.json")
    if summary.get("training") is not False or summary.get("test_accessed") is not False:
        raise ValueError("Stage lacks explicit no-training/no-Test scope: " + stage)
    if summary.get("smoke") is not smoke or not all_true(summary.get("immutable")):
        raise ValueError("Stage smoke/immutability gate failed: " + stage)
    if stage.startswith("keys"):
        manifest = read_json(folder / "manifest.json")
        protocol = read_json(folder / "protocol.json")
        if (summary.get("format") != BANK_FORMAT or summary.get("complete") is not True
                or manifest.get("format") != BANK_FORMAT or manifest.get("complete") is not True
                or manifest.get("smoke") is not smoke or manifest.get("train_only") is not True
                or manifest.get("training") is not False or manifest.get("test_accessed") is not False
                or not all_true(manifest.get("immutable"))):
            raise ValueError("Bank summary/manifest completion gate failed: " + stage)
        for name in ("fresh_label_checks_passed", "same_pass_q_checks_passed", "saved_fp32_roundtrip_exact", "pool_ok"):
            if manifest.get(name) is not True:
                raise ValueError("Bank evidence gate failed: " + name)
        for name in ("t128_storage", "f256_storage", "label_storage"):
            if manifest.get(name) != "torch.float32":
                raise ValueError("Bank must store actual FP32 paired keys/labels: " + name)
        count = summary.get("image_count")
        if (type(count) is not int or count <= 0 or (smoke and count != 4)
                or count != manifest.get("image_count")
                or count != summary.get("extraction_forward_count")
                or count != manifest.get("extraction_forward_count")):
            raise ValueError("Bank same-pass/image-count evidence failed")
        for name, expected in (("source_sha256", state["source_sha256"]), ("protocol_sha256", state["protocol_sha256"])):
            if manifest.get(name) != expected or protocol.get(name) != expected:
                raise ValueError("Bank source binding failed: " + name)
        if (protocol.get("format") != PROTOCOL_FORMAT or protocol.get("historical_replay_required") is not False
                or protocol.get("historical_fp32_reproduction_claim") is not False
                or manifest.get("rebuild_protocol_sha256") != digest(folder / "protocol.json")
                or manifest.get("paired_bank_sha256") != digest(folder / "paired_bank.pt")
                or summary.get("paired_bank_sha256") != manifest["paired_bank_sha256"]
                or summary.get("rebuild_protocol_sha256") != manifest["rebuild_protocol_sha256"]):
            raise ValueError("Independent new-bank protocol/checksum gate failed")
        return {"complete": True, "smoke": smoke, "image_count": count}
    result = read_json(folder / "result.json")
    if summary.get("format") != DIAG_FORMAT or result.get("format") != DIAG_FORMAT:
        raise ValueError("Unexpected diagnostic format")
    for name in ("image_count", "complete", "full_validation_evidence", "execution_complete", "smoke", "validation"):
        if summary.get(name) != result.get(name):
            raise ValueError("Diagnostic summary/result disagree: " + name)
    if (result.get("execution_complete") is not True or result.get("training") is not False
            or result.get("test_accessed") is not False or result.get("historical_reproduction_claim") is not False
            or result.get("labelvote_is_untrained_diagnostic") is not True
            or not all_true(result.get("immutable"))
            or not isinstance(result.get("validation"), dict)
            or not all(result["validation"].get(name) is True for name in CHECKS)):
        raise ValueError("Diagnostic evidence gates failed")
    expected_count = 2 if smoke else 450
    if (result.get("image_count") != expected_count or result.get("paired_bank_smoke") is not smoke
            or result.get("complete") is not (not smoke)
            or result.get("full_validation_evidence") is not (not smoke)):
        raise ValueError("Smoke/Val450 scope gate failed")
    keys = root / ("keys_smoke" if smoke else "keys")
    for name, expected in (("source_sha256", state["source_sha256"]),
            ("source_anchor_protocol_sha256", state["protocol_sha256"]),
            ("paired_bank_sha256", digest(keys / "paired_bank.pt")),
            ("paired_manifest_sha256", digest(keys / "manifest.json")),
            ("rebuild_protocol_sha256", digest(keys / "protocol.json"))):
        if result.get(name) != expected:
            raise ValueError("Diagnostic provenance binding failed: " + name)
    return {"execution_complete": True, "complete": not smoke, "full_validation_evidence": not smoke,
            "smoke": smoke, "image_count": expected_count}


def verify_ledger(root, state):
    if (state.get("format") != FORMAT or state.get("training") is not False
            or state.get("test_accessed") is not False or not isinstance(state.get("stages"), dict)):
        raise ValueError("Invalid V3 pipeline state")
    if digest(state["protocol"]) != state.get("protocol_sha256") or code_hash() != state.get("source_sha256"):
        raise ValueError("Protocol/source code changed since pipeline creation")
    order = SMOKE_STAGES + FULL_STAGES
    if set(state["stages"]) != set(order[:len(state["stages"])]):
        raise ValueError("Stage ledger is not a valid pipeline prefix")
    for stage in order[:len(state["stages"])]:
        row = state["stages"][stage]
        if stage not in SMOKE_STAGES + FULL_STAGES or row.get("status") != "complete" or row.get("returncode") != 0:
            raise ValueError("Partial or failed pipeline cannot resume: " + stage)
        if row.get("command") != command(stage, root, Path(state["protocol"])):
            raise ValueError("Recorded stage command differs: " + stage)
        expected_log = "logs/" + stage + ".log"
        if row.get("log") != expected_log or digest(root / expected_log) != row.get("log_sha256"):
            raise ValueError("Stage log changed: " + stage)
        if stage.startswith("report"):
            if row.get("artifacts") != {} or row.get("gate") != {"report_complete": True}:
                raise ValueError("Report ledger has unexpected artifacts")
        else:
            if stage_artifacts(root, stage) != row.get("artifacts"):
                raise ValueError("Saved stage artifacts changed: " + stage)
            if stage_gate(root, stage, state) != row.get("gate"):
                raise ValueError("Saved stage gates changed: " + stage)


def gpu_identifier(record):
    device = record.get("arguments", {}).get("device")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError("Source protocol must record a CUDA device")
    parts = device.split(":")
    if len(parts) > 2 or (len(parts) == 2 and not parts[1].isdigit()):
        raise ValueError("Invalid source CUDA device: " + device)
    index = int(parts[1]) if len(parts) == 2 else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        members = [item.strip() for item in visible.split(",")]
        if index >= len(members) or not members[index] or members[index] == "-1":
            raise ValueError("Source CUDA device is unavailable under CUDA_VISIBLE_DEVICES")
        return members[index]
    return str(index)


def gpu_pids(identifier):
    result = subprocess.run(["nvidia-smi", "--id=" + identifier,
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=30, check=False)
    if result.returncode != 0:
        raise RuntimeError("Cannot check source GPU availability: " + result.stderr.strip())
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not line.isdigit() for line in lines):
        raise RuntimeError("Unexpected GPU process listing: " + repr(result.stdout))
    return sorted(set(int(line) for line in lines))


def ensure_idle(identifier, cli):
    started, last_notice = time.monotonic(), None
    while True:
        pids = gpu_pids(identifier)
        if not pids:
            return
        if not cli.wait_gpu_idle:
            raise RuntimeError("Source GPU " + identifier + " is busy; PIDs " + str(pids)
                               + ". Retry when idle or explicitly add --wait-gpu-idle.")
        elapsed = time.monotonic() - started
        if cli.gpu_wait_timeout and elapsed >= cli.gpu_wait_timeout:
            raise TimeoutError("Timed out waiting for source GPU " + identifier)
        if last_notice is None or elapsed - last_notice >= 30:
            print("Waiting for source GPU " + identifier + "; existing PIDs " + str(pids)
                  + "; elapsed " + str(round(elapsed)) + "s", flush=True)
            last_notice = elapsed
        time.sleep(min(5, max(.01, cli.gpu_wait_timeout - elapsed)) if cli.gpu_wait_timeout else 5)


def run_child(argv, log):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with log.open("x", encoding="utf-8") as stream:
        stream.write("Command: " + shlex.join(argv) + "\n")
        stream.flush()
        with subprocess.Popen(argv, cwd=REPO, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = child.wait()
        stream.write("Return code: " + str(code) + "\n")
    return code


def preflight():
    commands = [[sys.executable, "-m", "unittest", "discover", "-s", str(ROOT), "-p", "test_*.py", "-t", str(REPO)]]
    commands += [[sys.executable, "-m", PACKAGE + "." + module, "--help"]
                 for module in ("build_keys", "diagnose", "report", "run_pipeline")]
    for argv in commands:
        print(shlex.join(argv), flush=True)
        result = subprocess.run(argv, cwd=REPO, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=False)
        if result.returncode:
            raise RuntimeError("CPU preflight failed: " + shlex.join(argv))
    print("CPU PREFLIGHT COMPLETE. No source E0, data, CUDA forward, training or Test required.", flush=True)


def new_or_resume(cli, root):
    protocol = cli.protocol.resolve()
    record = read_json(protocol)
    if (record.get("format") != "SRAC_HARDQUERY_PROTOCOL_V1" or record.get("complete") is not True
            or record.get("full_validation_evidence") is not True or record.get("image_count") != 450
            or record.get("smoke") is not False or record.get("test_accessed") is not False
            or record.get("training") is not False):
        raise ValueError("Require a passed, full Val450 E0 frozen-source protocol")
    fingerprint, own = digest(protocol), code_hash()
    gpu = gpu_identifier(record)
    if root.exists():
        if root.is_symlink() or cli.stage != "full" or cli.root is None:
            raise ValueError("Existing result root is refused; only explicit verified smoke-to-full continuation is accepted")
        state = read_json(root / "pipeline_state.json")
        if (state.get("protocol") != str(protocol) or state.get("protocol_sha256") != fingerprint
                or state.get("source_sha256") != own or state.get("root") != str(root)
                or state.get("gpu_identifier") != gpu or state.get("smoke_complete") is not True
                or state.get("complete") is not False or state.get("error") is not None
                or set(state.get("stages", {})) != set(SMOKE_STAGES)):
            raise ValueError("Existing root lacks exact completed smoke provenance")
        verify_ledger(root, state)
        if any((root / name).exists() or (root / "logs" / (name + ".log")).exists() for name in FULL_STAGES):
            raise ValueError("Formal outputs/logs already exist; never overwrite a partial or completed stage")
        ensure_idle(gpu, cli)
        state["resume_arguments"] = {key: str(value) if isinstance(value, Path) else value
                                     for key, value in vars(cli).items()}
        return state, FULL_STAGES
    ensure_idle(gpu, cli)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    state = dict(format=FORMAT, root=str(root), protocol=str(protocol), protocol_sha256=fingerprint,
                 source_sha256=own, gpu_identifier=gpu, arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(cli).items()},
                 training=False, test_accessed=False, historical_replay_required=False,
                 historical_reproduction_claim=False, smoke_complete=False, complete=False,
                 full_validation_evidence=False, current=None, error=None, stages={})
    write_state(root, state)
    return state, SMOKE_STAGES + (FULL_STAGES if cli.stage == "full" else ())


def run_pipeline(cli):
    if not math.isfinite(cli.gpu_wait_timeout) or cli.gpu_wait_timeout < 0:
        raise ValueError("--gpu-wait-timeout must be finite and nonnegative")
    if cli.stage == "preflight":
        preflight()
        return
    root = proposed_root(cli)
    if cli.stage == "plan":
        print("Read-only plan; proposed result root: " + str(root))
        for stage in SMOKE_STAGES + FULL_STAGES:
            print(shlex.join(command(stage, root, cli.protocol.resolve())))
        print("Source device comes from E0; no KEY_OUTPUT_ROOT/DIAG_OUTPUT/DEVICE shell variable is used.")
        print("No directories created, GPU queried, model run, training or Test access.")
        return
    if cli.stage == "report":
        if cli.root is None:
            raise ValueError("Read-only report requires --root with an explicit V3 result directory")
        state = read_json(root / "pipeline_state.json")
        verify_ledger(root, state)
        target = "diagnose" if "diagnose" in state["stages"] else "diagnose_smoke"
        if target not in state["stages"]:
            raise ValueError("No completed V3 diagnosis is recorded in this root")
        from .report import render
        print(render(root / target))
        return
    if cli.root is not None and cli.root.is_symlink():
        raise ValueError("Symbolic result roots are refused")
    state, stages = new_or_resume(cli, root)
    print("OUTPUT_ROOT=" + str(root), flush=True)
    try:
        for stage in stages:
            verify_ledger(root, state)
            if not stage.startswith("report"):
                ensure_idle(state["gpu_identifier"], cli)
            state["current"] = stage
            write_state(root, state)
            argv = command(stage, root, Path(state["protocol"]))
            log = root / "logs" / (stage + ".log")
            row = dict(status="running", command=argv, log=log.relative_to(root).as_posix(), artifacts={})
            state["stages"][stage] = row
            write_state(root, state)
            code = run_child(argv, log)
            row.update(returncode=code, log_sha256=digest(log))
            if code:
                raise RuntimeError("Stage " + stage + " exited " + str(code) + "; later stages blocked")
            if stage.startswith("report"):
                row["gate"] = {"report_complete": True}
            else:
                row["gate"] = stage_gate(root, stage, state)
                row["artifacts"] = stage_artifacts(root, stage)
            row["status"] = "complete"
            state["smoke_complete"] = all(state["stages"].get(name, {}).get("status") == "complete" for name in SMOKE_STAGES)
            if stage == "report":
                state.update(complete=True, full_validation_evidence=True)
            write_state(root, state)
        verify_ledger(root, state)
        state["current"] = "complete" if state["complete"] else "complete_smoke"
        write_state(root, state)
        print("COMPLETE " + cli.stage + ". New FP32 paired Memory evidence only. No training. No Test.", flush=True)
    except Exception as error:
        state.update(error=type(error).__name__ + ": " + str(error), complete=False, full_validation_evidence=False)
        current = state.get("current")
        if current in state["stages"] and state["stages"][current].get("status") != "complete":
            failed_row = state["stages"][current]
            failed_row["status"] = "failed"
            failed_row.setdefault("returncode", None)
            failed_log = root / failed_row["log"]
            if failed_log.is_file():
                failed_row["log_sha256"] = digest(failed_log)
        write_state(root, state)
        (root / "pipeline_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main(argv=None):
    run_pipeline(parser().parse_args(argv))


if __name__ == "__main__":
    main()
