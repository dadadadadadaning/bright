#!/usr/bin/env python3
"""Inspect one ablation run using stdlib only; print evidence, never alter it."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

FORMAT = "BRIGHT_SLOT_MEMORY_ABLATION_V1"
DEFAULT_RUN = (
    "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_V1/"
    "run_20261020_025710_3840142"
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def validate_identity(root, seed):
    root = Path(root)
    plan = read_object(root / "plan.json")
    manifest = read_object(root / ("seed_%s/train/manifest.json" % seed))
    if plan.get("format") != FORMAT or manifest.get("format") != FORMAT:
        raise ValueError("Expected slot-memory ablation; Extended/source runs are not accepted")
    arms = plan.get("arms")
    if (not isinstance(arms, list) or not arms
            or not all(isinstance(a, str) and a in set("efabcdrln") for a in arms)
            or len(arms) != len(set(arms))):
        raise ValueError("Invalid or duplicate arm IDs")
    seeds = plan.get("seeds")
    if (not isinstance(seeds, list) or seed not in seeds
            or type(manifest.get("seed")) is not int or manifest["seed"] != seed):
        raise ValueError("Seed does not match plan and training manifest")
    if manifest.get("plan") != plan:
        raise ValueError("Training manifest plan differs from root plan, including source/arms")
    return plan, manifest


def read_journal(path, arms):
    """A complete record requires newline, valid identity and all optimizer steps."""
    path = Path(path)
    result = dict(records=0, last_complete=None, contiguous_prefix_step=0,
                  issues=[], issue_count=0, trailing_incomplete=False,
                  next_unjournaled_step=None, stable_during_read=None)

    def issue(line, reason):
        result["issue_count"] += 1
        if len(result["issues"]) < 8:
            result["issues"].append(dict(line=line, reason=reason))

    if not path.is_file():
        issue(None, "train.jsonl missing")
        return result
    before = path.stat()
    prefix_valid = True
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n"):
                result["trailing_incomplete"] = True
                break
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("row is not an object")
                result["records"] += 1
                step = row.get("step")
                if type(step) is not int or step < 1:
                    raise ValueError("invalid step")
                if not isinstance(row.get("image"), str) or not row["image"]:
                    raise ValueError("missing image")
                updates = row.get("arms")
                if not isinstance(updates, dict) or set(updates) != set(arms):
                    raise ValueError("arm set differs")
                for arm in arms:
                    data = updates[arm]
                    if (not isinstance(data, dict)
                            or type(data.get("optimizer_step")) is not int
                            or data["optimizer_step"] != step):
                        raise ValueError("optimizer_step differs: " + arm)
                result["last_complete"] = dict(
                    step=step, image=row["image"], event=row.get("event"),
                    epoch=row.get("epoch"), epoch_seed=row.get("epoch_seed"),
                    optimizer_steps={a: updates[a]["optimizer_step"] for a in arms},
                )
                if step != result["contiguous_prefix_step"] + 1 or not prefix_valid:
                    raise ValueError("record is outside the verified contiguous prefix")
                result["contiguous_prefix_step"] = step
            except (ValueError, UnicodeError) as exc:
                issue(number, str(exc))
                prefix_valid = False
    after = path.stat()
    result["stable_during_read"] = (
        before.st_ino, before.st_size, before.st_mtime_ns
    ) == (after.st_ino, after.st_size, after.st_mtime_ns)
    if not result["stable_during_read"]:
        issue(None, "journal changed during read; no stable final-step conclusion")
    if result["issue_count"] == 0:
        result["next_unjournaled_step"] = result["contiguous_prefix_step"] + 1
    return result


def checkpoint_info(folder, arms):
    records = []
    for path in sorted(Path(folder).glob("checkpoint_*.pt")):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match is None:
            continue
        step = int(match.group(1))
        item = dict(name=path.name, step=step, bytes=path.stat().st_size,
                    sha256=sha256(path), sidecar_sha256_match=None, verified=False)
        sidecar = path.with_suffix(".pt.json")
        if not sidecar.is_file():
            item["error"] = "sidecar missing"
        else:
            try:
                metadata = read_object(sidecar)
                item["sidecar_sha256_match"] = metadata.get("sha256") == item["sha256"]
                hashes = metadata.get("head_hashes")
                item["verified"] = (
                    item["bytes"] > 0 and item["sidecar_sha256_match"]
                    and type(metadata.get("step")) is int and metadata["step"] == step
                    and metadata.get("roundtrip_verified") is True
                    and isinstance(hashes, dict) and set(hashes) == set(arms)
                    and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)
                            for h in hashes.values())
                )
                if not item["verified"]:
                    item["error"] = "checkpoint bytes or sidecar metadata differ"
            except (OSError, ValueError) as exc:
                item["error"] = str(exc)
        records.append(item)
    return records


def optional_object(path):
    try:
        return read_object(path)
    except (OSError, ValueError) as exc:
        return {"inspection_error": str(exc)}


def file_evidence(path):
    path = Path(path)
    item = {"path": str(path)}
    try:
        item.update(bytes=path.stat().st_size, sha256=sha256(path))
    except OSError as exc:
        item["inspection_error"] = str(exc)
    return item


def code_evidence(plan, manifest):
    """Hash recorded code paths without importing repository or Torch modules."""
    config = manifest.get("runtime_config") or plan.get("source", {}).get("config", {})
    files = []
    if isinstance(config, dict) and config.get("repo"):
        repo = Path(config["repo"])
        files.extend([
            repo / "bda_benchmark/experiments/damageformer_srac_fourstage_v2/losses.py",
            repo / "bda_benchmark/script/standard_ML/train_S5_S8_GPU.py",
        ])
    package = plan.get("source", {}).get("package")
    if package:
        files.append(Path(package) / "deps/original/base_runtime.py")
    return [file_evidence(p) for p in files]


def installed_torch():
    result = {}
    try:
        distribution = importlib.metadata.distribution("torch")
        result["distribution_version"] = distribution.version
        version_path = Path(distribution.locate_file("torch/version.py"))
        result["version_file"] = str(version_path)
        # CUDA build version, not a claim about available GPU/driver at failure.
        tree = ast.parse(version_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id in ("cuda", "__version__", "git_version"):
                    if isinstance(value, ast.Constant):
                        result[target.id] = value.value
    except (OSError, ValueError, SyntaxError, importlib.metadata.PackageNotFoundError) as exc:
        result["inspection_error"] = str(exc)
    return result


def tail_text(path, limit=12000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read(limit).decode("utf-8", errors="replace")
    except OSError as exc:
        return "inspection_error: " + str(exc)


def failure_evidence(root, train, seed):
    text = tail_text(train / "traceback.txt")
    if not text.startswith("inspection_error:"):
        # The exception file is authoritative for this worker; cap the display.
        result = {"worker_traceback_tail": text.splitlines()[-35:]}
    else:
        result = {"worker_traceback_tail": [text]}
    log_tail = tail_text(root / ("logs/%s_train.log" % seed), limit=65536)
    signals = [
        line[:500] for line in log_tail.splitlines()
        if any(key in line for key in (
            "nll_loss_backward", "Assertion", "STOPPED:", "RuntimeError:",
            "CUDA error:", "ValueError:", "Error:"
        ))
    ]
    result["log_error_lines_tail"] = signals[-6:]
    return result


def collect(root, seed=3407):
    root = Path(root).expanduser().resolve()
    plan, manifest = validate_identity(root, seed)
    train = root / ("seed_%s/train" % seed)
    arms = tuple(plan["arms"])
    summary = optional_object(train / "summary.json")
    workflow = optional_object(root / "workflow_state.json")
    journal = read_journal(train / "train.jsonl", arms)
    checkpoints = checkpoint_info(train / "checkpoints", arms)
    eligible = [
        c["step"] for c in checkpoints if c["verified"]
        and c["step"] <= journal["contiguous_prefix_step"]
    ]
    return dict(
        collector="BRIGHT_ABLATION_FAILURE_INSPECT_V1",
        run_root=str(root), seed=seed, arms=arms,
        endpoint_step=plan.get("endpoint_step"),
        addon_release_sha256=plan.get("addon_release_sha256"),
        python={"executable": sys.executable, "version": sys.version.split()[0]},
        torch_metadata_without_import=installed_torch(),
        summary={k: summary[k] for k in (
            "phase", "complete", "training_complete", "full_validation_evidence",
            "final_integrity_pending", "updates_per_arm", "error", "inspection_error"
        ) if k in summary},
        workflow={k: workflow[k] for k in (
            "phase", "current_stage", "error", "inspection_error"
        ) if k in workflow},
        journal=journal, checkpoints=checkpoints,
        latest_checkpoint_matching_sidecar_and_journal=max(eligible, default=None),
        source_code_files_now=code_evidence(plan, manifest),
        failure=failure_evidence(root, train, seed),
        interpretation={
            "next_unjournaled_step": "Candidate journal index only; failure step/image/arm not proven.",
            "checkpoint_verified": "Byte hash and recorded sidecar metadata only; no torch.load or replay.",
            "source_code_files_now": "Recorded paths on disk now; failed-process import binding not inspected.",
            "actions": "Read files and print only. No writes, GPU calls, training, resume, or report promotion.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=DEFAULT_RUN)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    try:
        report = collect(args.run_root, args.seed)
    except (OSError, ValueError) as exc:
        print("INSPECTION STOPPED: " + str(exc), file=sys.stderr)
        return 2
    print("BEGIN ABLATION FAILURE INSPECTION")
    for key, value in report.items():
        if key == "failure":
            for label, lines in value.items():
                print(label + ":")
                print("\n".join(lines))
        else:
            print(key + " = " + json.dumps(value, ensure_ascii=False, sort_keys=True))
    print("END ABLATION FAILURE INSPECTION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
