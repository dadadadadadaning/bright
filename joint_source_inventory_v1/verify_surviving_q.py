#!/usr/bin/env python3
"""CPU-only verification of surviving Q/A1 and standard Train/Val inputs.

This is an inventory/verification tool, not a training or model-recovery entry.
It does not create an E0 protocol, memory bank, predictions, or CUDA tensors.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import sys
import time
import traceback

FRESH_SOURCE_SHA256 = "9f08db4bca1dd2ff029b85966a3f5993a531cda6c21c9ad291c6da34d0f7d7d0"
AUDITED_A1_SHA256 = "cf834b5929c96f84642b8b856c985dc818eecac66a1f01ca3803e1b1dc200add"
DEFAULT_REPO = "/root/shared-nvme/BRIGHT-master"
DEFAULT_Q = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_FULL_CHAIN_V1/run_20261014_003226_vdy7ey5t/sar_q/formal/train_baseline/last.pth"
DEFAULT_A1 = "/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth"
DEFAULT_TOOL = DEFAULT_REPO + "/New/soft_memory_q_rebased_v1_fix2"
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")
EXCLUDED = {".git", "__pycache__", ".cache", ".ipynb_checkpoints", "conda", "conda-envs", "envs"}


def _excluded_name(value):
    return value.lower() in EXCLUDED or value.lower().startswith(("pfllib", "pfllip"))


def _json_safe(value, issues, path="metadata"):
    """Keep reports writable when a damaged payload contains nonfinite metadata."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        issues.append("Nonfinite metadata value: " + path)
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item, issues, path + "." + str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, issues, path + "[%d]" % i) for i, item in enumerate(value)]
    issues.append("Unsupported metadata type at " + path + ": " + type(value).__name__)
    return "<unsupported " + type(value).__name__ + ">"


def progress(message):
    print(message, flush=True)


def _digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_verified_source(tool):
    """Pin the exact existing validator before importing it; no source edits."""
    path = Path(tool).expanduser().resolve() / "fresh_source.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _digest(path)
    if actual != FRESH_SOURCE_SHA256:
        raise ValueError("fresh_source.py differs from the audited fix2 SHA256: " + actual)
    spec = importlib.util.spec_from_file_location("_bright_verified_surviving_q_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return data


def discover(repo):
    """Only walk this BRIGHT bda_benchmark tree; never follow directory links."""
    root = Path(repo).expanduser().resolve() / "bda_benchmark"
    found = dict(train=[], val=[], datasets=[], scan_errors=[])
    if not root.is_dir():
        found["scan_errors"].append("BRIGHT bda_benchmark directory missing: " + str(root))
        return found
    def error(exc):
        found["scan_errors"].append(str(exc))
    for current, dirs, files in os.walk(root, topdown=True, onerror=error, followlinks=False):
        here = Path(current)
        dirs[:] = sorted(name for name in dirs
                         if not _excluded_name(name)
                         and not (here / name).is_symlink())
        for split in ("train", "val"):
            name = split + "_set.txt"
            if name in files:
                found[split].append(str(here / name))
        if {"pre-event", "post-event", "target"} <= set(dirs):
            found["datasets"].append(str(here))
    return found


def match_splits(source, payload, found, overrides, errors):
    result, names = {}, {}
    for split, count in (("train", 2918), ("val", 450)):
        expected = payload.get(split + "_sha256")
        candidates = ([str(Path(overrides[split]).expanduser().resolve())] if overrides.get(split)
                      else sorted(set(found[split])))
        rows, matches = [], []
        for raw in candidates:
            path = Path(raw)
            row = dict(path=str(path), exists=path.is_file(), expected_sha256=expected)
            try:
                if path.name != split + "_set.txt" or path.is_symlink():
                    raise ValueError("Only a regular standard Train/Val list may be read; no Test list or symlink")
                if not path.is_file():
                    raise FileNotFoundError(path)
                row["sha256"] = source.digest(path)
                row["hash_matches_checkpoint"] = row["sha256"] == expected
                if row["hash_matches_checkpoint"]:
                    values = source._split_names(path, count)
                    if any(SAFE_NAME.fullmatch(name) is None for name in values):
                        raise ValueError("Split contains unsafe/noncanonical sample identifiers")
                    row.update(valid=True, image_count=len(values))
                    matches.append((str(path), values))
            except Exception as exc:
                row.update(valid=False, error=type(exc).__name__ + ": " + str(exc))
            rows.append(row)
        result[split] = dict(candidates=rows, matching_paths=[path for path, _ in matches],
                             selected_path=matches[0][0] if matches else None,
                             expected_count=count)
        if not matches:
            errors.append("No complete standard " + split + " split matches checkpoint SHA256")
        else:
            names[split] = matches[0][1]
    result["disjoint"] = bool(set(names) == {"train", "val"} and not set(names["train"]) & set(names["val"]))
    if set(names) == {"train", "val"} and not result["disjoint"]:
        errors.append("Verified Train and Val lists overlap")
    return result, names


def check_dataset(root, names, emit=progress):
    """Mirror only region.core.sample_paths' verified three-path convention.

    Avoid importing its model/metric dependency tree. No image file is opened;
    is_file() checks filesystem metadata for listed Train/Val samples only.
    """
    root = Path(root).expanduser().resolve()
    if any(_excluded_name(part) for part in root.parts):
        raise ValueError("Excluded dataset location; PFLlib/conda/.git are not inspected")
    required = ("pre-event", "post-event", "target")
    row = dict(path=str(root), path_rule="region.core.sample_paths equivalent; pathlib existence only",
               directory_layout=all((root / name).is_dir() and not (root / name).is_symlink() for name in required),
               image_contents_read=False, test_accessed=False, splits={})
    for split in ("train", "val"):
        entries = names.get(split, [])
        counts = dict(pre=0, post=0, label=0)
        missing = []
        for index, name in enumerate(entries):
            if SAFE_NAME.fullmatch(name) is None:
                raise ValueError("Unsafe sample identifier")
            paths = dict(pre=root / "pre-event" / (name + "_pre_disaster.tif"),
                         post=root / "post-event" / (name + "_post_disaster.tif"),
                         label=root / "target" / (name + "_building_damage.tif"))
            for field, path in paths.items():
                if path.is_file():
                    counts[field] += 1
                elif len(missing) < 30:
                    missing.append(str(path))
            if (index + 1) % 500 == 0 or index + 1 == len(entries):
                emit("数据存在性检查 %s %s %d/%d" % (root, split, index + 1, len(entries)))
        row["splits"][split] = dict(image_count=len(entries), expected_files=3 * len(entries),
            existing_files=sum(counts.values()), by_type=counts, missing_examples=missing,
            missing_count=3 * len(entries) - sum(counts.values()),
            complete=bool(entries) and all(value == len(entries) for value in counts.values()))
    row["complete"] = row["directory_layout"] and all(row["splits"][s]["complete"] for s in ("train", "val"))
    return row


def _adapter_report(payload, torch):
    adapter = payload.get("adapter")
    if not isinstance(adapter, dict) or not adapter:
        return dict(present=False, all_tensors_finite=False, errors=["Missing/non-dictionary adapter"])
    report = dict(present=True, tensors=0, elements=0, dtypes={}, all_tensors_finite=True, errors=[])
    for name, value in adapter.items():
        if not torch.is_tensor(value):
            report["errors"].append("Adapter entry is not a tensor: " + str(name))
            report["all_tensors_finite"] = False
            continue
        report["tensors"] += 1
        report["elements"] += value.numel()
        dtype = str(value.dtype)
        report["dtypes"][dtype] = report["dtypes"].get(dtype, 0) + 1
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            report["all_tensors_finite"] = False
            report["errors"].append("Nonfinite or non-CPU adapter tensor: " + str(name))
    report["architecture_shapes_verified"] = False
    return report


def verify(args, emit=progress):
    started = time.monotonic()
    repo, q, a1, tool = (Path(value).expanduser().resolve() for value in
                        (args.repo, args.q_checkpoint, args.a1_checkpoint, args.tool_dir))
    report = dict(format="BRIGHT_SURVIVING_Q_CPU_VERIFICATION_V1", execution_complete=False,
        training=False, cuda_forward=False, cuda_tensors_created=False, a1_weights_loaded=False,
        image_contents_read=False, test_accessed=False, historical_e0_reverified=False,
        historical_e0_source_reverified=False, memory_built=False, source_ready_for_training=False,
        cpu_foundation_verified=False, errors=[], warnings=[], inputs=dict(repo=str(repo),
        q_checkpoint=str(q), a1_checkpoint=str(a1), tool_dir=str(tool)),
        verification_scope="CPU artifact/formal-Q/split checks; dataset existence only; no E0 or Memory reconstruction")
    errors = report["errors"]
    source = None
    emit("开始：仅 CPU 核查 Q 与 A1；不加载 A1 权重，不训练，不读取 Test。")
    try:
        source = load_verified_source(tool)
        report["reused_validator"] = dict(path=str(tool / "fresh_source.py"), sha256=FRESH_SOURCE_SHA256,
            functions=["_validate_formal(TRAINING_EVIDENCE)", "_split_names", "_dependencies", "digest"])
    except Exception as exc:
        errors.append("Pinned validator: " + type(exc).__name__ + ": " + str(exc))
    digest = source.digest if source is not None else _digest
    for field, path in (("q", q), ("a1", a1)):
        record = report[field] = dict(path=str(path), exists=path.is_file())
        if path.is_file():
            record["bytes"] = path.stat().st_size
            emit("计算 %s SHA256（%.1f MiB，NFS 可能较慢）" % (field.upper(), path.stat().st_size / 2**20))
            try:
                record["sha256"] = digest(path)
            except Exception as exc:
                errors.append(field + " digest: " + str(exc))
        else:
            errors.append("Missing file: " + str(path))
    report["a1"]["matches_audited_a1"] = report["a1"].get("sha256") == AUDITED_A1_SHA256
    report["a1"]["audited_a1_sha256"] = AUDITED_A1_SHA256
    report["q"]["formal_last_path"] = (q.name == "last.pth" and q.parent.name == "train_baseline"
                                        and q.parent.parent.name == "formal")
    if not report["q"]["formal_last_path"]:
        errors.append("Q input must be formal/train_baseline/last.pth; best/smoke checkpoints are not accepted")
    if report["a1"].get("sha256") and not report["a1"]["matches_audited_a1"]:
        errors.append("A1 file differs from this repository's audited load_a1_strict SHA256")
    payload, torch = {}, None
    if q.is_file():
        emit("CPU 读取受信本地 Q 检查点并检查 adapter；不加载大 A1 检查点。")
        try:
            import torch as torch_module
            torch = torch_module
            payload = torch.load(q, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError("Q checkpoint is not a dictionary")
            fields = ("format", "arm", "step", "completed_steps", "run_complete", "smoke", "test_accessed",
                      "full_validation_evidence", "a1_unchanged", "downstream_unchanged", "config", "cue_gain",
                      "e0_pool_mode", "e0_profile", "e0_runtime", "source_hash", "inherited_source_hash",
                      "train_sha256", "val_sha256", "protocol_path", "protocol_sha256", "front_checkpoint",
                      "front_sha256", "a1_checkpoint", "a1_sha256", "arguments", "valid_miou")
            report["checkpoint_metadata"] = _json_safe({key: payload.get(key) for key in fields}, errors)
            report["adapter"] = _adapter_report(payload, torch)
            if not report["adapter"]["all_tensors_finite"]:
                errors.extend(report["adapter"]["errors"])
            report["a1"]["matches_checkpoint"] = report["a1"].get("sha256") == payload.get("a1_sha256")
            if not report["a1"]["matches_checkpoint"]:
                errors.append("A1 SHA256 does not match Q checkpoint")
        except Exception as exc:
            errors.append("Q CPU payload: " + type(exc).__name__ + ": " + str(exc))
            payload = {}
    paths = dict(q_checkpoint=q, a1_checkpoint=a1, q_train_validation=q.parent / "validation.jsonl")
    metadata = {}
    for name, path in (("summary", q.parent / "summary.json"), ("manifest", q.parent / "manifest.json")):
        try:
            metadata[name] = _read(path)
        except Exception as exc:
            errors.append("Formal Q " + name + ": " + str(exc))
    if not paths["q_train_validation"].is_file():
        errors.append("Missing formal Q training Val journal: " + str(paths["q_train_validation"]))
    independent = q.parent.parent / "val_baseline"
    report["validation_evidence"] = dict(selected="training_final_val450",
        training_journal=str(paths["q_train_validation"]),
        independent_summary_exists=(independent / "summary.json").is_file(),
        independent_result_exists=(independent / "result.json").is_file(),
        independent_evidence_used=False, formal_reused_validator_passed=False,
        explanation="Explicitly verify the preserved formal training Val450; independent summaries alone are not full evidence.")
    if (source is not None and payload and len(metadata) == 2 and paths["q_train_validation"].is_file()
            and a1.is_file() and report["q"]["formal_last_path"]):
        emit("复用原 _validate_formal：核对2000步正式训练及其 Val450 证据。")
        try:
            deps = source._dependencies(repo)
            source._validate_formal(payload, metadata["summary"], metadata["manifest"],
                                    None, None, paths, deps, source.TRAINING_EVIDENCE)
            report["validation_evidence"]["formal_reused_validator_passed"] = True
        except Exception as exc:
            errors.append("Reused formal Q validation: " + type(exc).__name__ + ": " + str(exc))
    emit("在 BRIGHT/bda_benchmark 内查找 Train/Val 名单和数据根；不扫描 PFLlib、conda 或 Test 名单。")
    found = discover(repo)
    report["discovery"] = found
    errors.extend(found["scan_errors"])
    names = {}
    if source is not None and payload:
        report["splits"], names = match_splits(source, payload, found,
            dict(train=args.train_list, val=args.val_list), errors)
    else:
        report["splits"] = dict(disjoint=False, reason="Checkpoint or verified split parser unavailable")
    candidates = ([str(Path(args.dataset).expanduser().resolve())] if args.dataset else sorted(set(found["datasets"])))
    report["datasets"] = []
    if set(names) == {"train", "val"} and report["splits"]["disjoint"]:
        for candidate in candidates:
            try:
                report["datasets"].append(check_dataset(candidate, names, emit))
            except Exception as exc:
                report["datasets"].append(dict(path=candidate, complete=False, error=str(exc)))
        if not any(row.get("complete") for row in report["datasets"]):
            errors.append("No complete dataset root contains all listed Train/Val pre/post/target files")
    else:
        report["warnings"].append("Dataset pixel-file existence check deferred until matching Train/Val lists are found")
    report["complete_dataset_roots"] = [r["path"] for r in report["datasets"] if r.get("complete")]
    report["cpu_foundation_verified"] = bool(not errors and report["validation_evidence"]["formal_reused_validator_passed"]
        and report.get("adapter", {}).get("all_tensors_finite") and report["complete_dataset_roots"])
    report.update(execution_complete=True, elapsed_seconds=time.monotonic() - started)
    report["warnings"].append("Even a successful CPU inventory does not establish a new training source: original E0 is not reverified; Memory selection/banks and CUDA validation are still absent.")
    return report


def render(report):
    lines = ["BRIGHT 现存 Q / A1 只读 CPU 核查", "执行完成: " + str(report["execution_complete"]),
             "CPU基础核查通过: " + str(report["cpu_foundation_verified"]),
             "可直接训练: False（本工具不是新训练入口）", "训练=False；CUDA前向=False；Test=False；Memory建库=False",
             "原 E0 重新校验=False", "", "证据: " + json.dumps(report.get("validation_evidence", {}), ensure_ascii=False),
             "匹配名单: " + json.dumps({s: report.get("splits", {}).get(s, {}).get("matching_paths", []) for s in ("train", "val")}, ensure_ascii=False),
             "完整数据根: " + json.dumps(report.get("complete_dataset_roots", []), ensure_ascii=False), "", "错误/缺失:"]
    lines.extend("- " + error for error in report["errors"])
    if not report["errors"]:
        lines.append("- 无；仍需新来源入口、新 Memory 和真实 CUDA 检查。")
    lines.extend(["", "Q文件与指纹:", json.dumps(report.get("q", {}), ensure_ascii=False, indent=2),
                  "A1文件与指纹比较:", json.dumps(report.get("a1", {}), ensure_ascii=False, indent=2),
                  "Adapter统计（不输出权重值）:", json.dumps(report.get("adapter", {}), ensure_ascii=False, indent=2),
                  "检查点元数据（包含 config、e0_runtime、原路径）:",
                  json.dumps(report.get("checkpoint_metadata", {}), ensure_ascii=False, indent=2),
                  "Train/Val名单候选及哈希:", json.dumps(report.get("splits", {}), ensure_ascii=False, indent=2),
                  "数据根完整度（仅存在性，不读取图像内容）:"])
    for row in report.get("datasets", []):
        concise = {key: row.get(key) for key in ("path", "complete", "directory_layout", "error") if key in row}
        concise["splits"] = {split: dict(image_count=value.get("image_count"),
            expected_files=value.get("expected_files"), existing_files=value.get("existing_files"),
            missing_count=value.get("missing_count"), missing_examples=value.get("missing_examples", [])[:3])
            for split, value in row.get("splits", {}).items()}
        lines.append(json.dumps(concise, ensure_ascii=False, indent=2))
    lines.extend(["", "完整结构化报告见 result.json。CPU核查通过仍不代表 Memory、CUDA 或训练入口已经就绪。"])
    return "\n".join(lines) + "\n"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--q-checkpoint", default=DEFAULT_Q)
    result.add_argument("--a1-checkpoint", default=DEFAULT_A1)
    result.add_argument("--repo", default=DEFAULT_REPO)
    result.add_argument("--tool-dir", default=DEFAULT_TOOL)
    result.add_argument("--train-list")
    result.add_argument("--val-list")
    result.add_argument("--dataset")
    result.add_argument("--output", help="New output directory; existing directories are never overwritten")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    # This process performs CPU inspection only. Recorded original GPU settings
    # are reported from the checkpoint; we do not compare them to this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output = Path(args.output).expanduser().resolve() if args.output else Path(args.repo).expanduser().resolve() / "New" / (
        "surviving_q_verify_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3))
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = verify(args)
    except Exception as exc:
        report = dict(format="BRIGHT_SURVIVING_Q_CPU_VERIFICATION_V1", execution_complete=False,
            cpu_foundation_verified=False, source_ready_for_training=False, training=False,
            cuda_forward=False, test_accessed=False, memory_built=False, historical_e0_reverified=False,
            errors=[type(exc).__name__ + ": " + str(exc)])
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    issues = []
    report = _json_safe(report, issues, "report")
    if issues:
        report["errors"].extend(issues)
        report["cpu_foundation_verified"] = False
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = render(report)
    (output / "REPORT.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    print("REPORT:", output / "REPORT.txt", flush=True)
    print("JSON:", output / "result.json", flush=True)
    return 0 if report["cpu_foundation_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
