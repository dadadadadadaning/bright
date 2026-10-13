#!/usr/bin/env python3
"""Replay the unchanged joint package's initial Val path, without training.

Run in its own process. CUDA_LAUNCH_BLOCKING=1 must be set before Python starts.
This worker does not change a pool implementation, numerical tolerance, model,
bank, inherited protocol or any file below the old failed output directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

FORMAT = "SRAC_JOINT_POOL_SEQUENCE_DIAGNOSTIC_V1"
ARMS = ("r0", "rt", "rf")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return data


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tool", required=True, help="Unmodified original joint training package")
    p.add_argument("--repo", required=True)
    p.add_argument("--failed-run", required=True, help="Preserved old workflow root or train directory")
    p.add_argument("--output", required=True, help="NEW diagnostic directory")
    p.add_argument("--scope", choices=("failed_image", "val450"), required=True)
    p.add_argument("--image", help="Failed-image override; must belong to the old exact Val list")
    return p


def evidence_paths(train_folder):
    paths = []
    for relative in ("manifest.json", "summary.json", "traceback.txt", "initial_val/result.json",
                     "initial_val/images.jsonl"):
        path = train_folder / relative
        if path.is_file():
            paths.append(path)
    debug = train_folder / "pool_debug"
    if debug.is_dir():
        paths.extend(path for path in debug.rglob("*") if path.is_file())
    return sorted(set(paths))


def fingerprints(paths):
    return {str(path.resolve()): digest(path) for path in paths}


def verify_fingerprints(record):
    return {name: Path(name).is_file() and digest(name) == expected
            for name, expected in record.items()}


def load_original(tool, repo):
    # Generic names common/model/train/run belong to the old standalone package.
    # Reject an already-imported foreign module instead of silently monkeypatching.
    tool = Path(tool).resolve()
    sys.path.insert(0, str(tool))
    sys.path.insert(1, str(Path(repo).resolve()))
    imported = {}
    for name in ("common", "model", "run", "train", "server_runtime"):
        existing = sys.modules.get(name)
        if existing is not None and (not getattr(existing, "__file__", None)
                or Path(existing.__file__).resolve().parent != tool):
            raise RuntimeError("Use a fresh worker process; conflicting module: " + name)
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != tool:
            raise RuntimeError("Original module imported from the wrong directory: " + name)
        imported[name] = module
    import torch
    return SimpleNamespace(**imported, torch=torch)


def environment_binding(manifest):
    settings = manifest.get("protocol_info", {}).get("source_runtime_settings", {})
    expected_cublas = settings.get("cublas_workspace_config")
    actual_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        raise ValueError("Set CUDA_LAUNCH_BLOCKING=1 before starting this fresh Python worker")
    if actual_cublas != expected_cublas:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG differs from the preserved source runtime: "
                         + repr(actual_cublas) + " != " + repr(expected_cublas))
    return {"CUDA_LAUNCH_BLOCKING": "1", "CUBLAS_WORKSPACE_CONFIG": actual_cublas,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "purpose": "synchronous CUDA fault localization; not a historical exact-execution claim"}


def choose_names(args, manifest, train_folder):
    names = manifest.get("val_names")
    if not isinstance(names, list) or len(names) != 450 or len(set(names)) != 450:
        raise ValueError("The preserved formal manifest must contain the exact unique Val450 list")
    if args.scope == "val450":
        if args.image:
            raise ValueError("--image is only valid with --scope failed_image")
        return list(names)
    name = args.image
    if not name:
        failure = read_json(train_folder / "pool_debug/failure.json")
        name = failure.get("image") or failure.get("failed_image")
    if not isinstance(name, str) or name not in names:
        raise ValueError("Failed image is absent from the preserved Val450 list")
    return [name]


def _cpu_failure_observations(output, runtime):
    """Never synchronize or copy a CUDA tensor after a CUDA exception."""
    record = {"pool_guard_report": getattr(runtime, "guard_report", None) if runtime else None}
    for relative in ("pool_debug/failure.json", "pool_debug/summary.json", "pool_debug/report.json"):
        path = output / relative
        if path.is_file():
            try:
                record[relative] = read_json(path)
            except (OSError, ValueError) as error:
                record[relative] = {"read_error": str(error)}
    return record


def run(args, modules=None):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Preserving existing output; use a new directory: " + str(output))
    failed_run = Path(args.failed_run).expanduser().resolve()
    train_folder = failed_run / "train" if (failed_run / "train/manifest.json").is_file() else failed_run
    tool, repo = Path(args.tool).expanduser().resolve(), Path(args.repo).expanduser().resolve()
    for protected in (tool, failed_run):
        if output == protected or output.is_relative_to(protected):
            raise ValueError("Diagnostic output must not be inside an original package/run")
    output.mkdir(parents=True)
    summary = {"format": FORMAT, "diagnostic_complete": False, "gates_passed": False,
               "whole_val450_processed": False, "training": False, "test_accessed": False,
               "formal_training_completed": False, "full_validation_evidence": False,
               "historical_reproduction_claim": False, "updates_per_arm": dict.fromkeys(ARMS, 0),
               "phase": "binding", "scope": args.scope, "arguments": vars(args)}
    write_json(output / "summary.json", summary)
    runtime = None
    before = {}
    diagnostic_hashes = {}
    input_ledger_before = {}
    try:
        manifest = read_json(train_folder / "manifest.json")
        old_summary = read_json(train_folder / "summary.json")
        original_args = manifest["arguments"]
        if original_args.get("mode") != "train" or original_args.get("steps") != 5000:
            raise ValueError("Expected the preserved formal 5000-step run manifest")
        if old_summary.get("phase") != "initial_val" or old_summary.get("updates_per_arm") != dict.fromkeys(ARMS, 0):
            raise ValueError("Expected an initial-Val failure with exactly zero updates in all three arms")
        if Path(original_args["repo"]).expanduser().resolve() != repo:
            raise ValueError("--repo differs from the preserved run; source relocation is not supported")
        names = choose_names(args, manifest, train_folder)
        summary["environment"] = environment_binding(manifest)
        before = fingerprints(evidence_paths(train_folder))
        package_files = [p for p in tool.iterdir() if p.is_file()]
        before.update(fingerprints(package_files))
        diagnostic_sources = [p for p in Path(__file__).resolve().parent.glob("*.py")
                              if not p.name.startswith("test_")]
        diagnostic_hashes = fingerprints(diagnostic_sources)
        summary.update(requested_image_count=len(names), requested_names=names,
                       preserved_manifest=str(train_folder / "manifest.json"),
                       preserved_manifest_sha256=digest(train_folder / "manifest.json"),
                       diagnostic_source_sha256=diagnostic_hashes)
        write_json(output / "preserved_input_sha256.json", before)
        write_json(output / "summary.json", summary)
        modules = modules or load_original(tool, repo)
        release = modules.run.verify_release()
        if modules.common.source_hash() != manifest.get("source_hash"):
            raise ValueError("Original training package source hash differs from the old manifest")
        summary["original_release"] = release
        replay_args = SimpleNamespace(**copy.deepcopy(original_args))
        replay_args.output = str(output)
        # Match the original order: seed, load frozen runtime (which restores E0
        # seed), reseed, construct identical heads and AdamW, then initial Val.
        modules.train._seed(int(replay_args.seed))
        with modules.server_runtime.Runtime(replay_args, output) as runtime:
            modules.train._seed(int(replay_args.seed))
            train_names, val_names = modules.train._names(runtime, False)
            if list(val_names) != manifest["val_names"] or list(train_names) != manifest["train_names"]:
                raise ValueError("Runtime Train/Val identities or order differ from the preserved manifest")
            if runtime.protocol_info != manifest["protocol_info"]:
                raise ValueError("Frozen source/bank/runtime bindings differ from the preserved manifest")
            heads, head_hashes = modules.train._new_heads(replay_args, runtime.device)
            if head_hashes != manifest["initial_head_hashes"]:
                raise ValueError("Initial head hashes differ from the preserved run")
            optimizers = {arm: modules.torch.optim.AdamW(head.parameters(), lr=replay_args.lr,
                                                        weight_decay=replay_args.weight_decay)
                          for arm, head in heads.items()}
            # Keep all three optimizers alive to match allocation/reference
            # lifetimes; no zero_grad, backward, state update or step is called.
            summary["optimizer_objects_created"] = list(optimizers)
            input_ledger_before = {str(p): h for p, h in getattr(runtime, "input_ledger", {}).items()}
            summary.update(phase="initial_val_replay", initial_head_hashes=head_hashes,
                           protocol_info=runtime.protocol_info)
            write_json(output / "summary.json", summary)
            result = modules.train._evaluate(runtime, heads, names, output / "initial_val",
                seed=replay_args.seed, smoke=args.scope != "val450", initial=True, save_arrays=True)
            summary.update(whole_val450_processed=result.get("whole_val450_processed") is True,
                           image_count=result.get("image_count"), phase="final_integrity",
                           initial_probability_checks=result.get("initial_probability_checks"))
            summary["runtime_integrity"] = modules.train._checked_verify(runtime)
            if any(result["coverage"][arm]["supported_queries"] == 0 for arm in ("rt", "rf")):
                raise ValueError("Replay queries contain no legal historical support")
            after_heads = {arm: modules.common.module_hash(head) for arm, head in heads.items()}
            if after_heads != head_hashes:
                raise ValueError("Evaluation changed initialized heads")
            summary["heads_unchanged"] = True
            summary["input_ledger_before"] = input_ledger_before
            summary["input_ledger_after"] = {str(p): h for p, h in getattr(runtime, "input_ledger", {}).items()}
        summary["pool_guard_report"] = modules.train._restored_guard(runtime)
        unchanged = verify_fingerprints(before)
        diagnostic_unchanged = verify_fingerprints(diagnostic_hashes)
        if not all(unchanged.values()) or not all(diagnostic_unchanged.values()):
            raise ValueError("Preserved evidence/package/diagnostic source changed during replay")
        if modules.common.source_hash() != manifest["source_hash"]:
            raise ValueError("Original package code changed during replay")
        summary.update(diagnostic_complete=True, gates_passed=True, phase="complete",
                       preserved_inputs_unchanged=unchanged,
                       diagnostic_sources_unchanged=diagnostic_unchanged,
                       meaning="No-training initial Val replay passed in this synchronous process only")
        result.update(final_integrity_pending=False, diagnostic_only=True,
                      full_validation_evidence=False, formal_training_completed=False,
                      training=False, test_accessed=False,
                      diagnostic_integrity=summary["runtime_integrity"],
                      guard_report=summary["pool_guard_report"])
        write_json(output / "initial_val/result.json", result)
        write_json(output / "summary.json", summary)
        print("SEQUENCE DIAGNOSTIC COMPLETE. No training. No Test. No trained-model score claim.", flush=True)
        return summary
    except BaseException as exc:
        # Runtime.__exit__ owns its existing restoration. Never call runtime
        # verify(), module_hash(), CUDA synchronize(), or CUDA allocator APIs here.
        summary.update(phase="failed", diagnostic_complete=False, gates_passed=False,
                       error=type(exc).__name__ + ": " + str(exc),
                       training=False, formal_training_completed=False,
                       full_validation_evidence=False)
        summary["cpu_failure_observations"] = _cpu_failure_observations(output, runtime)
        summary["input_ledger_before"] = input_ledger_before
        summary["input_ledger_after"] = {str(p): h for p, h in getattr(runtime, "input_ledger", {}).items()} if runtime else {}
        if before:
            try:
                summary["preserved_inputs_unchanged"] = verify_fingerprints(before)
                summary["diagnostic_sources_unchanged"] = verify_fingerprints(diagnostic_hashes)
            except OSError as error:
                summary["preserved_input_verification_error"] = str(error)
        journal = output / "initial_val/images.jsonl"
        if journal.is_file():
            # Count persisted rows without reopening any tensor or GPU result.
            summary["persisted_image_rows"] = sum(bool(line.strip()) for line in journal.read_text(errors="replace").splitlines())
        saved_result = output / "initial_val/result.json"
        if saved_result.is_file():
            try:
                result_record = read_json(saved_result)
                summary["whole_val450_processed"] = result_record.get("whole_val450_processed") is True
                summary["image_count"] = result_record.get("image_count")
                summary["initial_probability_checks"] = result_record.get("initial_probability_checks")
            except (OSError, ValueError) as error:
                summary["saved_result_read_error"] = str(error)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(output / "summary.json", summary)
        raise


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
