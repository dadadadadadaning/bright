#!/usr/bin/env python3
"""Read-only source certificate for the completed matched four-arm SAR-Q run.

No model, checkpoint pickle, GPU or training is loaded here. The source must
finish all four arms and its archived Val450 evidence must pass an independent
full-array audit. A positive SAR score is deliberately not a prerequisite.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

FORMAT = "SAR_SOFT_SOURCE_CERTIFICATE_V1"
SOURCE_FORMAT = "BRIGHT_SAR_Q_MATCHED_V1"
ARMS = ("baseline", "rgb_capacity", "shifted_sar", "sar_cue")
OPTIONS = {"baseline": "baseline", "rgb": "rgb_capacity", "shifted": "shifted_sar", "sar": "sar_cue"}
TRAIN_SETTINGS = ("steps", "eval_every", "batch", "accum", "workers", "lr", "aux_weight", "cue_gain", "amp")
HERE = Path(__file__).resolve().parent


class PendingSourceError(RuntimeError):
    """The source workflow has not yet certified complete formal evidence."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return record


def read_source(sar_root):
    """Read small status files only; does not write, load torch or audit arrays."""
    root = Path(sar_root).expanduser().resolve()
    state_path = root / "workflow_state.json"
    manifest_path = root / "manifest.json"
    return {"source_root": str(root),
            "state": read_json(state_path) if state_path.is_file() else None,
            "manifest": read_json(manifest_path) if manifest_path.is_file() else None}


def _true_dict(record, label):
    if not isinstance(record, dict) or not record or any(value is not True for value in record.values()):
        raise ValueError("Failed immutable checks: " + label)


def _bind(bindings, path, expected=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Required source file is missing: " + str(path))
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError("Source SHA256 mismatch: " + str(path))
    previous = bindings.get(str(path))
    if previous is not None and previous != actual:
        raise ValueError("Source file changed during prerequisite check: " + str(path))
    bindings[str(path)] = actual
    return actual


def _inside(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Missing source artifact relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Source artifact is missing or outside its evaluation: " + str(path))
    return path


def _source_complete(state):
    if not state:
        raise PendingSourceError("SAR four-arm workflow_state.json is not available yet")
    if state.get("format") != SOURCE_FORMAT:
        raise ValueError("Unexpected source workflow format")
    if state.get("complete") is not True or state.get("full_validation_evidence") is not True:
        raise PendingSourceError("SAR four-arm source is not formally complete; stage="
                                 + str(state.get("current_stage")) + "; failure=" + str(state.get("failure")))
    if state.get("test_accessed") is not False or state.get("immutable") is not True:
        raise ValueError("Source workflow did not pass Test/integrity gates")


def _default_validator(args):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE))
    module = importlib.import_module("verify_sar")
    if Path(module.__file__).resolve() != HERE / "verify_sar.py":
        raise ValueError("Use a fresh process; verify_sar imported from a different package")
    return module.validate(args)


def check(sar_root, output=None, validator=None, repo_root=None):
    """Return and optionally save a fresh source certificate, or raise.

    PendingSourceError and validation errors never produce a complete or partial
    certificate. ``validator`` is only a CPU-test injection point; the CLI always
    invokes this package's original verify_sar.validate with reports_only=False.
    """
    root = Path(sar_root).expanduser().resolve()
    source = read_source(root)
    manifest = source["manifest"]
    declared_repo = (manifest or {}).get("arguments", {}).get("repo_root")
    repo = Path(repo_root or declared_repo or ".").expanduser().resolve()
    destination = Path(output).expanduser().resolve() if output is not None else None
    if destination is not None:
        if destination.exists():
            raise FileExistsError("Certificate output must be new: " + str(destination))
        if destination.is_relative_to(root):
            raise ValueError("Certificate must be outside the preserved SAR source run")
    state, manifest = source["state"], source["manifest"]
    _source_complete(state)
    if not manifest or manifest.get("format") != SOURCE_FORMAT:
        raise ValueError("Missing or incompatible SAR workflow manifest")
    if manifest.get("test_accessed") is not False or manifest.get("unchanged_original_code") is not True:
        raise ValueError("Source workflow manifest does not certify unchanged source and no Test")
    if not (repo / "bda_benchmark").is_dir():
        raise ValueError("--repo-root must contain the installed bda_benchmark")
    arguments = manifest.get("arguments", {})
    if Path(arguments.get("repo_root", "")).expanduser().resolve() != repo:
        raise ValueError("--repo-root differs from the source pipeline")
    bindings, artifact_bindings = {}, {}
    _bind(bindings, root / "workflow_state.json")
    _bind(bindings, root / "manifest.json", state.get("manifest_sha256"))
    if read_json(root / "workflow_state.json") != state or read_json(root / "manifest.json") != manifest:
        raise ValueError("Source workflow metadata changed during discovery")
    if not isinstance(state.get("manifest_sha256"), str):
        raise ValueError("Missing workflow manifest SHA256")
    source_bindings = manifest.get("source_bindings")
    if not isinstance(source_bindings, dict) or not source_bindings:
        raise ValueError("Source pipeline has no bound source files")
    for path, expected in source_bindings.items():
        _bind(bindings, path, expected)
    plan, stages = manifest.get("stages"), state.get("stages")
    if (not isinstance(plan, list) or not plan or not isinstance(stages, list)
            or [s.get("name") for s in plan] != [s.get("name") for s in stages]):
        raise ValueError("Source stage ledger differs from the declared pipeline")
    stage_map = {}
    for stage in stages:
        name = stage.get("name")
        if name in stage_map or (stage.get("complete") is not True or stage.get("returncode") != 0
                               or stage.get("gate", {}).get("passed") is not True):
            raise ValueError("Incomplete, repeated or failed source stage: " + str(name))
        stage_map[name] = stage
        if stage.get("log"):
            _bind(bindings, stage["log"], stage.get("log_sha256"))
    protocol = Path(arguments.get("protocol", "")).expanduser().resolve()
    if protocol.is_dir():
        protocol = protocol / "protocol.json"
    protocol_hash = _bind(bindings, protocol, manifest.get("protocol_sha256"))
    if not isinstance(manifest.get("protocol_sha256"), str):
        raise ValueError("Missing source protocol SHA256")
    e0 = read_json(protocol)
    if (e0.get("format") != "SRAC_HARDQUERY_PROTOCOL_V1" or e0.get("scope") != "all"
            or e0.get("complete") is not True or e0.get("full_validation_evidence") is not True
            or e0.get("smoke") is not False or e0.get("image_count") != 450
            or e0.get("test_accessed") is not False or e0.get("reproduction", {}).get("passed") is not True
            or e0.get("profile") != "legacy" or e0.get("pool_mode") != "native"):
        raise ValueError("Source E0 is not a passed formal legacy/native all-path Val450 protocol")
    runtime = e0.get("runtime_settings")
    if not isinstance(runtime, dict) or not runtime or runtime != manifest.get("source_runtime_settings"):
        raise ValueError("Source pipeline runtime settings differ from E0")
    environment = manifest.get("runtime_environment", {})
    if environment.get("CUDA_LAUNCH_BLOCKING") != "1":
        raise ValueError("Source SAR pipeline did not use CUDA_LAUNCH_BLOCKING=1")
    if environment.get("CUBLAS_WORKSPACE_CONFIG") != runtime.get("cublas_workspace_config"):
        raise ValueError("Source SAR environment differs from E0 CUBLAS_WORKSPACE_CONFIG")
    e0_inputs = {}
    for name in ("front", "a1", "train_list", "val_list"):
        path = Path(e0["input_paths"][name]).expanduser().resolve()
        e0_inputs[name] = _bind(bindings, path, e0["input_fingerprints"][name])
    reports, train_manifests, checkpoints = {}, {}, {}
    for arm in ARMS:
        fit, val = root / "formal" / ("train_" + arm), root / "formal" / ("val_" + arm)
        train_stage, eval_stage = stage_map.get("formal_train_" + arm), stage_map.get("formal_val_" + arm)
        if not train_stage or not eval_stage:
            raise ValueError("Missing formal source stage: " + arm)
        _bind(bindings, fit / "summary.json", train_stage["gate"].get("summary_sha256"))
        _bind(bindings, fit / "manifest.json")
        _bind(bindings, val / "summary.json", eval_stage["gate"].get("summary_sha256"))
        ts, tm, report = read_json(fit / "summary.json"), read_json(fit / "manifest.json"), read_json(val / "summary.json")
        if (ts.get("format") != "SRAC_Q_FACTORIAL_V1" or ts.get("arm") != arm
                or ts.get("complete") is not True or ts.get("full_validation_evidence") is not True
                or ts.get("smoke") is not False or ts.get("test_accessed") is not False
                or ts.get("steps") != 2000 or ts.get("formal_checkpoint") != "last.pth"
                or ts.get("a1_unchanged") is not True or ts.get("downstream_unchanged") is not True):
            raise ValueError("Source Q training is not a complete frozen-model formal 2000-step run: " + arm)
        _true_dict(ts.get("immutable"), "train_" + arm)
        if (tm.get("format") != "SRAC_Q_FACTORIAL_V1" or tm.get("arm") != arm
                or tm.get("smoke") is not False or tm.get("test_accessed") is not False
                or tm.get("frozen_a1") is not True or tm.get("protocol_sha256") != protocol_hash
                or Path(tm.get("protocol_path", "")).resolve() != protocol
                or tm.get("e0_runtime") != runtime or tm.get("e0_profile") != e0["profile"]
                or tm.get("e0_pool_mode") != e0["pool_mode"]):
            raise ValueError("Source Q training provenance differs from E0: " + arm)
        for field, name in (("front_sha256", "front"), ("a1_sha256", "a1"),
                            ("train_sha256", "train_list"), ("val_sha256", "val_list")):
            if tm.get(field) != e0_inputs[name]:
                raise ValueError("Source Q training input differs: " + arm + "/" + field)
        if (report.get("format") != "SRAC_Q_FACTORIAL_EVAL_V1" or report.get("arm") != arm
                or report.get("complete") is not True or report.get("full_validation_evidence") is not True
                or report.get("smoke") is not False or report.get("test_accessed") is not False
                or report.get("image_count") != 450 or report.get("checkpoint_steps") != 2000
                or report.get("native_a1_replay_ok") is not True
                or report.get("reproduction", {}).get("passed") is not True
                or report.get("q_a1_reproduction", {}).get("passed") is not True
                or report.get("protocol_sha256") != protocol_hash
                or Path(report.get("protocol_path", "")).resolve() != protocol
                or report.get("numerical_profile") != e0["profile"] or report.get("pool_mode") != e0["pool_mode"]):
            raise ValueError("Source Q evaluation is not passed, matched formal Val450: " + arm)
        _true_dict(report.get("immutable"), "val_" + arm)
        expected_settings = {field: tm.get("arguments", {}).get(field) for field in TRAIN_SETTINGS}
        if expected_settings["steps"] != 2000 or report.get("training_hyperparameters") != expected_settings:
            raise ValueError("Q training and evaluation budget/settings differ: " + arm)
        checkpoint = fit / "last.pth"
        checkpoint_hash = _bind(bindings, checkpoint, train_stage["gate"].get("checkpoint_sha256"))
        if checkpoint.stat().st_size == 0 or report.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("Formal evaluation did not use the completed last.pth: " + arm)
        if Path(report.get("checkpoint_path", "")).resolve() != checkpoint.resolve():
            raise ValueError("Formal evaluation checkpoint path is not its trained last.pth: " + arm)
        _bind(bindings, _inside(val, report.get("result_file")), report.get("result_sha256"))
        journal = val / "images.jsonl"
        _bind(bindings, journal)
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 450 or [row.get("index") for row in rows] != list(range(450)):
            raise ValueError("Formal archive is not the ordered Val450 journal: " + arm)
        for row in rows:
            if not isinstance(row.get("prediction_sha256"), str):
                raise ValueError("Missing archived prediction hash: " + arm)
            _bind(artifact_bindings, _inside(val, row.get("prediction_file")), row["prediction_sha256"])
        checkpoints[arm] = {"path": str(checkpoint.resolve()), "sha256": checkpoint_hash}
        train_manifests[arm], reports[arm] = tm, report
    baseline_train, baseline_eval = train_manifests["baseline"], reports["baseline"]
    for arm in ARMS[1:]:
        for field in ("source_hash", "inherited_source_hash", "initial_front_step", "initial_front_complete"):
            if field not in baseline_train or train_manifests[arm].get(field) != baseline_train[field]:
                raise ValueError("Training source identity differs across arms: " + field)
        for field in ("source_sha256", "protocol_sha256", "numerical_profile", "pool_mode", "training_hyperparameters"):
            if not baseline_eval.get(field) or reports[arm].get(field) != baseline_eval[field]:
                raise ValueError("Evaluation source/settings differ across arms: " + field)
    prior_verification_path = root / "sar_verification.json"
    prior = read_json(prior_verification_path)
    if (prior.get("format") != "BRIGHT_EXTERNAL_SAR_VALIDATION_V1" or prior.get("complete") is not True
            or prior.get("reports_only") is not False or prior.get("training") is not False
            or prior.get("test_accessed") is not False):
        raise ValueError("Source pipeline's four-arm artifact certificate is incomplete")
    verify_stage = stage_map.get("verify_four_arm_artifacts")
    if not verify_stage:
        raise ValueError("Source workflow did not execute its four-arm verification stage")
    _bind(bindings, prior_verification_path, verify_stage["gate"].get("report_sha256"))
    kwargs = dict(repo_root=str(repo), reports_only=False, output=None)
    kwargs.update({option: str(root / "formal" / ("val_" + arm)) for option, arm in OPTIONS.items()})
    verification = (validator or _default_validator)(SimpleNamespace(**kwargs))
    if (verification.get("format") != "BRIGHT_EXTERNAL_SAR_VALIDATION_V1"
            or verification.get("complete") is not True or verification.get("reports_only") is not False
            or verification.get("training") is not False or verification.get("test_accessed") is not False):
        raise ValueError("Fresh full-array four-arm verification failed")
    for option in OPTIONS:
        row = verification.get("status", {}).get(option, {})
        audit = row.get("artifact_audit", {})
        if row.get("passed") is not True or audit.get("passed") is not True or audit.get("complete_audit") is not True or audit.get("image_count") != 450:
            raise ValueError("Missing full Val450 archived-array audit: " + option)
    for path, expected in {**bindings, **artifact_bindings}.items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError("Source changed during full-array prerequisite verification: " + path)
    certificate = dict(format=FORMAT, complete=True, full_validation_evidence=True, smoke=False,
        training=False, test_accessed=False, new_cross_experiment_completed=False,
        scope="four-arm SAR-Q source evidence only; no new SAR-Q x soft-route score",
        source_root=str(root), repo_root=str(repo), protocol=str(protocol), protocol_sha256=protocol_hash,
        baseline_checkpoint=checkpoints["baseline"]["path"], baseline_checkpoint_sha256=checkpoints["baseline"]["sha256"],
        sar_checkpoint=checkpoints["sar_cue"]["path"], sar_checkpoint_sha256=checkpoints["sar_cue"]["sha256"],
        checkpoints=checkpoints, q_eval_reports={arm: str(root / "formal" / ("val_" + arm) / "summary.json") for arm in ARMS},
        runtime_environment=environment, source_runtime_settings=runtime,
        source_bindings=bindings, artifact_bindings=artifact_bindings, source_bindings_verified_before_and_after=True,
        artifact_bindings_verified_before_and_after=True,
        fullverification=verification, positive_sar_effect_required=False)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(certificate, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return certificate


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check",))
    p.add_argument("--sar-root", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--output", required=True)
    return p


def main():
    args = parser().parse_args()
    certificate = check(args.sar_root, output=args.output, repo_root=args.repo_root)
    print(json.dumps({key: certificate[key] for key in ("format", "complete", "source_root", "baseline_checkpoint", "sar_checkpoint")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
