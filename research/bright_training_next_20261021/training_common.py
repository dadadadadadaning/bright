"""Launch-time provenance and scoped observation; no objective changes."""
from __future__ import annotations

import contextlib
import functools
import json
from pathlib import Path

from grid_probe import DEFAULT_ROOT, read_json, sha256
from evaluate_checkpoint import observed_evaluation, REQUIRED_CHECKS, TRAINER_SHA256, write_json, concise_metrics

DEFAULT_EVALUATION = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1/eval1500_y5qlliyp"
CHECKPOINT_SHA256 = "ce9a4fd3876ed947bb6399fe96bcba5137f9ea44c96fee1629c60df738691b3d"


def certificate(context, evaluation_root=DEFAULT_EVALUATION):
    """Bind successful standalone evidence to the exact failed-parent checkpoint."""
    import evidence
    parent = context["parent"]
    root = Path(DEFAULT_ROOT)
    if (context["parent_root"] != root or parent["step"] != 1500
            or parent["sha256"] != CHECKPOINT_SHA256
            or context["plan"]["endpoint_step"] != 2500
            or context["seed"] != 3407):
        raise ValueError("This release requires the recorded seed3407 checkpoint1500 and endpoint2500")
    if sha256(root / "trainer.pyz") != TRAINER_SHA256:
        raise ValueError("Original trainer archive differs")
    evaluation_root = Path(evaluation_root).resolve()
    summary = read_json(evaluation_root / "summary.json")
    path = evaluation_root / "val_step_001500/result.json"
    report = read_json(path)
    pending_path = path.with_name("result.pending.json")
    pending = read_json(pending_path)
    if (summary.get("phase") != "standalone_evaluation_complete"
            or summary.get("complete") is not True or summary.get("image_count") != 450
            or report.get("complete") is not True
            or report.get("phase") != "standalone_evaluation_complete"
            or report.get("full_validation_evidence") is not True
            or report.get("optimizer_updates") != 0):
        raise ValueError("Require the completed standalone Val450 report")
    for checks in (summary.get("integrity_checks", {}), report.get("standalone_integrity_checks", {})):
        if any(checks.get(k) is not True for k in REQUIRED_CHECKS):
            raise ValueError("Standalone integrity evidence is incomplete")
    provenance = report["standalone_provenance"]
    if (provenance.get("source_root") != str(root)
            or provenance.get("checkpoint_step") != 1500
            or provenance.get("checkpoint_sha256") != parent["sha256"]
            or provenance.get("trainer_sha256") != TRAINER_SHA256
            or provenance.get("lineage") != context["lineage"]
            or provenance.get("evaluation_source_sha256") != sha256(context["addon"] / "evaluation.py")):
        raise ValueError("Standalone evaluation belongs to a different source/checkpoint")
    if report.get("pending_report_sha256") != sha256(pending_path):
        raise ValueError("Standalone pending-report backup differs")
    names = context["manifest"]["val_names"]
    evidence.validate_report(pending, step=1500, split="val", names=names,
                             arms=list(context["plan"]["arms"]), primary=False)
    evidence.validate_saved_report(path, report, names, sha256, evaluation_root)
    if pending["metrics"] != report["metrics"]:
        raise ValueError("Standalone metrics changed during promotion")
    paths = [root / "trainer.pyz", evaluation_root / "summary.json", path,
             path.with_name("result.json.sha256"), pending_path, path.with_name("images.jsonl")]
    context["evidence_files"].update({str(p): sha256(p) for p in paths})
    context["standalone_evaluation"] = str(evaluation_root)
    return context


def wrap_evaluator(evaluator, root):
    """Observe validation failures, retaining the exact original acceptance rule."""
    if getattr(evaluator, "_next_observed", False):
        return evaluator
    original = evaluator.evaluate

    @functools.wraps(original)
    def evaluate(runtime, heads, names, folder, **kwargs):
        import torch
        destination = Path(root) / "probability_observations" / Path(folder).name
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / "markers.jsonl").open("x") as markers:
            with observed_evaluation(torch, runtime, evaluator, names, destination, markers) as state:
                try:
                    report = original(runtime, heads, names, folder, **kwargs)
                    print("VALIDATION METRICS " + json.dumps(dict(step=kwargs["step"],
                        split=kwargs["split"], metrics=concise_metrics(report)), allow_nan=False), flush=True)
                    return report
                except BaseException as exc:
                    write_json(Path(root) / "evaluation_failure.json", dict(
                        **state, error=type(exc).__name__ + ": " + str(exc),
                        evidence_directory=str(destination), probabilities_modified=False))
                    raise
    evaluator.evaluate = evaluate
    evaluator._next_observed = True
    return evaluator


@contextlib.contextmanager
def observe_bindings(context, output):
    bridge = context["bridge"]
    named = bridge.named

    def load(path, name):
        module = named(path, name)
        return wrap_evaluator(module, output) if Path(path).name == "evaluation.py" else module

    bridge.named = load
    try:
        yield
    finally:
        bridge.named = named
