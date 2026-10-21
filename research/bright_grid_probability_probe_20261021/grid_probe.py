#!/usr/bin/env python3
"""Inspect frozen Grid probabilities in a new directory; never train or promote reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace

DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/resume_20261020_161047_4184092"
DEFAULT_IMAGE = "morocco-earthquake_00000402"
FORMAT = "BRIGHT_GRID_PROBABILITY_PROBE_V1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def probability_details(torch, probability, target):
    """Separate the original validator's conditions without changing its input."""
    result = dict(is_tensor=isinstance(probability, torch.Tensor), reasons=[])
    if not result["is_tensor"]:
        result["reasons"].append("not_tensor")
        return result
    result.update(dtype=str(probability.dtype), shape=list(probability.shape),
                  device=str(probability.device), target_device=str(target.device),
                  expected_shape=[1, 4, *target.shape[1:]],
                  floating=probability.is_floating_point())
    if not result["floating"]:
        result["reasons"].append("not_floating")
    shape_ok = result["shape"] == result["expected_shape"]
    if not shape_ok:
        result["reasons"].append("shape_mismatch")
    if probability.device != target.device:
        result["reasons"].append("device_mismatch")
    if not result["floating"]:
        return result
    # Analyze a CPU copy. Never clamp, normalize, or replace the source tensor.
    p = probability.detach().to(device="cpu", dtype=torch.float64)
    finite = torch.isfinite(p)
    count = int(p.numel())
    nf = int((~finite).sum())
    result.update(elements=count, nonfinite_count=nf,
                  nan_count=int(torch.isnan(p).sum()),
                  positive_inf_count=int(torch.isposinf(p).sum()),
                  negative_inf_count=int(torch.isneginf(p).sum()),
                  finite_min=float(p[finite].min()) if bool(finite.any()) else None,
                  finite_max=float(p[finite].max()) if bool(finite.any()) else None,
                  negative_count=int((p < 0).sum()), above_one_count=int((p > 1).sum()),
                  dtype_epsilon=float(torch.finfo(probability.dtype).eps))
    if nf:
        result["reasons"].append("nonfinite")
    if result["negative_count"] or result["above_one_count"]:
        result["reasons"].append("outside_0_1")
    for key, value in (("negative_excess", result["finite_min"]),
                       ("above_one_excess", result["finite_max"])):
        result[key] = None if value is None else max(0., -value if key == "negative_excess" else value - 1.)
    if shape_ok and count:
        sums = p.sum(1)
        valid = torch.isfinite(sums)
        tolerance = max(1e-5, 4 * result["dtype_epsilon"])
        delta = (sums - 1).abs()
        result.update(sum_tolerance=tolerance,
                      sum_max_abs_error=float(delta[valid].max()) if bool(valid.any()) else None,
                      unnormalized_pixel_count=int(((delta > tolerance) | ~valid).sum()))
        if result["unnormalized_pixel_count"]:
            result["reasons"].append("unnormalized")
    return result


def locate(root, image):
    manifest = read_json(root / "manifest.json")
    summary = read_json(root / "summary.json")
    if manifest.get("format") != "BRIGHT_SLOT_MEMORY_RESUME_V1" or summary.get("phase") != "failed":
        raise ValueError("Require the recorded failed formal continuation")
    if summary.get("failure_phase") != "evaluate_val_1500":
        raise ValueError("Expected the recorded evaluate_val_1500 failure")
    names = manifest["val_names"]
    if len(names) != 450 or len(set(names)) != 450 or image not in names:
        raise ValueError("Require the exact Val450 and a candidate in that split")
    rows = [json.loads(s) for s in (root / "val_step_001500/images.jsonl").read_text().splitlines() if s.strip()]
    n = len(rows)
    if [r["image"] for r in rows] != names[:n] or n >= len(names) or names[n] != image:
        raise ValueError("Candidate is not the next image after the verified completed prefix")
    cp = root / "checkpoints/checkpoint_001500.pt"
    record = summary["last_checkpoint"]
    if record.get("step") != 1500 or record.get("roundtrip_verified") is not True:
        raise ValueError("Require the recorded roundtrip-verified step1500 checkpoint")
    digest = sha256(cp)
    if digest != record.get("sha256") or digest != read_json(cp.with_suffix(".pt.json")).get("sha256"):
        raise ValueError("Checkpoint bytes differ from recorded hash")
    return manifest, dict(candidate=image, candidate_index_zero_based=n,
                          completed_old_images=n, checkpoint_sha256=digest,
                          checkpoint_path=str(cp)), names[:n + 1]


def inspect_images(torch, runtime, factory, evaluator, original_train, names, arms, emit):
    """Run only frozen prepare/support prepare/validation, never head or optimizer updates."""
    checked = []
    modes = evaluator._capture_modes(runtime, {})
    try:
        for module, _ in modes:
            module.training = False
        with torch.no_grad():
            for index, batch in enumerate(runtime.evaluation_batches(names, seed=runtime.args.seed)):
                if index >= len(names):
                    raise ValueError("Evaluation loader returned extra images")
                emit(dict(event="image_begin", index=index, expected_image=names[index]))
                example = runtime.prepare(batch)
                if example.image != names[index]:
                    raise ValueError("Evaluation image order differs")
                original_train._assert_example(example)
                p = example.common["base_probability_full"]
                before = probability_details(torch, p, example.target)
                emit(dict(event="grid_before_support", index=index, image=example.image,
                          probability=before))
                support = factory.prepare(example, arms)
                if set(support) != set(arms):
                    raise ValueError("Support factory returned different arms")
                after = probability_details(torch, example.common["base_probability_full"], example.target)
                validation_error = None
                try:
                    evaluator._validate_probability(example.common["base_probability_full"],
                                                    example.target, "grid_probq")
                except ValueError as exc:
                    validation_error = str(exc)
                row = dict(event="grid_after_support", index=index, image=example.image,
                           probability=after, original_validator_error=validation_error,
                           before_support_probability=before)
                emit(row)
                checked.append(row)
                if validation_error or before["reasons"] or after["reasons"]:
                    return dict(phase="invalid_grid_probability_observed", images_checked=len(checked),
                                first_invalid=row)
        if len(checked) != len(names):
            raise ValueError("Evaluation loader returned too few images")
        return dict(phase="frozen_forward_passed", images_checked=len(checked),
                    candidate_result=checked[-1])
    finally:
        for module, training in modes:
            module.training = training


def run(args):
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    root = Path(args.root).resolve()
    print("GRID PROBE: verifying existing metadata and step1500 checkpoint bytes", flush=True)
    manifest, evidence, prefix = locate(root, args.image)
    names = [args.image] if args.scope == "candidate" else prefix
    archive = root / "trainer.pyz"
    if not archive.is_file():
        raise ValueError("Pinned trainer.pyz is missing")
    # Import helpers only, never __main__, start, worker or run_training.
    sys.path.insert(0, str(archive))
    from resume_context import prepare, inactive_parent
    context = prepare(SimpleNamespace(original_run=manifest["lineage"]["original_run"],
                                     seed=manifest["seed"], checkpoint_every=50,
                                     addon_dir=None, parent_root=str(root)))
    if context["parent"]["step"] != 1500 or context["parent"]["sha256"] != evidence["checkpoint_sha256"]:
        raise ValueError("Selected parent differs from recorded checkpoint1500")
    plan, original = context["plan"], context["manifest"]
    source, bridge, addon = plan["source"], context["bridge"], context["addon"]
    # Match the recorded CUDA setting before importing Torch.
    blocking = original["protocol_info"]["execution_settings"].get("cuda_launch_blocking")
    if blocking is None:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    else:
        os.environ["CUDA_LAUNCH_BLOCKING"] = str(blocking)
    output_base = root.parent.parent / "SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS"
    output_base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="grid_probability_", dir=output_base))
    report = dict(format=FORMAT, output=str(output), **evidence,
                  scope=args.scope, planned_images=len(names), source_root=str(root),
                  optimizer_updates=0, heads_restored=False, test_accessed=False,
                  formal_evaluation=False, historical_state_replayed=False,
                  diagnostic_complete=False, trainer_sha256=sha256(archive),
                  limitation="Frozen frontend and support preparation only; no historical training or head-forward state replay. Passing does not resolve the previous failure.")
    print("OUTPUT=" + str(output), flush=True)
    stage = "load_runtime"
    last_marker = None
    with (output / "trace.jsonl").open("x", encoding="utf-8") as journal:
        def emit(row):
            nonlocal last_marker
            journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            if row["event"] == "image_begin":
                last_marker = row
                if row["index"] % 10 == 0 or row["expected_image"] == args.image:
                    print("GRID PROBE %s/%s %s" % (row["index"] + 1, len(names), row["expected_image"]), flush=True)
        try:
            with inactive_parent(root):
                import torch
                policy = read_json(source["config"]["protocol"])["numeric_policy"]
                torch.set_num_threads(int(policy["num_threads"]))
                if torch.get_num_interop_threads() != int(policy["num_interop_threads"]):
                    torch.set_num_interop_threads(int(policy["num_interop_threads"]))
                if not torch.cuda.is_available():
                    raise ValueError("This probe requires the original CUDA environment")
                bindings = bridge.load_runtime(source)
                sys.modules["runtime_bridge"] = bridge
                sys.path.insert(0, str(addon))
                evaluator = bridge.named(addon / "evaluation.py", "grid_probe_evaluation")
                supports = bridge.named(addon / "supports.py", "grid_probe_supports")
                report["evaluation_source_sha256"] = sha256(addon / "evaluation.py")
                runtime_output = output / "runtime"
                runtime_output.mkdir()
                runtime_args = bridge.runtime_arguments(source, seed=context["seed"], output=runtime_output)
                print("GRID PROBE: loading the original frozen frontend and banks; zero optimizer updates", flush=True)
                stage = "enter_frozen_runtime"
                with bindings.Runtime(runtime_args, runtime_output) as runtime:
                    bindings.original_train._formal_runtime(runtime, False)
                    train_names, val_names = bindings.original_train._names(runtime, False)
                    if train_names != original["train_names"] or val_names != original["val_names"]:
                        raise ValueError("Live split identities differ")
                    from probe import json_value
                    if json_value(runtime.protocol_info) != original["protocol_info"]:
                        raise ValueError("Live protocol differs from recorded protocol")
                    bindings.original_train._checked_verify(runtime)
                    factory = supports.SupportFactory(runtime, seed=context["seed"])
                    stage = "inspect_frozen_grid"
                    report.update(inspect_images(torch, runtime, factory, evaluator,
                                                 bindings.original_train, names, tuple(plan["arms"]), emit))
                    stage = "verify_frozen_inputs"
                    report["runtime_integrity"] = json_value(bindings.original_train._checked_verify(runtime))
                bindings.original_train._restored_guard(runtime)
                bridge.verify_ledgers(source)
                if bridge.check_package(addon) != plan["addon_release_sha256"]:
                    raise ValueError("Original addon changed during probe")
                if sha256(evidence["checkpoint_path"]) != evidence["checkpoint_sha256"]:
                    raise ValueError("Existing checkpoint changed during probe")
                report["diagnostic_complete"] = True
        except BaseException as exc:
            # No tensor or CUDA operations after a runtime exception.
            report.update(phase="diagnostic_failed", failure_stage=stage,
                          error=type(exc).__name__ + ": " + str(exc), last_marker=last_marker)
            (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(output / "summary.json", report)
    print("BEGIN GRID PROBABILITY PROBE", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print("END GRID PROBABILITY PROBE", flush=True)
    return 0 if report["diagnostic_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--scope", choices=("prefix", "candidate"), default="prefix")
    args = parser.parse_args()
    try:
        return run(args)
    except BaseException as exc:
        print("GRID PROBE STOP: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
