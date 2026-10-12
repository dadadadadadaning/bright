"""E0: immutable source reproduction and fresh-process prediction comparison.

Run once per process with --profile legacy/strict. Compare completed inference
with ``protocol compare --left RUN1 --right RUN2 --output NEW_DIRECTORY``.
The historical 1e-5 reproduction threshold is never configurable.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .numeric import configure_runtime, runtime_settings


FORMAT = "SRAC_HARDQUERY_PROTOCOL_V1"


def parser():
    from .io_utils import add_source_arguments
    p = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(p)
    p.add_argument("--scope", choices=("a1", "all"), default="all")
    p.add_argument("--save_probabilities", action="store_true",
                   help="Also store full-resolution float32 probabilities (large files)")
    return p


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def a1_probability(front, pre, post):
    """Call only A1; no SRAC partition, pooling, bank or Reader forward occurs."""
    _localization, pixel, _auxiliary = front.base(pre, post, return_aux=True)
    return pixel.float().softmax(1)


def reproduction_for_scope(actual, expected, scope, smoke, image_count):
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report
    if scope == "all":
        return reproduction_report(actual, expected, smoke, image_count)
    if scope != "a1":
        raise ValueError("Unknown reproduction scope")
    # Reuse the exact historical scalar/class comparison logic, but label this
    # explicitly as A1-only evidence. No invented results for unevaluated arms.
    repeated_actual = {name: actual["a1_pixel"] for name in ("a1_pixel", "direct_region", "j1", "j2", "j3")}
    repeated_expected = {name: expected["a1_pixel"] for name in repeated_actual}
    report = reproduction_report(repeated_actual, repeated_expected, smoke, image_count)
    if "checks" in report:
        report["checks"] = [row for row in report["checks"] if row["name"].startswith("a1_pixel/")]
    report["scope"] = "a1_only_not_a_joint_training_gate"
    return report


@torch.no_grad()
def run(args):
    # Lazy imports keep comparison and CPU contract tests independent of model
    # construction/dependencies; load_bundle retains every inherited gate.
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        load_bundle, fingerprint, input_fingerprints, diagnostic_code_fingerprint,
        seed_everything, full_image_loader, FrozenFrontCapture, parse_batch, event,
        select_ids, validate_selected, gather_support, ARMS, decoder_forward,
        upsample_probability, probability_prediction, confusion_update, metrics, digest,
    )
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from .io_utils import code_hash

    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    evidence_dir = output / "predictions"
    evidence_dir.mkdir()
    started = time.perf_counter()
    report = dict(format=FORMAT, complete=False, inference_complete=False,
                  full_validation_evidence=False, smoke=bool(args.smoke), image_count=0,
                  profile=args.profile, pool_mode=args.pool_mode, scope=args.scope,
                  arguments=vars(args), test_accessed=False, training=False,
                  legacy_caveat="Startup defaults with historical seed-only policy; not proof of undocumented historical environment equality")
    _save(output / "protocol.json", report)
    capture = None
    guard = None
    current_image = None
    phase = "configuration"
    try:
        overrides = {key: None if getattr(args, key, None) is None else bool(getattr(args, key))
                     for key in ("matmul_tf32", "cudnn_tf32", "deterministic")}
        configure_runtime(args.profile, **overrides)
        seed_everything(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        phase = "load_immutable_source"
        bundle = load_bundle(args, device)
        before = fingerprint(bundle)
        code_before = code_hash()
        numeric_before = runtime_settings()
        names = bundle["val_names"][:2] if args.smoke else bundle["val_names"]
        report.update(runtime_settings=numeric_before, source_sha256=code_before,
                      source_hashes=bundle["hashes"], source_head_hashes=bundle["head_hashes"],
                      reference_hashes=bundle["reference_hashes"], input_fingerprints=bundle["inputs_before"],
                      input_paths={key: str(path) for key, path in bundle["input_paths"].items()},
                      source_format=bundle["source_manifest"]["format"], selected_images=names,
                      numeric_overrides=overrides)
        _save(output / "protocol.json", report)
        data = full_image_loader(args.dataset, names, False, args.workers, args.seed + 2)
        if args.scope == "all":
            capture = FrozenFrontCapture(bundle["front"])
            guard = PoolGuard(output / "pool_debug", mode=args.pool_mode, capture="failure", check_reference=False)
        cms, per_event = {}, {}
        with guard if guard is not None else nullcontext():
            for batch in data:
                phase = "Val_inference"
                pre, post, _loc, target, image_names = parse_batch(batch, device)
                current_image = str(image_names[0])
                index = report["image_count"]
                if len(pre) != 1 or index >= len(names) or current_image != names[index]:
                    raise ValueError("Unexpected Val image/order/batch")
                selected = {}
                if args.scope == "a1":
                    probabilities = {"a1_pixel": a1_probability(bundle["front"], pre, post)}
                else:
                    guard.set_image(current_image)
                    result, pixel_feature = capture(pre, post)
                    query = result["tokens"][0]
                    bank = bundle["bank"]
                    event_name = event(current_image)
                    event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
                    selected = {"j2": select_ids(query, bank, event_id, -1, bundle["topk"]),
                                "j3": select_ids(query, bank, event_id, -1, bundle["topk"], projector=bundle["metric"])}
                    for ids in selected.values():
                        validate_selected(ids, bank, event_id, -1)
                    supports = {arm: gather_support(ids, bank) for arm, ids in selected.items()}
                    probabilities = {"a1_pixel": result["old_pixel"].float().softmax(1),
                                     "direct_region": result["prob"]}
                    for arm in ARMS:
                        decoded = decoder_forward(bundle["decoders"][arm], arm, pixel_feature,
                                                  result, supports["j2"], supports["j3"])
                        probabilities[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
                predictions = {name: probability_prediction(value, target.shape[-2:])
                               for name, value in probabilities.items()}
                arrays = {"target": target.detach().cpu().numpy().astype(np.int16)}
                image_cms = {}
                event_name = event(current_image)
                for name, prediction in predictions.items():
                    cm = torch.zeros(4, 4, dtype=torch.long)
                    confusion_update(cm, target, prediction)
                    cms.setdefault(name, torch.zeros_like(cm)).add_(cm)
                    per_event.setdefault(event_name, {}).setdefault(name, torch.zeros_like(cm)).add_(cm)
                    image_cms[name] = cm.tolist()
                    arrays["prediction_" + name] = prediction.detach().cpu().numpy().astype(np.uint8)
                    arrays["confusion_" + name] = cm.numpy()
                    if args.save_probabilities:
                        arrays["probability_" + name] = probabilities[name].float().cpu().numpy()
                for arm, ids in selected.items():
                    arrays["support_" + arm] = ids.cpu().numpy()
                prediction_path = evidence_dir / f"{index:04d}.npz"
                np.savez_compressed(prediction_path, **arrays)
                row = dict(image=current_image, event=event_name, confusion=image_cms,
                           prediction_file=prediction_path.relative_to(output).as_posix(),
                           prediction_sha256=digest(prediction_path))
                with (output / "images.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                report["image_count"] += 1
                if report["image_count"] % 10 == 0 or report["image_count"] == len(names):
                    print(f"E0 {args.profile}/{args.scope}: {report['image_count']}/{len(names)}", flush=True)
        phase = "final_reproduction_and_immutability"
        actual = {name: metrics(cm) for name, cm in cms.items()}
        repro = reproduction_for_scope(actual, bundle["reference"], args.scope, args.smoke, report["image_count"])
        gates = dict(frozen_evidence_unchanged=before == fingerprint(bundle),
                     input_files_unchanged=input_fingerprints(bundle["input_paths"]) == bundle["inputs_before"],
                     code_unchanged=code_before == code_hash() and bundle["code_before"] ==
                     diagnostic_code_fingerprint(bundle["source_manifest"]["format"]),
                     numerical_unchanged=numeric_before == runtime_settings())
        complete = report["image_count"] == len(names) and all(gates.values()) and repro["passed"] is not False
        report.update(complete=complete, inference_complete=report["image_count"] == len(names),
                      full_validation_evidence=bool(complete and not args.smoke and args.scope == "all"
                                                    and report["image_count"] == 450 and repro["passed"] is True),
                      metrics=actual, confusion={key: cm.tolist() for key, cm in cms.items()},
                      per_event={key: {arm: metrics(cm) for arm, cm in values.items()} for key, values in per_event.items()},
                      reproduction=repro, **gates)
        if guard is not None:
            report["pool_guard"] = guard.report()
            if not guard.restored or guard.failures:
                report.update(complete=False, full_validation_evidence=False)
        _save(output / "protocol.json", report)
        if not report["complete"]:
            raise RuntimeError("E0 reproduction/immutability failed; evidence preserved and 1e-5 threshold unchanged")
        return report
    except Exception as error:
        report.update(complete=False, full_validation_evidence=False,
                      failure=dict(type=type(error).__name__, message=str(error), phase=phase, image=current_image))
        raise
    finally:
        if capture is not None:
            capture.close()
        if guard is not None:
            report["pool_guard"] = guard.report()
        report["wall_seconds"] = time.perf_counter() - started
        _save(output / "protocol.json", report)


def _artifact(path, row):
    from .io_utils import digest
    root = Path(path).resolve()
    file = (root / row["prediction_file"]).resolve()
    if root not in file.parents or digest(file) != row["prediction_sha256"]:
        raise ValueError("Prediction artifact path/hash mismatch")
    return np.load(file, allow_pickle=False)


def compare_predictions(left, right):
    """Summarize flips and rank/set changes, including failed-reproduction runs."""
    roots = [Path(left), Path(right)]
    reports = [_read(path / "protocol.json") for path in roots]
    for report in reports:
        if report.get("format") != FORMAT or report.get("inference_complete") is not True:
            raise ValueError("Comparison requires completed inference, even if final reproduction failed")
        if not all(report.get(key) is True for key in (
            "input_files_unchanged", "code_unchanged", "frozen_evidence_unchanged", "numerical_unchanged")):
            raise ValueError("Cannot compare a run whose immutable evidence changed")
    for key in ("scope", "selected_images", "source_hashes", "source_head_hashes", "reference_hashes",
                "input_fingerprints", "source_sha256"):
        if reports[0].get(key) != reports[1].get(key):
            raise ValueError("Unmatched comparison source: " + key)
    dataset_roots, compute_devices = [], []
    for report in reports:
        arguments = report.get("arguments", {})
        if not arguments.get("dataset") or not arguments.get("device"):
            raise ValueError("Comparison requires the recorded dataset root and compute device")
        dataset_roots.append(Path(arguments["dataset"]).resolve())
        device = torch.device(arguments["device"])
        compute_devices.append(f"cuda:{device.index or 0}" if device.type == "cuda" else device.type)
    if dataset_roots[0] != dataset_roots[1]:
        raise ValueError("Dataset root differs between comparison runs; identical names/GT do not identify input images")
    rows = [[json.loads(line) for line in (path / "images.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            for path in roots]
    if any([row["image"] for row in values] != reports[0]["selected_images"] for values in rows):
        raise ValueError("Image evidence missing, duplicated or out of order")
    total, support_total, image_rows = {}, {}, []
    for row_left, row_right in zip(*rows):
        with _artifact(roots[0], row_left) as a, _artifact(roots[1], row_right) as b:
            if set(a.files) != set(b.files) or not np.array_equal(a["target"], b["target"]):
                raise ValueError("Prediction evidence schema/GT mismatch")
            valid = (a["target"] >= 0) & (a["target"] < 4)
            image_result, support_result = {}, {}
            for key in sorted(a.files):
                if key.startswith("prediction_"):
                    name = key.removeprefix("prediction_")
                    if a[key].shape != b[key].shape or a[key].shape != valid.shape:
                        raise ValueError("Prediction shape mismatch")
                    values = dict(pixels=int(valid.size), valid_pixels=int(valid.sum()),
                                  flips_all_pixels=int((a[key] != b[key]).sum()),
                                  flips_valid_pixels=int(((a[key] != b[key]) & valid).sum()))
                    image_result[name] = values
                    for field, value in values.items():
                        total.setdefault(name, {}).setdefault(field, 0)
                        total[name][field] += value
                elif key.startswith("support_"):
                    name = key.removeprefix("support_")
                    aa, bb = a[key], b[key]
                    if aa.shape != bb.shape or aa.ndim != 2:
                        raise ValueError("Support shape mismatch")
                    values = dict(queries=len(aa), slots=int(aa.size),
                                  changed_slots=int((aa != bb).sum()),
                                  queries_with_rank_change=int((aa != bb).any(-1).sum()),
                                  queries_with_set_change=sum(set(x[x >= 0].tolist()) != set(y[y >= 0].tolist())
                                                              for x, y in zip(aa, bb)))
                    support_result[name] = values
                    for field, value in values.items():
                        support_total.setdefault(name, {}).setdefault(field, 0)
                        support_total[name][field] += value
            image_rows.append(dict(image=row_left["image"], prediction_changes=image_result,
                                   support_changes=support_result))
    confusion_delta = {}
    for name in reports[0]["confusion"]:
        confusion_delta[name] = (np.asarray(reports[1]["confusion"][name], dtype=np.int64)
                                 - np.asarray(reports[0]["confusion"][name], dtype=np.int64)).tolist()
    runtime_diff = {key: dict(left=reports[0]["runtime_settings"].get(key), right=reports[1]["runtime_settings"].get(key))
                    for key in set(reports[0]["runtime_settings"]) | set(reports[1]["runtime_settings"])
                    if reports[0]["runtime_settings"].get(key) != reports[1]["runtime_settings"].get(key)}
    if compute_devices[0] != compute_devices[1]:
        runtime_diff["compute_device"] = dict(left=compute_devices[0], right=compute_devices[1])
    return dict(format="SRAC_HARDQUERY_PROTOCOL_COMPARISON_V1", complete=True,
                left=str(roots[0].resolve()), right=str(roots[1].resolve()), image_count=len(image_rows),
                runtime_differences=runtime_diff, pool_modes=[r["pool_mode"] for r in reports],
                reproduction=[r["reproduction"] for r in reports], metrics=[r["metrics"] for r in reports],
                prediction_changes=total, support_changes=support_total,
                confusion_right_minus_left=confusion_delta, images=image_rows,
                causal_caveat="Multiple differing runtime factors cannot identify a unique cause; use fresh-process single-factor contrasts")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        p = argparse.ArgumentParser("Compare fresh E0 process artifacts")
        p.add_argument("--left", required=True)
        p.add_argument("--right", required=True)
        p.add_argument("--output", required=True)
        args = p.parse_args(sys.argv[2:])
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        try:
            _save(output / "comparison.json", compare_predictions(args.left, args.right))
        except Exception as error:
            _save(output / "comparison.json", dict(complete=False, failure=str(error)))
            raise
    else:
        run(parser().parse_args())


if __name__ == "__main__":
    main()
