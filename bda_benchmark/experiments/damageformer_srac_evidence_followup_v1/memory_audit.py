"""Isolated Memory audit with pre-kernel evidence and truthful single/prefix gates.

Inherits immutable-artifact checks and intervention semantics without changing
any historical source files. Grouped pooling is explicit, never an error retry.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import torch

from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1 import memory_audit as parent
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
    save_json, read_json, seed_everything, configure_determinism, assert_determinism,
    execution_settings, load_bundle, fingerprint, digest, inventory, full_image_loader,
    FrozenFrontCapture, parse_batch, event, select_ids, validate_selected, gather_support,
    ARMS, decoder_forward, upsample_probability, random_matched_ids, shuffle_support_labels,
    probability_prediction, confusion_update, metrics, transition_counts, transition_report,
    region_targets, label_evidence, project_label_evidence, cohort_evidence,
    region_retrieval_report, add_nested_counts, distribution_inventory, eligible_mask,
    reproduction_report, input_fingerprints, diagnostic_code_fingerprint, resources,
)
from .native_pool import PoolGuard
from .cohorts import difficult_retrieval
from .provenance import code_hash, select_images, validate_failed_memory


def parser():
    p = parent.parser()
    p.description = __doc__
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--pool_mode", choices=("native", "grouped"), default="native")
    p.add_argument("--pool_capture", choices=("failure", "all"), default="failure")
    p.add_argument("--pool_check_reference", action="store_true")
    return p


def recheck_main():
    p = argparse.ArgumentParser(description="Replay a failed Memory image or exact prefix in a new run.")
    p.add_argument("--failed_run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prefix", action="store_true")
    p.add_argument("--pool_mode", choices=("native", "grouped"), default="native")
    p.add_argument("--pool_capture", choices=("failure", "all"), default="failure")
    p.add_argument("--pool_check_reference", action="store_true")
    args = argparse.Namespace(**validate_failed_memory(p.parse_args()))
    print("Recheck:", args.recheck_origin, flush=True)
    run(args)


@torch.no_grad()
def run(args):
    if args.workers < 0 or not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Invalid audit arguments")
    if len(args.intervention_arms) != len(set(args.intervention_arms)):
        raise ValueError("Do not repeat an intervention arm")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "summary.json", dict(complete=False, gates_passed=False, full_validation_evidence=False,
                                           full_val_evidence=False, smoke=args.smoke, test_accessed=False, training=False))
    capture = None
    pool_guard = None
    followup_hash = code_hash()
    device = None
    current_image = None
    current_phase = "initialization"
    image_count = 0
    started = time.perf_counter()
    try:
        seed_everything(args.seed)
        configure_determinism()
        # Explicit strict FP32 policy belongs only to this NEW audit.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        numerical_before = assert_determinism()
        current_phase = "artifact_loading"
        bundle = load_bundle(args, device)
        names, subset = select_images(bundle["val_names"], args.smoke, args.start_index, args.max_images)
        if args.pool_capture == 'all' and len(names) > 2:
            raise ValueError('Capture-all is only for one/two images; choose failure for full/prefix runs')
        pool_guard = PoolGuard(output / "pool_debug", mode=args.pool_mode, capture=args.pool_capture,
                               check_reference=args.pool_check_reference)
        pool_guard.__enter__()
        before = fingerprint(bundle)
        manifest = dict(format="SRAC_STAGE_MEMORY_DIAGNOSTIC_V1", arguments=vars(args),
                        numerical_policy=numerical_before, hashes=bundle["hashes"], head_hashes=bundle["head_hashes"],
                        resolved_paths=bundle["paths"], source_format=bundle["source_manifest"]["format"],
                        source_snapshot_protocol=bundle["source_manifest"].get("snapshot_protocol", False),
                        parent_run_complete=bundle["front_payload"].get("run_complete", False),
                        source_front_step=bundle["front_payload"]["step"], joint_step=bundle["joint_step"],
                        train_sha256=digest(args.train_list), val_sha256=digest(args.val_list),
                        prediction_cohort="GT Damaged and matched J1 wrong; GT diagnostic only after retrieval",
                        label_evidence="softmax(cosine or projected-cosine / temperature) weighted Train labels; NOT Reader weights/calibrated posterior",
                        random_control="same valid slots; uniform eligible Train rows without replacement per query",
                        off_control="same trained Memory decoder with all support invalid -> existing learned fallback; NOT retrained J1",
                        test_accessed=False, training=False, smoke=args.smoke)
        manifest.update(followup_source_sha256=followup_hash, selected_images=names,
                        diagnostic_subset=subset, pool_mode=args.pool_mode,
                        pool_capture=args.pool_capture, pool_reference_check=args.pool_check_reference)
        manifest["reference_hashes"] = bundle["reference_hashes"]
        manifest["diagnostic_source_sha256"] = bundle["code_before"]["memory_code_sha256"]
        manifest["code_fingerprints_before"] = bundle["code_before"]
        manifest["input_fingerprints_before"] = bundle["inputs_before"]
        manifest["all_input_paths"] = {name: str(path) for name, path in bundle["input_paths"].items()}
        save_json(output / "manifest.json", manifest)
        current_phase = "bank_inventory"
        save_json(output / "inventory.json", inventory(bundle, args.inventory_shards))
        # Full-shard inventory adds its read inputs to the immutable-file ledger.
        manifest["input_fingerprints_before"] = dict(bundle["inputs_before"])
        manifest["all_input_paths"] = {name: str(path) for name, path in bundle["input_paths"].items()}
        save_json(output / "manifest.json", manifest)
        data = full_image_loader(args.dataset, names, False, args.workers, args.seed + 2)
        bank, decoders = bundle["bank"], bundle["decoders"]
        capture = FrozenFrontCapture(bundle["front"])
        generator = torch.Generator().manual_seed(args.seed + 1729)
        cms, transitions, by_event, cohort_total, retrieval_total, eligibility = {}, {}, {}, {}, {}, {}
        event_transitions, event_cohorts, event_retrieval = {}, {}, {}
        difficult_retrieval_total, event_difficult_retrieval = {}, {}
        image_count = 0
        current_phase = "Val_inference"
        for batch in data:
            current_phase = 'batch_transfer'
            pre, post, _loc, target, image_names = parse_batch(batch, device)
            image_name = str(image_names[0])
            current_image = image_name
            if image_name != names[image_count] or len(pre) != 1:
                raise ValueError("Unexpected image/batch in Val audit")
            pool_guard.set_image(image_name)
            current_phase = 'native_capture'
            result, pixel_feature = capture(pre, post)
            frozen_inputs = {"F": pixel_feature.detach().cpu().clone()}
            for name in ("q", "ids", "tokens", "B"):
                if name in result:
                    frozen_inputs[name] = result[name].detach().cpu().clone()
            query = result["tokens"][0]
            current_phase = 'retrieval_and_classification'
            event_name = event(image_name)
            event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
            image_id = -1  # Val is disjoint from the checked Train list.
            # No GT or query class is passed into either selector.
            selected = {"j2": select_ids(query, bank, event_id, image_id, bundle["topk"]),
                        "j3": select_ids(query, bank, event_id, image_id, bundle["topk"], projector=bundle["metric"])}
            for ids in selected.values():
                validate_selected(ids, bank, event_id, image_id)
            supports = {name: gather_support(ids, bank) for name, ids in selected.items()}
            probabilities = {"a1_pixel": result["old_pixel"].float().softmax(1), "direct_region": result["prob"]}
            for arm in ARMS:
                decoded = decoder_forward(decoders[arm], arm, pixel_feature, result, supports["j2"], supports["j3"])
                probabilities[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
            for arm in args.intervention_arms:
                original_support = supports[arm]
                random_ids = random_matched_ids(selected[arm], bank, event_id, image_id, generator)
                changed = {"random": gather_support(random_ids, bank)}
                token, distribution, valid, metadata = original_support
                changed["shuffle_labels"] = (token, shuffle_support_labels(distribution, valid, generator), valid, metadata)
                changed["off"] = (token, distribution, torch.zeros_like(valid), metadata)
                for mode, support in changed.items():
                    decoded = decoder_forward(decoders[arm], arm, pixel_feature, result, support, support)
                    probabilities[arm + "_" + mode] = upsample_probability(decoded["logits"], target.shape[-2:])
            predictions = {name: probability_prediction(p, target.shape[-2:]) for name, p in probabilities.items()}
            row_cms, row_transitions = {}, {}
            for name, prediction in predictions.items():
                cm = torch.zeros(4, 4, dtype=torch.long)
                confusion_update(cm, target, prediction)
                cms.setdefault(name, torch.zeros_like(cm)).add_(cm)
                row_cms[name] = metrics(cm)
                by_event.setdefault(event_name, {}).setdefault(name, torch.zeros_like(cm)).add_(cm)
                if name != "j1":
                    trans = transition_counts(target, predictions["j1"], prediction)
                    transitions.setdefault("j1_to_" + name, torch.zeros_like(trans)).add_(trans)
                    event_transitions.setdefault(event_name, {}).setdefault("j1_to_" + name, torch.zeros_like(trans)).add_(trans)
                    row_transitions["j1_to_" + name] = transition_report(trans)
            # Intervention transitions use the same trained Reader as their reference.
            for arm in args.intervention_arms:
                for mode in ("random", "shuffle_labels", "off"):
                    key = arm + "_to_" + arm + "_" + mode
                    trans = transition_counts(target, predictions[arm], predictions[arm + "_" + mode])
                    transitions.setdefault(key, torch.zeros_like(trans)).add_(trans)
                    event_transitions.setdefault(event_name, {}).setdefault(key, torch.zeros_like(trans)).add_(trans)
                    row_transitions[key] = transition_report(trans)
            # Only now use Val GT for query-label and difficult-cohort diagnostics.
            current_phase = 'post_prediction_GT_diagnostics'
            query_distribution, query_mass = region_targets(target, result)
            row_cohort, row_retrieval, row_difficult_retrieval = {}, {}, {}
            for arm in ("j2", "j3"):
                evidence, supported, _weights = label_evidence(query, selected[arm], bank, args.temperature,
                                                               bundle["metric"] if arm == "j3" else None)
                probability, mass, evidence_pred = project_label_evidence(
                    evidence, supported, result["q"], result["ids"], result["shape"], target.shape[-2:])
                row_cohort[arm] = cohort_evidence(target, predictions["j1"], predictions[arm], evidence_pred, mass, probability)
                row_retrieval[arm] = region_retrieval_report(selected[arm], bank, query_distribution[0], query_mass[0])
                row_difficult_retrieval[arm] = difficult_retrieval(
                    target, predictions['j1'], selected[arm], bank,
                    result['q'], result['ids'], result['shape'])
            add_nested_counts(difficult_retrieval_total, row_difficult_retrieval)
            add_nested_counts(event_difficult_retrieval.setdefault(event_name, {}), row_difficult_retrieval)
            add_nested_counts(cohort_total, row_cohort)
            add_nested_counts(retrieval_total, row_retrieval)
            add_nested_counts(event_cohorts.setdefault(event_name, {}), row_cohort)
            add_nested_counts(event_retrieval.setdefault(event_name, {}), row_retrieval)
            eligibility.setdefault(event_name, distribution_inventory(bank["distribution"], eligible_mask(bank, event_id, image_id)))
            for name, previous in frozen_inputs.items():
                actual = pixel_feature if name == "F" else result[name]
                actual = actual.detach().cpu().contiguous()
                if not torch.equal(previous.contiguous().view(torch.uint8), actual.view(torch.uint8)):
                    raise RuntimeError("Captured evidence mutated: " + name)
            image_count += 1
            with (output / "images.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(image=image_name, event=event_name, metrics=row_cms,
                                              transitions=row_transitions, difficult_cohorts=row_cohort,
                                              retrieval=row_retrieval,
                                              difficult_retrieval=row_difficult_retrieval), allow_nan=False) + "\n")
            if image_count % 10 == 0 or image_count == len(names):
                print(f"Memory stage audit {image_count}/{len(names)}", flush=True)
            current_image = None
            current_phase = 'await_next_batch'
        actual = {name: metrics(cm) for name, cm in cms.items()}
        current_phase = "final_gates"
        reproduction = (dict(passed=None, reason="Diagnostic subset: not full-Val reproduction", tolerance=1e-5)
                        if subset else reproduction_report(actual, bundle["reference"], False, image_count))
        after = fingerprint(bundle)
        immutable = before == after
        inputs_after = input_fingerprints(bundle["input_paths"])
        code_after = diagnostic_code_fingerprint(bundle["source_manifest"]["format"])
        numerical_after = execution_settings()
        numerical_error = None
        try:
            assert_determinism()
        except Exception as error:
            numerical_error = str(error)
        files_unchanged = inputs_after == bundle["inputs_before"]
        code_unchanged = code_after == bundle["code_before"] and code_hash() == followup_hash
        numerical_unchanged = numerical_after == numerical_before and numerical_error is None
        passed = (immutable and files_unchanged and code_unchanged and numerical_unchanged
                  and reproduction["passed"] is not False and image_count == len(names))
        full = passed and not subset and image_count == 450 and reproduction["passed"] is True
        report = dict(complete=passed, gates_passed=passed, full_val_evidence=full, full_validation_evidence=full,
                      image_count=image_count, metrics=actual, diagnostic_subset=subset,
                      followup_source_sha256=followup_hash, pool_guard=pool_guard.report(),
                      transitions={name: transition_report(value) for name, value in transitions.items()},
                      per_event={e: {name: metrics(cm) for name, cm in rows.items()} for e, rows in by_event.items()},
                      per_event_transitions={e: {name: transition_report(t) for name, t in rows.items()} for e, rows in event_transitions.items()},
                      per_event_difficult_cohorts=event_cohorts, per_event_retrieval=event_retrieval,
                      difficult_cohorts=cohort_total, retrieval=retrieval_total, eligible_bank_by_query_event=eligibility,
                      difficult_retrieval=difficult_retrieval_total,
                      per_event_difficult_retrieval=event_difficult_retrieval,
                      reproduction=reproduction, frozen_evidence_unchanged=immutable,
                      fingerprints_before=before, fingerprints_after=after,
                      input_fingerprints_before=bundle["inputs_before"], input_fingerprints_after=inputs_after,
                      input_files_unchanged=files_unchanged, code_fingerprints_before=bundle["code_before"],
                      code_fingerprints_after=code_after, code_unchanged=code_unchanged,
                      numerical_before=numerical_before, numerical_after=numerical_after,
                      numerical_unchanged=numerical_unchanged, numerical_error=numerical_error,
                      resources=resources(started, device),
                      interpretation_limits=["No training: interventions test dependence, not retrained causal benefit",
                                             "Hit@K is at least one same-dominant-class neighbor, not retrieval recall",
                                             "Soft label evidence is diagnostic, not actual Reader attention",
                                             "Any supported pixel can still have mostly unsupported Q mass"],
                      smoke=args.smoke, test_accessed=False, training=False)
        save_json(output / "audit.json", report)
        save_json(output / "summary.json", {key: report[key] for key in
                  ("complete", "gates_passed", "full_validation_evidence", "full_val_evidence", "image_count", "reproduction",
                   "frozen_evidence_unchanged", "input_files_unchanged", "code_unchanged", "numerical_unchanged",
                   "resources", "smoke", "test_accessed", "training", "diagnostic_subset",
                   "followup_source_sha256", "pool_guard")})
        if not passed:
            raise RuntimeError("Evidence saved, but reproduction/immutability gate FAILED; do not interpret intervention gains")
        print({name: value["valid_miou"] for name, value in actual.items()}, flush=True)
        print("MEMORY STAGE AUDIT COMPLETE. No training. No bank changes. No Test.", flush=True)
    except Exception as error:
        summary = read_json(output / "summary.json")
        failure = dict(type=type(error).__name__, message=str(error), image=current_image,
                       phase=current_phase, images_completed=image_count)
        if "failure" in summary:
            summary.setdefault("subsequent_failures", []).append(failure)
        else:
            summary["failure"] = failure
        summary.update(complete=False, gates_passed=False, full_validation_evidence=False,
                       full_val_evidence=False, resources=resources(started, device),
                       followup_source_sha256=followup_hash,
                       pool_guard=pool_guard.report() if pool_guard is not None else None)
        save_json(output / "summary.json", summary)
        raise
    finally:
        try:
            if capture is not None:
                capture.close()
        finally:
            if pool_guard is not None:
                pool_guard.__exit__(None, None, None)
                for name in ('summary.json', 'audit.json'):
                    path = output / name
                    if path.is_file():
                        record = read_json(path)
                        record['pool_guard'] = pool_guard.report()
                        if not pool_guard.report()['restored']:
                            record.update(complete=False, gates_passed=False,
                                          full_validation_evidence=False, full_val_evidence=False)
                        save_json(path, record)
                if not pool_guard.report()['restored']:
                    raise RuntimeError('Pool aliases were not restored; audit invalid')


if __name__ == "__main__":
    run(parser().parse_args())

