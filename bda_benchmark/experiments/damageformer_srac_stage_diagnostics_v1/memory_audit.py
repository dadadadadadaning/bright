"""Read-only J1-error -> bank -> retrieval -> Reader audit. No fitting or Test."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import time
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import JointContextDecoder
from bda_benchmark.experiments.damageformer_srac_joint123_v1.joint_runtime import (
    bank_fingerprint, decoder_forward, upsample_probability,
)
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import module_fingerprint
from bda_benchmark.experiments.damageformer_srac_joint123_v1.train import full_image_loader
from bda_benchmark.experiments.damageformer_srac_partition_probe_v1.numeric_runtime import (
    assert_determinism, configure_determinism, execution_settings,
)
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import gather_support, select_ids
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    confusion_update, digest, event, lists, parse_batch, region_targets, save_json, seed_everything,
)
from .memory_core import (
    add_nested_counts, cohort_evidence, distribution_inventory, eligible_mask, label_evidence,
    project_label_evidence, random_matched_ids, region_retrieval_report, reproduction_report,
    shuffle_support_labels, transition_counts, transition_report, validate_selected,
)


FORMATS = {"SRAC_JOINT123_V1": "damageformer_srac_joint123_v1",
           "SRAC_SNAPSHOT_JOINT123_V1": "damageformer_srac_snapshot_joint123_v1"}
ARMS = ("j1", "j2", "j3")


def parser():
    p = argparse.ArgumentParser(__doc__)
    for key in ("joint_run", "a1_checkpoint", "dataset", "train_list", "val_list", "output"):
        p.add_argument("--" + key, required=True)
    for key in ("checkpoint", "cache", "metric_checkpoint"):
        p.add_argument("--" + key, help="Optional moved-artifact path; exact SHA256 must still match")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Fixed diagnostic label-evidence temperature, not Reader attention")
    p.add_argument("--intervention_arms", nargs="*", choices=("j2", "j3"), default=["j2", "j3"])
    p.add_argument("--inventory_shards", action="store_true",
                   help="Also hash/read all Train shards and quantify selection losses; no GPU extraction")
    p.add_argument("--smoke", action="store_true", help="First two Val images using completed formal artifacts")
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_fingerprints(paths):
    return {name: digest(path) for name, path in paths.items()}


def diagnostic_code_fingerprint(source_format):
    common = importlib.import_module("bda_benchmark.experiments." + FORMATS[source_format] + ".common")
    inherited = common.source_hash()
    local = hashlib.sha256()
    for name in ("memory_core.py", "memory_audit.py"):
        local.update(name.encode())
        local.update((Path(__file__).parent / name).read_bytes())
    # Also record helpers used by the new audit outside the archived J source tree.
    for package, name in (("damageformer_srac_partition_probe_v1", "numeric_runtime.py"),
                          ("damageformer_srac_cause_audit_v1", "support.py")):
        path = Path(__file__).parent.parent / package / name
        local.update((package + '/' + name).encode())
        local.update(path.read_bytes())
    return dict(memory_code_sha256=local.hexdigest(), inherited_source_sha256=inherited,
                protocol=source_format)


def resources(started, device):
    result = dict(wall_seconds=time.perf_counter() - started,
                  cuda_peak_allocated_bytes=None, cuda_peak_reserved_bytes=None)
    if device is not None and device.type == "cuda" and torch.cuda.is_initialized():
        try:
            result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        except Exception as error:
            # A failed CUDA context must not mask the original audit failure.
            result["cuda_statistics_error"] = str(error)
    return result


def validate_joint_header(manifest, summary):
    if manifest.get("format") not in FORMATS:
        raise ValueError("Only audited J1/J2/J3 and explicitly labelled snapshot J1/J2/J3 are supported")
    for record in (manifest, summary):
        if record.get("smoke") is not False or record.get("probe") is not False or record.get("test_accessed") is not False:
            raise ValueError("Joint source must be formal, non-probe, Train/Val only")
    if summary.get("complete") is not True or summary.get("run_complete") is not True:
        raise ValueError("Joint-head training is not complete")
    if summary.get("frozen_fingerprints_unchanged") is not True:
        raise ValueError("Joint training did not certify frozen evidence")
    snapshot = manifest["format"] == "SRAC_SNAPSHOT_JOINT123_V1"
    if snapshot and (manifest.get("snapshot_protocol") is not True or summary.get("snapshot_protocol") is not True):
        raise ValueError("Snapshot protocol must be explicit; never relabel unfinished parent as formal-complete")
    budget = int(manifest.get("arguments", {}).get("steps", 0))
    if budget != 5000 or any(summary.get("train_image_counts", {}).get(arm) != budget for arm in ARMS):
        raise ValueError("Expected completed, equal 5000-update J1/J2/J3 heads")
    return budget


def validate_head(payload, manifest, arm, budget):
    if payload.get("format") != manifest["format"] or payload.get("arm") != arm:
        raise ValueError("Wrong joint head format/arm")
    if (payload.get("step") != budget or payload.get("completed_steps") != budget
            or payload.get("run_complete") is not True or payload.get("smoke") is not False
            or payload.get("probe") is not False or payload.get("test_accessed") is not False):
        raise ValueError("Use matching completed _last heads, not differently selected best/probe weights")
    for key in ("source_hash", "front_sha256", "a1_sha256", "cache_manifest_sha256", "bank_sha256", "metric_sha256"):
        if not manifest.get(key) or payload.get(key) != manifest[key]:
            raise ValueError("Joint head provenance mismatch: " + key)
    if payload.get("arguments") != manifest.get("arguments"):
        raise ValueError("Joint head training arguments differ from manifest")


def load_bundle(args, device):
    run = Path(args.joint_run)
    reference_paths = {name: run / name for name in ("manifest.json", "summary.json", "validation.jsonl")}
    reference_before = input_fingerprints(reference_paths)
    manifest, summary = read_json(run / "manifest.json"), read_json(run / "summary.json")
    budget = validate_joint_header(manifest, summary)
    common = importlib.import_module("bda_benchmark.experiments." + FORMATS[manifest["format"]] + ".common")
    if manifest.get("source_hash") != common.source_hash():
        raise ValueError("Joint source changed; restore archived matching source rather than weakening this gate")
    original = manifest["arguments"]
    paths = {key: Path(getattr(args, key, None) or original[key]) for key in ("checkpoint", "cache", "metric_checkpoint")}
    input_paths = dict(front=paths["checkpoint"], a1=Path(args.a1_checkpoint),
                       cache_manifest=paths["cache"] / "manifest.json", bank=paths["cache"] / "bank.pt",
                       metric=paths["metric_checkpoint"], train_list=Path(args.train_list), val_list=Path(args.val_list),
                       **{arm: run / f"{arm}_last.pth" for arm in ARMS},
                       **{"joint_" + name: path for name, path in reference_paths.items()})
    if manifest["format"] == "SRAC_SNAPSHOT_JOINT123_V1":
        input_paths["snapshot_manifest"] = paths["checkpoint"].parent / "snapshot.json"
    inputs_before = input_fingerprints(input_paths)
    if any(inputs_before["joint_" + name] != value for name, value in reference_before.items()):
        raise ValueError("Joint reference files changed while loading")
    code_before = diagnostic_code_fingerprint(manifest["format"])
    hashes = dict(front_sha256=digest(paths["checkpoint"]), a1_sha256=digest(args.a1_checkpoint),
                  cache_manifest_sha256=digest(paths["cache"] / "manifest.json"),
                  bank_sha256=digest(paths["cache"] / "bank.pt"), metric_sha256=digest(paths["metric_checkpoint"]))
    for key, actual in hashes.items():
        if manifest.get(key) != actual:
            raise ValueError("Artifact override/source mismatch: " + key)
    train_names, val_names = lists(args.train_list, args.val_list)
    front, front_payload = common.load_front(paths["checkpoint"], device, args.a1_checkpoint)
    front.eval().requires_grad_(False)
    root, cache, bank_cpu = common.read_cache(paths["cache"], paths["checkpoint"])
    if cache.get("smoke") is not False or cache.get("test_accessed") is not False or len(cache.get("shards", [])) != 2918:
        raise ValueError("Expected full, non-smoke Train2918 bank")
    for key, path in (("train_sha256", args.train_list), ("val_sha256", args.val_list)):
        if front_payload.get(key) != digest(path) or cache.get(key) != digest(path):
            raise ValueError("Train/Val split hash mismatch")
    if bank_cpu.get("checkpoint_sha256") != hashes["front_sha256"]:
        raise ValueError("Bank payload belongs to different front")
    expected_events = sorted({event(name) for name in train_names})
    if bank_cpu.get("event_names") != expected_events or cache.get("event_names") != expected_events:
        raise ValueError("Bank Train event identities disagree")
    images, events = bank_cpu["images"].long(), bank_cpu["events"].long()
    if bool(((images < 0) | (images >= len(train_names))).any()):
        raise ValueError("Bank image ID outside Train")
    if bool(((events < 0) | (events >= len(expected_events))).any()):
        raise ValueError("Bank event ID outside Train")
    for image_id in images.unique().tolist():
        expected = expected_events.index(event(train_names[image_id]))
        if not bool((events[images == image_id] == expected).all()):
            raise ValueError("Bank image/event provenance conflict")
    labels = bank_cpu["distribution"].float()
    if not torch.isfinite(labels).all() or (labels < 0).any() or not torch.allclose(labels.sum(-1), torch.ones(len(labels)), atol=1e-5, rtol=0):
        raise ValueError("Bank label distributions are invalid")
    bank = common.device_bank(bank_cpu, device)
    metric, metric_payload = common.load_metric(paths["metric_checkpoint"], cache, paths["checkpoint"], device)
    if metric_payload.get("smoke") is not False:
        raise ValueError("A completed formal metric is required")
    decoders, head_hashes = {}, {}
    for arm in ARMS:
        head_path = run / f"{arm}_last.pth"
        head = torch.load(head_path, map_location="cpu", weights_only=False)
        validate_head(head, manifest, arm, budget)
        decoder = JointContextDecoder(front.base.clf_2, dim=cache["config"]["dim"],
                                      num_events=len(cache["event_names"]), slots=original["topk"]).to(device)
        decoder.load_state_dict(head["model"], strict=True)
        decoders[arm] = decoder.eval().requires_grad_(False)
        head_hashes[arm] = digest(head_path)
    rows = [json.loads(line) for line in (run / "validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    final = [row for row in rows if row.get("step") == budget]
    if len(final) != 1 or final[0].get("image_count") != 450:
        raise ValueError("Need unique matching final-step full-Val reference")
    if final[0].get("reproduction", {}).get("passed") is False:
        raise ValueError("Source run baseline reproduction failed")
    if inputs_before != input_fingerprints(input_paths) or code_before != diagnostic_code_fingerprint(manifest["format"]):
        raise ValueError("Input files or source code changed during artifact loading")
    return dict(front=front, front_payload=front_payload, cache_root=root, cache=cache, bank=bank,
                bank_cpu=bank_cpu, metric=metric, decoders=decoders, topk=original["topk"],
                train_names=train_names, val_names=val_names, reference=final[0]["metrics"],
                source_manifest=manifest, source_summary=summary, paths={k: str(v) for k, v in paths.items()},
                hashes=hashes, head_hashes=head_hashes, joint_step=budget,
                reference_hashes=reference_before, input_paths=input_paths,
                inputs_before=inputs_before, code_before=code_before)


def quality_inventory(distribution, mass=None):
    """Diagnostic bins, not tuning thresholds or true building-size measurements."""
    d = distribution.detach().cpu().float()
    purity = d.max(-1).values
    result = dict(purity={
        "lt_080": distribution_inventory(d, purity < .8),
        "080_095": distribution_inventory(d, (purity >= .8) & (purity < .95)),
        "ge_095": distribution_inventory(d, purity >= .95)})
    if mass is not None:
        mass = mass.detach().cpu().float()
        if mass.shape != purity.shape or not torch.isfinite(mass).all() or (mass < 0).any():
            raise ValueError("Invalid Q region mass")
        result["q_mass"] = {
            "lt_16_cells": distribution_inventory(d, mass < 16),
            "16_64_cells": distribution_inventory(d, (mass >= 16) & (mass < 64)),
            "ge_64_cells": distribution_inventory(d, mass >= 64)}
    return result


def inventory(bundle, scan_shards):
    bank, cache = bundle["bank_cpu"], bundle["cache"]
    result = dict(bank=distribution_inventory(bank["distribution"]), by_event={},
                  train_shards_scanned=False, train_labels_only=True,
                  soft_label_sum_units="sum of per-region label proportions, NOT pixels or building instances")
    result["bank_quality"] = quality_inventory(bank["distribution"])
    result["bank_quality_by_event"] = {}
    result["bin_definition"] = dict(purity="maximum Train GT class fraction; [0,.8), [.8,.95), [.95,1]",
                                    mass="Q assignment mass in feature-grid cells: <16, 16-64, >=64; NOT building/object area")
    for i, name in enumerate(bank["event_names"]):
        result["by_event"][name] = distribution_inventory(bank["distribution"], bank["events"] == i)
        result["bank_quality_by_event"][name] = quality_inventory(bank["distribution"][bank["events"] == i])
    if not scan_shards:
        result["missing_check"] = "All-Train pre-selection coverage unavailable until --inventory_shards"
        result["bank_mass_missing"] = "Bank does not store Q mass; requires matching source shards, not guessed geometry"
        return result
    totals, event_totals = {}, {}
    bank_mass = torch.full((len(bank["tokens"]),), float("nan"))
    for index, record in enumerate(cache["shards"]):
        path = (bundle["cache_root"] / record["path"]).resolve()
        if not path.is_relative_to(bundle["cache_root"].resolve()) or digest(path) != record["sha256"]:
            raise ValueError("Invalid Train shard path/hash: " + record["path"])
        bundle["input_paths"][f"train_shard_{index:05d}"] = path
        bundle["inputs_before"][f"train_shard_{index:05d}"] = record["sha256"]
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard["name"] != bundle["train_names"][index] or int(shard["image"]) != index:
            raise ValueError("Train shard identity/order mismatch")
        event_id = bank["event_names"].index(event(shard["name"]))
        if int(shard["event"]) != event_id:
            raise ValueError("Train shard event mismatch")
        d = shard["distribution"].float()
        valid = (shard["weights"] > 0) & (shard["mass"] > 1e-6)
        eligible = valid & (d.max(-1).values >= cache["min_purity"])
        row = dict(valid_regions=distribution_inventory(d, valid), eligible_regions=distribution_inventory(d, eligible))
        row["valid_quality"] = quality_inventory(d[valid], shard["mass"][valid])
        row["eligible_quality"] = quality_inventory(d[eligible], shard["mass"][eligible])
        row["feature_grid_label_mass"] = dict(
            valid=(d[valid].double() * shard["weights"][valid, None].double()).sum(0).tolist(),
            eligible=(d[eligible].double() * shard["weights"][eligible, None].double()).sum(0).tolist())
        add_nested_counts(totals, row)
        add_nested_counts(event_totals.setdefault(bank["event_names"][event_id], {}), row)
        # Confirm bank payload really equals its declared source rows.
        mask = bank["images"] == index
        region_ids = bank["regions"][mask].long()
        if len(region_ids):
            if bool(((region_ids < 0) | (region_ids >= len(d))).any()) or not bool(eligible[region_ids].all()):
                raise ValueError("Bank selected ineligible/out-of-range source region")
            if not torch.equal(d[region_ids], bank["distribution"][mask].float()):
                raise ValueError("Bank soft labels do not match hashed Train shard")
            if not torch.equal(shard["tokens"][region_ids], bank["tokens"][mask]):
                raise ValueError("Bank tokens do not match hashed Train shard")
            bank_mass[mask] = shard["mass"][region_ids].float()
        if (index + 1) % 250 == 0:
            print(f"Train inventory {index + 1}/{len(cache['shards'])}", flush=True)
    result.update(train_shards_scanned=True, all_train=totals, all_train_by_event=event_totals,
                  matched_source_shards=True)
    result["bank_quality"] = quality_inventory(bank["distribution"], bank_mass)
    for i, name in enumerate(bank["event_names"]):
        mask = bank["events"] == i
        result["bank_quality_by_event"][name] = quality_inventory(bank["distribution"][mask], bank_mass[mask])
    return result


def fingerprint(bundle):
    return dict(front=module_fingerprint(bundle["front"]), metric=module_fingerprint(bundle["metric"]),
                bank=bank_fingerprint(bundle["bank"]),
                **{arm: module_fingerprint(decoder) for arm, decoder in bundle["decoders"].items()})


def probability_prediction(probability, size):
    if probability.ndim != 4 or probability.shape[1] != 4 or tuple(probability.shape[-2:]) != tuple(size):
        raise ValueError("Output probability grid mismatch")
    if not torch.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Invalid probabilities")
    if not torch.allclose(probability.sum(1), torch.ones_like(probability[:, 0]), atol=2e-5, rtol=0):
        raise ValueError("Probabilities are not normalized")
    return probability.argmax(1)


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
        names = bundle["val_names"][:2] if args.smoke else bundle["val_names"]
        data = full_image_loader(args.dataset, names, False, args.workers, args.seed + 2)
        bank, decoders = bundle["bank"], bundle["decoders"]
        capture = FrozenFrontCapture(bundle["front"])
        generator = torch.Generator().manual_seed(args.seed + 1729)
        cms, transitions, by_event, cohort_total, retrieval_total, eligibility = {}, {}, {}, {}, {}, {}
        event_transitions, event_cohorts, event_retrieval = {}, {}, {}
        image_count = 0
        current_phase = "Val_inference"
        for batch in data:
            pre, post, _loc, target, image_names = parse_batch(batch, device)
            image_name = str(image_names[0])
            current_image = image_name
            if image_name != names[image_count] or len(pre) != 1:
                raise ValueError("Unexpected image/batch in Val audit")
            result, pixel_feature = capture(pre, post)
            frozen_inputs = {"F": pixel_feature.detach().cpu().clone()}
            for name in ("q", "ids", "tokens", "B"):
                if name in result:
                    frozen_inputs[name] = result[name].detach().cpu().clone()
            query = result["tokens"][0]
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
            query_distribution, query_mass = region_targets(target, result)
            row_cohort, row_retrieval = {}, {}
            for arm in ("j2", "j3"):
                evidence, supported, _weights = label_evidence(query, selected[arm], bank, args.temperature,
                                                               bundle["metric"] if arm == "j3" else None)
                probability, mass, evidence_pred = project_label_evidence(
                    evidence, supported, result["q"], result["ids"], result["shape"], target.shape[-2:])
                row_cohort[arm] = cohort_evidence(target, predictions["j1"], predictions[arm], evidence_pred, mass, probability)
                row_retrieval[arm] = region_retrieval_report(selected[arm], bank, query_distribution[0], query_mass[0])
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
                                              retrieval=row_retrieval), allow_nan=False) + "\n")
            if image_count % 10 == 0 or image_count == len(names):
                print(f"Memory stage audit {image_count}/{len(names)}", flush=True)
        actual = {name: metrics(cm) for name, cm in cms.items()}
        current_phase = "final_gates"
        reproduction = reproduction_report(actual, bundle["reference"], args.smoke, image_count)
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
        code_unchanged = code_after == bundle["code_before"]
        numerical_unchanged = numerical_after == numerical_before and numerical_error is None
        passed = (immutable and files_unchanged and code_unchanged and numerical_unchanged
                  and reproduction["passed"] is not False and image_count == len(names))
        full = passed and not args.smoke
        report = dict(complete=passed, gates_passed=passed, full_val_evidence=full, full_validation_evidence=full,
                      image_count=image_count, metrics=actual,
                      transitions={name: transition_report(value) for name, value in transitions.items()},
                      per_event={e: {name: metrics(cm) for name, cm in rows.items()} for e, rows in by_event.items()},
                      per_event_transitions={e: {name: transition_report(t) for name, t in rows.items()} for e, rows in event_transitions.items()},
                      per_event_difficult_cohorts=event_cohorts, per_event_retrieval=event_retrieval,
                      difficult_cohorts=cohort_total, retrieval=retrieval_total, eligible_bank_by_query_event=eligibility,
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
                   "resources", "smoke", "test_accessed", "training")})
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
                       full_val_evidence=False, resources=resources(started, device))
        save_json(output / "summary.json", summary)
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
