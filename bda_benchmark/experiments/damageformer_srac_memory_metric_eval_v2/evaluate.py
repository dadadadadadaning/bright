"""Evaluate new Train-only retrieval projectors with the original J3 Reader.

Historical A1, region, J1, J2 and J3 paths are replayed first in the original
order.  Only the Top-8 support IDs change for the two experimental J3 arms.
Validation labels are used for scoring and post-hoc retrieval diagnostics only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


FORMAT = "SRAC_MEMORY_METRIC_EVAL_V3"
BASE_ARMS = ("a1_pixel", "direct_region", "j1", "j2", "j3")
METRIC_ARMS = ("j3_j1_hard", "j3_mislead")
CONTROL_ARMS = ("j3_old_diverse",)
SUPPORT_ARMS = ("j3", *METRIC_ARMS, *CONTROL_ARMS)
ALL_ARMS = (*BASE_ARMS, "grid_probq", *METRIC_ARMS, *CONTROL_ARMS)
PAIRS = ("j1__j3", "j3__j3_j1_hard", "j3__j3_mislead",
         "j3_j1_hard__j3_mislead", "a1_pixel__grid_probq",
         "j3__j3_old_diverse", "j3_old_diverse__j3_j1_hard",
         "j3_old_diverse__j3_mislead")


def source_hash() -> str:
    """Bind this evaluator and its local native A1/Q and retrieval helpers."""
    root = Path(__file__).parent
    experiments = root.parent
    paths = [p for p in root.glob("*.py") if not p.name.startswith("test_")]
    paths += list(root.glob("*.sh"))
    sha = hashlib.sha256()
    for path in sorted(paths):
        sha.update(path.relative_to(experiments).as_posix().encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True,
                   help="Passed full Val450 legacy/native E0 directory or protocol.json")
    p.add_argument("--metrics", required=True,
                   help="Completed Train-only memory_metric_v2 output directory")
    p.add_argument("--output", required=True, help="Fresh Val output directory")
    p.add_argument("--limit", type=int, default=0,
                   help="First N images for debugging; 0 is formal Val450")
    return p


def _record_failure(path: Path, summary: dict, phase: str, image: str | None,
                    error: Exception) -> None:
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import write_json

    summary.update(complete=False, full_validation_evidence=False,
                   phase=phase, image=image,
                   error=f"{type(error).__name__}: {error}")
    write_json(path, summary)


def run(cli):
    import torch

    with torch.no_grad():
        return _run(cli)


def _run(cli):
    import numpy as np
    import torch

    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, load_verified_bundle, new_output, safe_child, verify_unchanged,
        write_json)
    from .native import (NativeA1LogitsCapture, native_grid_probq,
                         protocol_record, sum_nested_counts)
    from .retrieve import (
        add_support_diagnostics, final_support_diagnostics, select_diverse,
        posthoc_query_label_compatibility, support_diagnostics,
        support_provenance_changes)
    from bda_benchmark.experiments.damageformer_srac_memory_metric_v2.train import (
        experiment_source_hash, load_metric_checkpoint)
    from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import Accumulator
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, decoder_forward, event, full_image_loader,
        gather_support, parse_batch, probability_prediction, select_ids,
        upsample_probability, validate_selected)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import (
        region_retrieval_report, reproduction_report)
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
        region_targets)

    if cli.limit < 0 or cli.limit > 450:
        raise ValueError("--limit must be 0..450")
    protocol_path, protocol = protocol_record(cli.protocol)
    output = new_output(cli.output)
    summary_path = output / "summary.json"
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   image_count=0, phase="load", test_accessed=False,
                   training=False)
    write_json(summary_path, summary)
    started = time.perf_counter()
    own_hash = source_hash()
    capture = native_capture = guard = None
    phase, current_image = "load", None
    try:
        args = argparse.Namespace(**protocol["arguments"])
        args.protocol, args.output = str(protocol_path), str(output)
        bundle = load_verified_bundle(args)
        if bundle["topk"] != 8:
            raise ValueError("Metric evaluation requires the unchanged Top-8 J3 Reader")
        names = bundle["val_names"]
        if names != protocol["selected_images"] or len(names) != 450:
            raise ValueError("Val image list/order differs from passed E0")
        selected_names = names[:cli.limit] if cli.limit else names
        metric_root = Path(cli.metrics).resolve()
        metric_summary_path = metric_root / "summary.json"
        metric_manifest_path = metric_root / "manifest.json"
        metric_summary = json.loads(metric_summary_path.read_text(encoding="utf-8"))
        if (metric_summary.get("complete") is not True
                or metric_summary.get("smoke") is not False
                or metric_summary.get("train_only") is not True
                or metric_summary.get("test_accessed") is not False
                or metric_summary.get("protocol_sha256") != digest(protocol_path)):
            raise ValueError("New metric summary is incomplete or belongs to another E0")
        projectors = {}
        metric_files = {}
        for key in ("j1_hard", "mislead"):
            path = metric_root / f"metric_{key}_last.pth"
            expected_sha = metric_summary.get("checkpoint_sha256", {}).get(key)
            if expected_sha != digest(path):
                raise ValueError(f"New {key} metric SHA differs from its Train-only summary")
            projectors[key] = load_metric_checkpoint(path, bundle, key, smoke=False)
            metric_files[key] = path
            bundle["input_paths"][f"new_metric_{key}"] = path
            bundle["inputs_before"][f"new_metric_{key}"] = expected_sha
        summary_sha = digest(metric_summary_path)
        bundle["input_paths"]["new_metric_summary"] = metric_summary_path
        bundle["inputs_before"]["new_metric_summary"] = summary_sha
        manifest_sha = digest(metric_manifest_path)
        bundle["input_paths"]["new_metric_manifest"] = metric_manifest_path
        bundle["inputs_before"]["new_metric_manifest"] = manifest_sha
        metric_code_sha = experiment_source_hash()
        frozen_metrics = {key: tuple((name, tensor.detach().clone())
                                      for name, tensor in model.state_dict().items())
                          for key, model in projectors.items()}
        data = full_image_loader(args.dataset, selected_names, False,
                                 args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        native_capture = NativeA1LogitsCapture(bundle["front"])
        guard = PoolGuard(output / "pool_debug", mode=args.pool_mode,
                          capture="failure", check_reference=False)
        counts = Accumulator(arms=ALL_ARMS, pairs=PAIRS)
        support_totals = {arm: {} for arm in SUPPORT_ARMS}
        retrieval = {arm: {} for arm in SUPPORT_ARMS}
        soft_compatibility = {arm: {} for arm in SUPPORT_ARMS}
        source_changes = {arm: {} for arm in (*METRIC_ARMS, *CONTROL_ARMS)}
        diversity_relaxation = {arm: {} for arm in CONTROL_ARMS}
        native_a1_max_abs = 0.0
        artifacts = output / "predictions"
        artifacts.mkdir()
        phase = "Val_same_pass"
        with guard:
            with (output / "images.jsonl").open("x", encoding="utf-8") as journal:
                for index, batch in enumerate(data):
                    pre, post, _loc, target, image_names = parse_batch(
                        batch, bundle["device"])
                    current_image = str(image_names[0])
                    if (len(pre) != 1 or index >= len(selected_names)
                            or current_image != selected_names[index]):
                        raise ValueError("Val image/order/batch differs from E0")
                    guard.set_image(current_image)
                    native_capture.clear()
                    result, feature = capture(pre, post)
                    native_low_logits = native_capture.take()
                    bank = bundle["bank"]
                    event_name = event(current_image)
                    event_id = (bank["event_names"].index(event_name)
                                if event_name in bank["event_names"] else -1)
                    query = result["tokens"][0]
                    # Keep the E0 historical call order intact, including J2.
                    j2_ids = select_ids(query, bank, event_id, -1, 8)
                    j3_ids = select_ids(query, bank, event_id, -1, 8,
                                        projector=bundle["metric"])
                    validate_selected(j2_ids, bank, event_id, -1)
                    validate_selected(j3_ids, bank, event_id, -1)
                    j2_support = gather_support(j2_ids, bank)
                    j3_support = gather_support(j3_ids, bank)
                    probability = dict(a1_pixel=result["old_pixel"].float().softmax(1),
                                       direct_region=result["prob"])
                    for arm in ("j1", "j2", "j3"):
                        decoded = decoder_forward(bundle["decoders"][arm], arm,
                                                  feature, result, j2_support,
                                                  j3_support)
                        probability[arm] = upsample_probability(
                            decoded["logits"], target.shape[-2:])
                    grid_probq, native_replay = native_grid_probq(
                        result, feature, native_low_logits, target.shape[-2:])
                    probability["grid_probq"] = grid_probq
                    native_a1_max_abs = max(native_a1_max_abs,
                                            float(native_replay["max_abs"]))
                    if native_replay["passed"] is not True:
                        summary.update(native_a1_replay_failure=dict(
                            index=index, image=current_image, **native_replay))
                        write_json(summary_path, summary)
                        raise ValueError("Same-pass native A1 replay differs: " + current_image)
                    support_ids = {"j3": j3_ids}
                    # Preserve the bank-v2 control's order after the historical
                    # paths and native Q probe, before the new metric arms.
                    old_diverse_ids, old_relax = select_diverse(
                        query, bank, event_id, -1, 8, bundle["metric"])
                    support_ids["j3_old_diverse"] = old_diverse_ids
                    decoded = decoder_forward(bundle["decoders"]["j3"],
                                              "j3", feature, result,
                                              j2_support, gather_support(old_diverse_ids, bank))
                    probability["j3_old_diverse"] = upsample_probability(
                        decoded["logits"], target.shape[-2:])
                    sum_nested_counts(diversity_relaxation["j3_old_diverse"], old_relax)
                    for key, arm in (("j1_hard", "j3_j1_hard"),
                                     ("mislead", "j3_mislead")):
                        ids = select_ids(query, bank, event_id, -1, 8,
                                         projector=projectors[key])
                        validate_selected(ids, bank, event_id, -1)
                        support_ids[arm] = ids
                        decoded = decoder_forward(bundle["decoders"]["j3"],
                                                  "j3", feature, result,
                                                  j2_support, gather_support(ids, bank))
                        probability[arm] = upsample_probability(
                            decoded["logits"], target.shape[-2:])
                    predictions = {arm: probability_prediction(value,
                                       target.shape[-2:])
                                   for arm, value in probability.items()}
                    for arm, ids in support_ids.items():
                        add_support_diagnostics(support_totals[arm],
                                                support_diagnostics(ids, bank))
                    for arm in (*METRIC_ARMS, *CONTROL_ARMS):
                        sum_nested_counts(source_changes[arm],
                                          support_provenance_changes(
                                              j3_ids, bank, support_ids[arm], bank))
                    # Validation truth is first used here, after all support IDs
                    # and every model prediction have been frozen for this image.
                    distribution, weights = region_targets(target, result)
                    for arm, ids in support_ids.items():
                        sum_nested_counts(retrieval[arm], region_retrieval_report(
                            ids, bank, distribution[0], weights[0]))
                        row = posthoc_query_label_compatibility(
                            ids, bank, distribution[0], weights[0])
                        sum_nested_counts(soft_compatibility[arm], row)
                    image_cms = counts.add(current_image, event_name, target,
                                           predictions)
                    arrays = {"target": target.detach().cpu().numpy().astype(np.int16),
                              "support_j2": j2_ids.detach().cpu().numpy().astype(np.int32)}
                    for arm in ALL_ARMS:
                        arrays["prediction_" + arm] = (
                            predictions[arm].detach().cpu().numpy().astype(np.uint8))
                    for arm, ids in support_ids.items():
                        arrays["support_" + arm] = ids.detach().cpu().numpy().astype(np.int32)
                    artifact = artifacts / f"{index:04d}.npz"
                    np.savez_compressed(artifact, **arrays)
                    journal.write(json.dumps(dict(
                        index=index, image=current_image, event=event_name,
                        confusion=image_cms, native_a1_replay=native_replay,
                        prediction_file=artifact.relative_to(output).as_posix(),
                        prediction_sha256=digest(artifact)),
                        ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    summary["image_count"] = counts.image_count
                    summary["native_a1_replay_max_abs"] = native_a1_max_abs
                    if counts.image_count % 10 == 0 or counts.image_count == len(selected_names):
                        print(f"Memory metric Val {counts.image_count}/{len(selected_names)}",
                              flush=True)

        phase = "reproduction_and_integrity"
        rows = [json.loads(line) for line in (output / "images.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if (len(rows) != len(selected_names)
                or [row.get("image") for row in rows] != selected_names
                or [row.get("index") for row in rows] != list(range(len(selected_names)))):
            raise ValueError("Per-image Val evidence journal is incomplete or out of order")
        for row in rows:
            path = safe_child(output, row["prediction_file"])
            if digest(path) != row["prediction_sha256"]:
                raise ValueError("Saved Val prediction/support SHA mismatch")
            with np.load(path, allow_pickle=False) as saved:
                if "target" not in saved.files:
                    raise ValueError("Saved prediction artifact lacks target")
                for arm in ALL_ARMS:
                    key = "prediction_" + arm
                    if key not in saved.files or saved[key].shape != saved["target"].shape:
                        raise ValueError("Saved prediction shape mismatch: " + arm)
                for arm in SUPPORT_ARMS:
                    if "support_" + arm not in saved.files:
                        raise ValueError("Saved support IDs missing: " + arm)
        results = counts.finish()
        historical = {arm: results["metrics"][arm] for arm in BASE_ARMS}
        smoke = len(selected_names) != 450
        reproduction = reproduction_report(historical, bundle["reference"],
                                           smoke=smoke, image_count=len(selected_names))
        e0_comparison = reproduction_report(historical, protocol["metrics"],
                                            smoke=smoke, image_count=len(selected_names))
        immutable = verify_unchanged(bundle)
        immutable["new_metric_state"] = all(
            all(torch.equal(tensor, model.state_dict()[name])
                for name, tensor in frozen_metrics[key])
            for key, model in projectors.items())
        immutable["new_metric_training_code"] = experiment_source_hash() == metric_code_sha
        immutable["new_code"] = source_hash() == own_hash
        pool_ok = bool(guard.restored and not guard.failures)
        formal = bool(len(selected_names) == len(names) == 450
                      and reproduction.get("passed") is True
                      and all(immutable.values()) and pool_ok)
        report = dict(format=FORMAT, complete=formal,
                      full_validation_evidence=formal, **results,
                      historical_arms=list(BASE_ARMS),
                      metric_arms=list(METRIC_ARMS), control_arms=list(CONTROL_ARMS),
                      comparison_pairs=list(PAIRS),
                      support_diagnostics={arm: final_support_diagnostics(row)
                                           for arm, row in support_totals.items()},
                      retrieval=retrieval,
                      posthoc_region_soft_compatibility=soft_compatibility,
                      support_source_changes_vs_j3=source_changes,
                      diversity_relaxation=diversity_relaxation,
                      reproduction=reproduction,
                      e0_metric_comparison_informative_only=e0_comparison,
                      native_a1_replay=dict(same_pass_identity_gate=True,
                                            failed_count=0, max_abs=native_a1_max_abs),
                      immutable=immutable, pool_guard=guard.report(),
                      protocol_path=str(protocol_path.resolve()),
                      protocol_sha256=digest(protocol_path),
                      original_bank_sha256=bundle["hashes"]["bank_sha256"],
                      metric_summary_sha256=summary_sha,
                      metric_manifest_sha256=manifest_sha,
                      metric_checkpoint_sha256={key: digest(path)
                                                for key, path in metric_files.items()},
                      image_journal_sha256=digest(output / "images.jsonl"),
                      source_sha256=own_hash,
                      profile=protocol["profile"], pool_mode=protocol["pool_mode"],
                      test_accessed=False, training=False,
                      inference_rule="Same front/Q/original bank/J3 Reader; only projector and Top-8 differ",
                      gt_rule="Val truth is used after retrieval and predictions, for scoring only",
                      elapsed_seconds=time.perf_counter() - started)
        write_json(output / "audit.json", report)
        summary.update(complete=formal, full_validation_evidence=formal,
                       reproduction_passed=reproduction.get("passed"),
                       immutable=immutable, audit_file="audit.json",
                       phase="complete" if formal else "failed_formal_gate")
        write_json(summary_path, summary)
        if not smoke and not formal:
            raise RuntimeError("Memory metric Val reproduction/immutability failed; fixed 1e-5 gate unchanged")
        return report
    except Exception as error:
        _record_failure(summary_path, summary, phase, current_image, error)
        raise
    finally:
        if capture is not None:
            capture.close()
        if native_capture is not None:
            native_capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
