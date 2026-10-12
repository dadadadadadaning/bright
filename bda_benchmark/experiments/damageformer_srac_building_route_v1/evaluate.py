"""Full same-pass Val450 audit of E0 and four fixed-Q route controls.

Historical A1/region/J1/J2/J3 keep their original 1e-5 metric gate. The
candidate readouts are additional arms; neither Val labels nor Memory support
are used to construct a route. A prefix is explicitly diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .core import ARMS, FORMAT, head_state, load_head_state, load_route_source, make_heads, source_hash

HISTORICAL = ("a1_pixel", "direct_region", "j1", "j2", "j3")


def same_head_state(current, snapshot):
    """Compare readout state exactly against the saved CPU snapshot."""
    import torch

    if set(current) != set(snapshot):
        return False
    for section, saved_tensors in snapshot.items():
        if set(current[section]) != set(saved_tensors):
            return False
        for name, saved in saved_tensors.items():
            if not torch.equal(current[section][name].detach().cpu(), saved):
                return False
    return True


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, help="Passed formal legacy/native all-path E0")
    p.add_argument("--fit", required=True, help="Completed route Train directory")
    p.add_argument("--output", required=True, help="Fresh Val output directory")
    p.add_argument("--limit", type=int, default=0, help="1..449 is a diagnostic prefix; 0 is formal Val450")
    return p


def _make_predictions(bundle, result, feature, target, image_name):
    import torch
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        decoder_forward, event, gather_support, select_ids, upsample_probability, validate_selected,
    )
    bank = bundle["bank"]
    event_name = event(image_name)
    event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
    query = result["tokens"][0]
    selected = {
        "j2": select_ids(query, bank, event_id, -1, 8),
        "j3": select_ids(query, bank, event_id, -1, 8, projector=bundle["metric"]),
    }
    for ids in selected.values():
        validate_selected(ids, bank, event_id, -1)
    support = {arm: gather_support(ids, bank) for arm, ids in selected.items()}
    probabilities = dict(a1_pixel=result["old_pixel"].float().softmax(1),
                         direct_region=result["prob"])
    for arm in ("j1", "j2", "j3"):
        decoded = decoder_forward(bundle["decoders"][arm], arm, feature, result,
                                  support["j2"], support["j3"])
        probabilities[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
    return probabilities, selected, event_name


def evaluate_bundle(bundle, source_model, heads, output, limit=0, initial=False, q_metadata=None):
    """Evaluate a shared, frozen Q source and independent readouts together."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, new_output, safe_child, verify_unchanged, write_json,
    )
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.native import (
        NativeA1LogitsCapture, native_replay,
    )
    from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.core import checked_journal
    from bda_benchmark.experiments.damageformer_srac_q_factorial_v1.evaluate_q import PartitionBuildingCapture
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, confusion_update, event, full_image_loader,
        metrics, parse_batch, probability_prediction,
    )
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import (
        transition_counts, transition_summary,
    )
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.protocol import reproduction_for_scope
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.train_front import module_fingerprint

    if type(limit) is not int or not 0 <= limit < 450:
        raise ValueError("--limit must be 0 or 1..449")
    if set(heads) != set(ARMS) or bundle["topk"] != 8:
        raise ValueError("All four heads and original E0 Top-8 are required")
    names = bundle["val_names"]
    if len(names) != 450 or list(names) != bundle["protocol"]["selected_images"]:
        raise ValueError("E0 Val450 order changed")
    selected = names[:limit] if limit else names
    smoke = bool(limit)
    out = new_output(output)
    summary = dict(format=FORMAT + "_EVAL", complete=False, full_validation_evidence=False,
                   image_count=0, expected_image_count=450, smoke=smoke, test_accessed=False,
                   training=False, initial=initial, phase="initialize")
    write_json(out / "summary.json", summary)
    source_before = source_hash()
    model_before = module_fingerprint(source_model)
    heads_before = {arm: {section: {name: tensor.detach().cpu().clone()
                                   for name, tensor in values.items()}
                          for section, values in head_state(heads[arm]).items()}
                    for arm in ARMS}
    source_model.eval().requires_grad_(False)
    for head in heads.values():
        head.eval()
    original_capture = source_capture = original_native = source_native = building_capture = guard = None
    current_image = None
    start = time.monotonic()
    try:
        args = bundle["args"]
        data = full_image_loader(args.dataset, selected, False, args.workers, args.seed + 2)
        original_capture = FrozenFrontCapture(bundle["front"])
        source_capture = original_capture if source_model is bundle["front"] else FrozenFrontCapture(source_model)
        original_native = NativeA1LogitsCapture(bundle["front"])
        source_native = (None if source_capture is original_capture
                         else NativeA1LogitsCapture(source_model))
        building_capture = PartitionBuildingCapture(source_model)
        guard = PoolGuard(out / "pool_debug", mode=args.pool_mode,
                          capture="failure", check_reference=False)
        cms = {arm: torch.zeros(4, 4, dtype=torch.long) for arm in (*HISTORICAL, "q_a1_pixel", *ARMS)}
        building_strata = {stratum: {arm: torch.zeros(4, 4, dtype=torch.long)
                                     for arm in ARMS}
                           for stratum in ("predicted_B_below_threshold", "predicted_B_at_least_threshold")}
        hard_soft_transitions = {stratum: torch.zeros(4, 4, 4, dtype=torch.long)
                                 for stratum in building_strata}
        by_event = {}
        native_ok = True
        native_max = 0.0
        initial_route_max_abs = 0.0
        (out / "predictions").mkdir()
        summary["phase"] = "Val_same_pass"
        write_json(out / "summary.json", summary)
        with torch.no_grad(), guard, (out / "images.jsonl").open("x", encoding="utf-8") as journal:
            for index, batch in enumerate(data):
                pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                current_image = str(image_names[0])
                if len(pre) != 1 or current_image != selected[index]:
                    raise ValueError("Val image order or batch changed")
                guard.set_image(current_image)
                original_native.clear()
                original, original_feature = original_capture(pre, post)
                low_logits = original_native.take()
                historical, selected_ids, event_name = _make_predictions(
                    bundle, original, original_feature, target, current_image)
                native = native_replay(low_logits, original["old_pixel"], target.shape[-2:])
                native_ok &= native.get("passed") is True
                native_max = max(native_max, float(native.get("max_abs") or 0.0))
                if not native_ok:
                    raise ValueError("Native A1 replay failed within image: " + current_image)
                if source_capture is original_capture:
                    result, feature = original, original_feature
                    source_replay = native
                else:
                    source_native.clear()
                    result, feature = source_capture(pre, post)
                    source_replay = native_replay(source_native.take(), result["old_pixel"],
                                                  target.shape[-2:])
                    native_ok &= source_replay.get("passed") is True
                    native_max = max(native_max, float(source_replay.get("max_abs") or 0.0))
                    if not native_ok:
                        raise ValueError("Q-source native A1 replay failed within image: " + current_image)
                building = building_capture.take(result["shape"])
                probabilities = dict(historical)
                probabilities["q_a1_pixel"] = result["old_pixel"].float().softmax(1)
                for arm in ARMS:
                    probabilities[arm] = heads[arm].from_frozen(
                        result, feature, building, target.shape[-2:])["prob"]
                if initial:
                    difference = (probabilities["unsplit"].float()
                                  - result["prob"].float()).abs().max().item()
                    initial_route_max_abs = max(initial_route_max_abs, difference)
                    if difference > 2e-5:
                        raise ValueError("Unsplit readout does not replay its initialized source region probability")
                predictions = {arm: probability_prediction(probabilities[arm], target.shape[-2:])
                               for arm in probabilities}
                # GT is used only here, after every old/new prediction exists.
                full_building = F.interpolate(building.float(), target.shape[-2:],
                                              mode="bilinear", align_corners=False)[:, 0]
                low_b = full_building < heads["hard"].threshold
                stratum_masks = {"predicted_B_below_threshold": low_b,
                                 "predicted_B_at_least_threshold": ~low_b}
                for stratum, mask in stratum_masks.items():
                    valid = mask & (target >= 0) & (target < 4)
                    for arm in ARMS:
                        y = target[valid].long()
                        p = predictions[arm][valid].long()
                        building_strata[stratum][arm].add_(
                            torch.bincount(y * 4 + p, minlength=16).reshape(4, 4).cpu())
                    hard_soft_transitions[stratum].add_(transition_counts(
                        target, predictions["hard"], predictions["soft"], mask=mask))
                image_cms = {}
                for arm, prediction in predictions.items():
                    cm = torch.zeros(4, 4, dtype=torch.long)
                    confusion_update(cm, target, prediction)
                    cms[arm].add_(cm)
                    by_event.setdefault(event_name, {}).setdefault(
                        arm, torch.zeros_like(cm)).add_(cm)
                    image_cms[arm] = cm.tolist()
                artifact = out / "predictions" / f"{index:04d}.npz"
                arrays = {"target": target.detach().cpu().numpy().astype(np.int16),
                          "building_grid": building.detach().cpu().numpy().astype(np.float32)}
                arrays.update({"prediction_" + arm: pred.detach().cpu().numpy().astype(np.uint8)
                               for arm, pred in predictions.items()})
                arrays.update({"support_" + arm: ids.detach().cpu().numpy().astype(np.int32)
                               for arm, ids in selected_ids.items()})
                np.savez_compressed(artifact, **arrays)
                journal.write(json.dumps(dict(index=index, image=current_image,
                    event=event_name, confusion=image_cms,
                    native_a1_replay=dict(original=native, q_source=source_replay),
                    prediction_file=artifact.relative_to(out).as_posix(),
                    prediction_sha256=digest(artifact)), ensure_ascii=False,
                    allow_nan=False) + "\n")
                journal.flush()
                summary["image_count"] = index + 1
                if (index + 1) % 25 == 0 or index + 1 == len(selected):
                    print(f"building routes {index + 1}/{len(selected)}", flush=True)
        checked_journal(out, selected, digest, safe_child)
        actual = {arm: metrics(cm) for arm, cm in cms.items()}
        reproduction = reproduction_report(actual, bundle["reference"],
                                           smoke=smoke, image_count=len(selected))
        e0_comparison = reproduction_report(actual, bundle["protocol"]["metrics"],
                                            smoke=smoke, image_count=len(selected))
        q_a1_reproduction = reproduction_for_scope(
            {"a1_pixel": actual["q_a1_pixel"]},
            {"a1_pixel": bundle["reference"]["a1_pixel"]}, "a1",
            smoke=smoke, image_count=len(selected))
        immutable = verify_unchanged(bundle)
        immutable["building_route_code"] = source_before == source_hash()
        immutable["route_q_source"] = model_before == module_fingerprint(source_model)
        immutable["readouts"] = all(same_head_state(head_state(heads[arm]), heads_before[arm])
                                    for arm in ARMS)
        pool_ok = bool(guard.restored and not guard.failures)
        complete = bool(len(selected) == summary["image_count"] and all(immutable.values())
                        and pool_ok and native_ok and
                        (smoke or (reproduction.get("passed") is True
                                   and e0_comparison.get("passed") is True
                                   and q_a1_reproduction.get("passed") is True)))
        formal = bool(not smoke and complete and len(selected) == 450)
        report = dict(format=FORMAT + "_EVAL", complete=complete,
                      full_validation_evidence=formal, image_count=len(selected),
                      expected_image_count=450, smoke=smoke, initial=initial,
                      metrics=actual, per_event={evt: {arm: metrics(cm) for arm, cm in row.items()}
                                                for evt, row in by_event.items()},
                      predicted_building_strata={stratum: dict(
                          rule="B is predicted by frozen A1; bilinear resize to GT grid for retrospective statistics only",
                          threshold=heads["hard"].threshold,
                          class_confusion={arm: cm.tolist() for arm, cm in rows.items()},
                          class_metrics={arm: metrics(cm) for arm, cm in rows.items()},
                          hard_to_soft_gt_old_new=hard_soft_transitions[stratum].tolist(),
                          hard_to_soft_summary=transition_summary(hard_soft_transitions[stratum]))
                          for stratum, rows in building_strata.items()},
                      reproduction=reproduction, e0_metric_comparison=e0_comparison,
                      q_a1_reproduction=q_a1_reproduction,
                      immutable=immutable, native_a1_replay_ok=native_ok,
                      native_a1_replay_max_abs=native_max,
                      initial_unsplit_max_abs=initial_route_max_abs if initial else None,
                      pool_guard=guard.report(), q_source=q_metadata,
                      protocol_sha256=bundle["protocol_sha256"],
                      source_sha256=source_before,
                      test_accessed=False, training=False,
                      gt_rule="Val GT used only after historical supports and all route predictions exist",
                      elapsed_seconds=time.monotonic() - start)
        write_json(out / "result.json", report)
        summary.update(complete=complete, full_validation_evidence=formal,
                       reproduction_passed=reproduction.get("passed"),
                       e0_protocol_comparison_passed=e0_comparison.get("passed"), immutable=immutable,
                       q_a1_reproduction_passed=q_a1_reproduction.get("passed"),
                       native_a1_replay_ok=native_ok, result_file="result.json",
                       result_sha256=digest(out / "result.json"), phase="complete" if complete else "failed_checks")
        write_json(out / "summary.json", summary)
        if not complete or (not smoke and not formal):
            raise RuntimeError("Building-route Val failed E0 reproduction/integrity; 1e-5 gate unchanged")
        return report
    except Exception as exc:
        summary.update(complete=False, full_validation_evidence=False,
                       image=current_image, error=f"{type(exc).__name__}: {exc}")
        write_json(out / "summary.json", summary)
        raise
    finally:
        for value in (building_capture, source_native, original_native, source_capture, original_capture):
            if value is not None and value is not original_capture and hasattr(value, "close"):
                value.close()
        if original_capture is not None:
            original_capture.close()


def evaluate_routes_light(bundle, source_model, heads, limit=0):
    """Periodic all-route Val curve. Diagnostic only: no historical E0 replay."""
    import torch
    from bda_benchmark.experiments.damageformer_srac_q_factorial_v1.evaluate_q import PartitionBuildingCapture
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, confusion_update, full_image_loader, metrics,
        parse_batch, probability_prediction,
    )

    names = bundle["val_names"][:limit] if limit else bundle["val_names"]
    cms = {arm: torch.zeros(4, 4, dtype=torch.long) for arm in ARMS}
    capture = FrozenFrontCapture(source_model)
    building_capture = PartitionBuildingCapture(source_model)
    source_model.eval().requires_grad_(False)
    for head in heads.values():
        head.eval()
    try:
        data = full_image_loader(bundle["args"].dataset, names, False,
                                 bundle["args"].workers, bundle["args"].seed + 2)
        with torch.no_grad():
            for index, batch in enumerate(data):
                pre, post, _loc, target, images = parse_batch(batch, bundle["device"])
                if len(pre) != 1 or str(images[0]) != names[index]:
                    raise ValueError("Periodic Val image order changed")
                result, feature = capture(pre, post)
                building = building_capture.take(result["shape"])
                for arm in ARMS:
                    probability = heads[arm].from_frozen(
                        result, feature, building, target.shape[-2:])["prob"]
                    prediction = probability_prediction(probability, target.shape[-2:])
                    confusion_update(cms[arm], target, prediction)
    finally:
        building_capture.close()
        capture.close()
    return dict(image_count=len(names), smoke=len(names) != 450,
                complete=False, full_validation_evidence=False,
                reason="Periodic route-only validation lacks historical E0 reproduction gate",
                metrics={arm: metrics(cm) for arm, cm in cms.items()})


def run(cli):
    import torch
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.core import load_source

    root = Path(cli.fit)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("format") != FORMAT or summary.get("complete") is not True
            or summary.get("test_accessed") is not False
            or summary.get("smoke") is not False and not cli.limit
            or summary.get("checkpoint") != "heads_last.pth"):
        raise ValueError("Requires completed Train-only final-step route fit")
    bundle = load_source(cli.protocol, cli.output)
    if summary.get("protocol_sha256") != bundle["protocol_sha256"] or summary.get("source_sha256") != source_hash():
        raise ValueError("Route fit belongs to another E0 or code revision")
    checkpoint = root / "heads_last.pth"
    if digest(checkpoint) != summary.get("checkpoint_sha256"):
        raise ValueError("Route checkpoint hash changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (state.get("format") != FORMAT or state.get("complete") is not True
            or state.get("completed_steps") != summary.get("steps")
            or state.get("protocol_sha256") != bundle["protocol_sha256"]
            or state.get("source_sha256") != source_hash()):
        raise ValueError("Route checkpoint is not the verified final step")
    q_checkpoint = summary.get("q_source", {}).get("checkpoint")
    source_model, q_meta = load_route_source(bundle, q_checkpoint, smoke=bool(cli.limit))
    if q_meta != summary.get("q_source"):
        raise ValueError("Q source provenance differs from trained route fit")
    heads = make_heads(source_model, bundle["device"],
                       threshold=summary["threshold"], soft_floor=summary["soft_floor"])
    for arm in ARMS:
        load_head_state(heads[arm], state["heads"][arm])
    bundle["input_paths"]["route_checkpoint"] = checkpoint
    bundle["inputs_before"]["route_checkpoint"] = digest(checkpoint)
    return evaluate_bundle(bundle, source_model, heads, cli.output, limit=cli.limit, q_metadata=q_meta)


if __name__ == "__main__":
    run(parser().parse_args())
