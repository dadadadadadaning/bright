"""Same-pass legacy/native Val450 evaluation of fixed-F/Q classifier adaptation.

The original five paths retain the historical absolute 1e-5 reproduction gate.
Initialized P/R/T probabilities are checked against pixel_grid/region_mean/
direct_region with the established 2e-5 tensor tolerance. These are distinct
checks: the tensor tolerance never replaces the metric reproduction threshold.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import time


FORMAT = "SRAC_REGION_HEAD_ADAPT_EVAL_V1"
BASE_ARMS = ("a1_pixel", "direct_region", "j1", "j2", "j3")
PROBE_ARMS = ("pixel_grid", "feature_mean", "region_mean", "grid_probq")
ADAPT_ARMS = ("p", "r", "t")
INTERVENTIONS = ("uniform_attention", "no_structure", "no_geometry", "no_stats")
PAIRS = (
    "pixel_grid__p", "region_mean__r", "direct_region__t", "p__r",
    "pixel_grid__region_mean", "pixel_grid__grid_probq",
    "pixel_grid__feature_mean", "feature_mean__region_mean",
    "region_mean__direct_region", "a1_pixel__pixel_grid",
    "a1_pixel__grid_probq", "a1_pixel__p", "a1_pixel__r", "a1_pixel__t",
    "grid_probq__r", "j1__j3",
)
INITIAL_PAIRS = {"p": "pixel_grid", "r": "region_mean", "t": "direct_region"}


def formal_gate(image_count, reproduction, immutable, pool_ok,
                native_a1_replay_ok, initial_ok=True, smoke=False):
    return bool(
        not smoke and image_count == 450 and reproduction.get("passed") is True
        and immutable and all(value is True for value in immutable.values())
        and pool_ok is True and native_a1_replay_ok is True and initial_ok is True
    )


def initial_closeness(probabilities):
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import close_report

    checks = {}
    for arm, baseline in INITIAL_PAIRS.items():
        actual, reference = probabilities[arm], probabilities[baseline]
        row = close_report(actual, reference, atol=2e-5, rtol=2e-5)
        row["prediction_differences"] = int((actual.argmax(1) != reference.argmax(1)).sum())
        row["reference"] = baseline
        checks[arm] = row
    return checks


def _checks_passed(checks):
    """Accept only a nonempty map of explicit bools or passed-check records."""
    return bool(checks) and all(
        (value is True if isinstance(value, bool) else
         isinstance(value, dict) and value.get("passed") is True)
        for value in checks.values())


def _merge_checks(total, checks):
    for name, row in checks.items():
        if not isinstance(row, dict) or "passed" not in row:
            continue
        dest = total.setdefault(name, dict(passed=True, checked_images=0,
                                           failed_images=0, max_abs=0.0,
                                           prediction_differences=0))
        dest["checked_images"] += 1
        dest["passed"] &= row.get("passed") is True
        dest["failed_images"] += int(row.get("passed") is not True)
        if row.get("max_abs") is not None:
            dest["max_abs"] = max(dest["max_abs"], float(row["max_abs"]))
        dest["prediction_differences"] += int(row.get("prediction_differences", 0))


def _deltas(metrics, pairs):
    out = {}
    for pair in pairs:
        old, new = pair.split("__")
        a, b = metrics[old]["valid_miou"], metrics[new]["valid_miou"]
        out[pair] = None if a is None or b is None else 100.0 * (b - a)
    return out


def select_validation_images(names, limit=0, image=None):
    if type(limit) is not int or not 0 <= limit < 450:
        raise ValueError("--limit must be 0 (formal) or 1..449 (debug prefix)")
    if image is not None:
        if limit or not isinstance(image, str) or image not in names:
            raise ValueError("Single-image diagnostic requires one E0 Val name and no --limit")
        return [image], True
    return (names[:limit] if limit else names), bool(limit)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, help="Passed legacy/native all-path E0 Val450")
    p.add_argument("--fit", help="Completed equal-budget Train head-fit directory; omit for initialization")
    p.add_argument("--output", required=True, help="Fresh output directory")
    p.add_argument("--limit", type=int, default=0, help="Debug Val prefix 1..449; 0 is formal Val450")
    p.add_argument("--image", help="One exact E0 Val name; diagnostic only, never formal Val450")
    p.add_argument("--interventions", action="store_true",
                   help="Four fixed-weight region-embedding dependency probes, not retrained ablations")
    return p


def evaluate_heads(bundle, heads, output=None, limit=0, interventions=False, initial=False, image=None):
    import torch

    with torch.no_grad():
        return _evaluate_heads(bundle, heads, output, limit, interventions, initial, image)


def _evaluate_heads(bundle, heads, output, limit, interventions, initial, image=None):
    import numpy as np
    import torch
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, new_output, safe_child, verify_unchanged, write_json)
    from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import Accumulator
    from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.core import checked_journal
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import feature_branches
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, decoder_forward, event, full_image_loader, gather_support,
        parse_batch, probability_prediction, select_ids, upsample_probability, validate_selected)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report
    from .core import head_probability, initial_parameter_checks, prepare_inputs, source_hash
    from .native import NativeA1LogitsCapture, native_replay
    from .probability_diagnostics import checked_predictions

    if type(limit) is not int or not 0 <= limit < 450:
        raise ValueError("--limit must be 0 (formal) or 1..449 (debug prefix)")
    names = bundle["val_names"]
    protocol = bundle["protocol"]
    if len(names) != 450 or list(names) != protocol["selected_images"]:
        raise ValueError("Val image list/order differs from passed E0")
    if set(heads.keys()) != set(ADAPT_ARMS):
        raise ValueError("Expected exactly p/r/t adaptation heads")
    if bundle["topk"] != 8:
        raise ValueError("Historical J2/J3 require unchanged Top-8")
    selected, smoke = select_validation_images(names, limit, image)
    args = bundle["args"]
    initialized = initial_parameter_checks(bundle["front"], heads) if initial else {}
    if initial and not _checks_passed(initialized):
        raise ValueError("Initial P/R/T head parameters differ from their frozen sources")
    arms = (*BASE_ARMS, *PROBE_ARMS, *ADAPT_ARMS)
    pairs = PAIRS
    if interventions:
        arms += tuple("region_" + mode for mode in INTERVENTIONS)
        pairs += tuple("direct_region__region_" + mode for mode in INTERVENTIONS)
    out = new_output(output) if output is not None else None
    temp = tempfile.TemporaryDirectory(prefix="srac_head_eval_") if out is None else None
    work = out if out is not None else Path(temp.name)
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   image_count=0, test_accessed=False, training=False, smoke=smoke,
                   initial=bool(initial), diagnostic_image=image,
                   probability_recheck=bundle.get("probability_recheck"), phase="initialize")
    if out:
        write_json(out / "summary.json", summary)
    source_before = source_hash()
    saved_heads = {name: value.detach().clone() for name, value in heads.state_dict().items()}
    training_states = {name: module.training for name, module in heads.named_modules()}
    heads.eval()
    capture = native_capture = guard = None
    started = time.perf_counter()
    current_image, phase = None, "initialize"
    try:
        data = full_image_loader(args.dataset, selected, False, args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        native_capture = NativeA1LogitsCapture(bundle["front"])
        guard = PoolGuard(work / "pool_debug", mode=args.pool_mode,
                          capture="failure", check_reference=False)
        counts = Accumulator(arms=arms, pairs=pairs)
        initial_totals, probe_totals = {}, {}
        native_ok, native_max = True, 0.0
        if out:
            (out / "predictions").mkdir()
        journal_context = ((out / "images.jsonl").open("x", encoding="utf-8")
                           if out else nullcontext(None))
        phase = "Val_same_pass"
        with guard, journal_context as journal:
            for index, batch in enumerate(data):
                pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                current_image = str(image_names[0])
                summary["current_val_index_one_based"] = names.index(current_image) + 1
                if len(pre) != 1 or index >= len(selected) or current_image != selected[index]:
                    raise ValueError("Val image/order/batch differs from E0")
                guard.set_image(current_image)
                native_capture.clear()
                result, feature = capture(pre, post)
                low_logits = native_capture.take()
                bank = bundle["bank"]
                event_name = event(current_image)
                event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
                query = result["tokens"][0]
                # E0 historical order: select both sets immediately after front.
                selected_ids = {
                    "j2": select_ids(query, bank, event_id, -1, 8),
                    "j3": select_ids(query, bank, event_id, -1, 8, projector=bundle["metric"]),
                }
                for ids in selected_ids.values():
                    validate_selected(ids, bank, event_id, -1)
                support = {arm: gather_support(ids, bank) for arm, ids in selected_ids.items()}
                probability = dict(a1_pixel=result["old_pixel"].float().softmax(1),
                                   direct_region=result["prob"])
                for arm in ("j1", "j2", "j3"):
                    decoded = decoder_forward(bundle["decoders"][arm], arm, feature, result,
                                              support["j2"], support["j3"])
                    probability[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
                replay = native_replay(low_logits, result["old_pixel"], target.shape[-2:])
                native_ok &= replay.get("passed") is True
                native_max = max(native_max, float(replay.get("max_abs") or 0.0))
                if not native_ok:
                    summary["native_a1_replay_failure"] = dict(image=current_image, **replay)
                    raise ValueError("Captured native A1 replay failed: " + current_image)
                probes, probe_checks, _ = feature_branches(
                    feature.float(), result["q"], result["ids"], result["count"],
                    bundle["front"].base.clf_2, target.shape[-2:])
                probability.update(probes)
                _merge_checks(probe_totals, probe_checks)
                inputs = prepare_inputs(feature, result)
                for arm in ADAPT_ARMS:
                    probability[arm] = head_probability(arm, heads[arm], inputs, target.shape[-2:])
                checks = initial_closeness(probability) if initial else {}
                _merge_checks(initial_totals, checks)
                if interventions:
                    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_audit import embedding_probability
                    for mode in INTERVENTIONS:
                        probability["region_" + mode] = embedding_probability(
                            bundle["front"], result, feature, target.shape[-2:], mode=mode)
                predictions = checked_predictions(
                    probability, arms, target.shape[-2:], validator=probability_prediction,
                    image=current_image, output=out,
                    context=dict(feature=feature, q=result["q"], ids=result["ids"],
                                 count=result["count"], shape=result["shape"],
                                 tokens=result["tokens"], mean_tokens=inputs["mean_tokens"],
                                 mass=inputs["mass"], native_low_logits=low_logits,
                                 heads=heads.state_dict()))
                # Val GT first enters scoring after all predictions/supports exist.
                cms = counts.add(current_image, event_name, target, predictions)
                if out:
                    arrays = {"target": target.detach().cpu().numpy().astype(np.int16)}
                    arrays.update({"prediction_" + arm: predictions[arm].detach().cpu().numpy().astype(np.uint8)
                                   for arm in arms})
                    arrays.update({"support_" + arm: ids.detach().cpu().numpy().astype(np.int32)
                                   for arm, ids in selected_ids.items()})
                    artifact = out / "predictions" / f"{index:04d}.npz"
                    np.savez_compressed(artifact, **arrays)
                    journal.write(json.dumps(dict(
                        index=index, image=current_image, event=event_name, confusion=cms,
                        native_a1_replay=replay, initial_head_checks=checks,
                        prediction_file=artifact.relative_to(out).as_posix(),
                        prediction_sha256=digest(artifact)), ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                summary["image_count"] = counts.image_count
                if counts.image_count % 10 == 0 or counts.image_count == len(selected):
                    print(f"Region head Val {counts.image_count}/{len(selected)}", flush=True)
        phase = "reproduction_and_integrity"
        if out:
            checked_journal(out, selected, digest, safe_child)
        result_report = counts.finish()
        reproduction = reproduction_report(result_report["metrics"], bundle["reference"],
                                           smoke=smoke, image_count=counts.image_count)
        e0_comparison = reproduction_report(result_report["metrics"], protocol["metrics"],
                                            smoke=smoke, image_count=counts.image_count)
        immutable = verify_unchanged(bundle)
        immutable["head_adapt_code"] = source_before == source_hash()
        immutable["evaluated_heads"] = all(torch.equal(value, heads.state_dict()[name])
                                           for name, value in saved_heads.items())
        pool_ok = bool(guard.restored and not guard.failures)
        init_ok = (not initial or (_checks_passed(initialized) and _checks_passed(initial_totals)))
        complete = bool(counts.image_count == len(selected) and all(immutable.values())
                        and pool_ok and native_ok and init_ok
                        and (smoke or reproduction.get("passed") is True))
        formal = formal_gate(counts.image_count, reproduction, immutable, pool_ok,
                             native_ok, init_ok, smoke=smoke)
        report = dict(
            format=FORMAT, complete=complete, full_validation_evidence=formal,
            **result_report, expected_image_count=450, smoke=smoke, diagnostic_image=image,
            initial=bool(initial), initial_parameters=initialized,
            initial_probability_checks=initial_totals, initial_checks_passed=init_ok,
            diagnostic_probe_checks=probe_totals,
            reproduction=reproduction, e0_metric_comparison=e0_comparison,
            immutable=immutable, native_a1_replay_ok=native_ok,
            native_a1_replay_max_abs=native_max, pool_guard=guard.report(),
            comparison_pairs=list(pairs), arms=list(arms),
            miou_delta_percentage_points=_deltas(result_report["metrics"], pairs),
            per_event_miou_delta_percentage_points={event_key: _deltas(value, pairs)
                for event_key, value in result_report["per_event"].items()},
            protocol_path=str(bundle["protocol_path"]), protocol_sha256=bundle["protocol_sha256"],
            source_sha256=source_before, profile=protocol["profile"], pool_mode=protocol["pool_mode"],
            head_fit=bundle.get("head_fit_metadata"),
            probability_recheck=bundle.get("probability_recheck"),
            interventions=bool(interventions),
            intervention_rule="Fixed-weight dependency probes, not retrained ablations",
            grid_probq_rule="Original frozen feature_branches probability pooling; extra A1 classifier call",
            native_rule="Identity from captured original classifier logits; extra classifier replay is not a gate",
            support_id_rule="Fresh same-pass Top-8; historical ID equality not required",
            gt_rule="Val labels used only after all predictions and support IDs exist",
            test_accessed=False, training=False, elapsed_seconds=time.perf_counter() - started)
        if out:
            write_json(out / "result.json", report)
            summary.update(complete=complete, full_validation_evidence=formal,
                           reproduction_passed=reproduction.get("passed"), immutable=immutable,
                           native_a1_replay_ok=native_ok, initial_checks_passed=init_ok,
                           result_file="result.json", result_sha256=digest(out / "result.json"),
                           phase="complete" if complete else "failed_checks")
            write_json(out / "summary.json", summary)
        if not complete or (not smoke and not formal):
            raise RuntimeError("Head evaluation failed reproduction/integrity/initialization; 1e-5 unchanged")
        return report
    except Exception as error:
        if hasattr(error, "detail"):
            summary["probability_failure"] = error.detail
        summary.update(complete=False, full_validation_evidence=False,
                       phase=phase, image=current_image, error=f"{type(error).__name__}: {error}")
        if out:
            write_json(out / "summary.json", summary)
        raise
    finally:
        if native_capture is not None:
            native_capture.close()
        if capture is not None:
            capture.close()
        for name, module in heads.named_modules():
            module.training = training_states[name]
        if temp is not None:
            temp.cleanup()


def run(cli):
    from .core import load_source, load_fit, make_heads
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest

    bundle = load_source(cli.protocol, cli.output)
    if cli.fit:
        heads, metadata = load_fit(cli.fit, bundle, allow_smoke=bool(cli.limit or cli.image))
        bundle["head_fit_metadata"] = metadata
        for name in ("summary.json", "heads_last.pth"):
            path = Path(cli.fit) / name
            key = "head_fit_" + name
            bundle["input_paths"][key] = path
            bundle["inputs_before"][key] = digest(path)
    else:
        heads = make_heads(bundle["front"])
    return evaluate_heads(bundle, heads, output=cli.output, limit=cli.limit,
                          interventions=cli.interventions, initial=not bool(cli.fit), image=cli.image)


if __name__ == "__main__":
    run(parser().parse_args())
