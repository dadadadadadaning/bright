"""Same-pass, immutable Val evaluation of bounded pixel/region corrections.

The original five paths must independently reproduce both the historical joint
reference and the passed E0 metrics at absolute 1e-5. A two-image smoke never
counts as a formal result. No labels enter predictions or support selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

FORMAT = "SRAC_REGION_RESIDUAL_EVAL_V1"
BASE_ARMS = ("a1_pixel", "direct_region", "j1", "j2", "j3")
PROBE_ARMS = ("pixel_grid", "region_mean", "grid_probq")
ADAPT_ARMS = ("local_delta", "region_delta")
ARMS = (*BASE_ARMS, *PROBE_ARMS, *ADAPT_ARMS)
PAIRS = (
    "grid_probq__local_delta", "grid_probq__region_delta",
    "local_delta__region_delta", "region_mean__direct_region",
    "pixel_grid__grid_probq", "a1_pixel__grid_probq",
    "a1_pixel__local_delta", "a1_pixel__region_delta", "j1__j3",
)


def source_hash():
    """Hash every external executable and its unchanged repository dependencies."""
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.core import source_hash as inherited_hash

    checksum = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ("residual_core.py", "residual_eval.py", "region_residual.py"):
        data = (root / name).read_bytes()  # Missing executable is a hard failure.
        checksum.update(name.encode("utf-8") + b"\0")
        checksum.update(str(len(data)).encode("ascii") + b"\0" + data)
    checksum.update(b"inherited_region_head_adapt\0" + inherited_hash().encode("ascii"))
    return checksum.hexdigest()


def checks_passed(checks):
    return bool(checks) and all(
        value is True if isinstance(value, bool) else
        isinstance(value, dict) and value.get("passed") is True
        for value in checks.values())


def formal_gate(image_count, reproduction, e0_comparison, immutable, pool_ok,
                native_ok, initial_ok=True, q_ok=True, smoke=False):
    return bool(
        not smoke and image_count == 450
        and reproduction.get("passed") is True
        and e0_comparison.get("passed") is True
        and checks_passed(immutable) and pool_ok is True and native_ok is True
        and initial_ok is True and q_ok is True)


def initial_closeness(probabilities):
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import close_report

    checks = {}
    for arm in ADAPT_ARMS:
        actual, reference = probabilities[arm], probabilities["grid_probq"]
        row = close_report(actual, reference, atol=2e-5, rtol=2e-5)
        row["prediction_differences"] = int((actual.argmax(1) != reference.argmax(1)).sum())
        row["reference"] = "grid_probq"
        checks[arm] = row
    return checks


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


def _deltas(metrics):
    result = {}
    for pair in PAIRS:
        old, new = pair.split("__")
        a, b = metrics[old]["valid_miou"], metrics[new]["valid_miou"]
        result[pair] = None if a is None or b is None else 100.0 * (b - a)
    return result


def evaluate(bundle, heads, output, *, initial=False, smoke=False, source_sha256=None):
    import torch

    with torch.no_grad():
        return _evaluate(bundle, heads, output, initial=initial, smoke=smoke,
                         source_sha256=source_sha256)


def _evaluate(bundle, heads, output, *, initial, smoke, source_sha256):
    import numpy as np
    import torch
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, new_output, safe_child, verify_unchanged, write_json)
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.native import (
        NativeA1LogitsCapture, native_replay)
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.probability_diagnostics import checked_predictions
    from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import Accumulator
    from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.core import checked_journal
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import feature_branches
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, decoder_forward, event, full_image_loader, gather_support,
        parse_batch, probability_prediction, select_ids, upsample_probability, validate_selected)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report
    from residual_core import prepare, probability, parameter_checks

    if type(smoke) is not bool or type(initial) is not bool:
        raise ValueError("Smoke and initial must be explicit booleans")
    names, protocol = bundle["val_names"], bundle["protocol"]
    if len(names) != 450 or len(set(names)) != 450 or list(names) != protocol["selected_images"]:
        raise ValueError("Requires original E0 Val450 image order, without duplicate names")
    if set(heads.keys()) != set(ADAPT_ARMS):
        raise ValueError("Expected exactly local_delta and region_delta")
    if bundle["topk"] != 8:
        raise ValueError("Historical J2/J3 require unchanged Top-8")
    current_source = source_hash()
    if source_sha256 is not None and source_sha256 != current_source:
        raise ValueError("Requested source hash differs from executable source")
    initialized = parameter_checks(bundle["front"], heads) if initial else {}
    if initial and initialized.get("passed") is not True:
        raise ValueError("Zero-initialized correction parameters failed their initial contract")
    selected = names[:2] if smoke else names
    args = bundle["args"]
    out = new_output(output)
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   gates_passed=False, image_count=0, test_accessed=False,
                   training=False, smoke=smoke, initial=initial, phase="initialize",
                   source_sha256=current_source, protocol_sha256=bundle["protocol_sha256"])
    write_json(out / "summary.json", summary)
    saved_heads = {name: value.detach().clone() for name, value in heads.state_dict().items()}
    training_states = {name: module.training for name, module in heads.named_modules()}
    heads.eval()
    capture = native_capture = guard = None
    started = time.perf_counter()
    current_image, phase = None, "initialize"
    try:
        # Check original evidence before running the first new forward as well.
        verify_unchanged(bundle)
        data = full_image_loader(args.dataset, selected, False, args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        native_capture = NativeA1LogitsCapture(bundle["front"])
        guard = PoolGuard(out / "pool_debug", mode=args.pool_mode,
                          capture="failure", check_reference=False)
        counts = Accumulator(arms=ARMS, pairs=PAIRS)
        initial_totals, probe_totals = {}, {}
        native_ok, native_max = True, 0.0
        (out / "predictions").mkdir()
        phase = "Val_same_pass"
        with guard, (out / "images.jsonl").open("x", encoding="utf-8") as journal:
            for index, batch in enumerate(data):
                pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                current_image = str(image_names[0])
                if len(pre) != 1 or index >= len(selected) or current_image != selected[index]:
                    raise ValueError("Val image/order/batch differs from E0")
                summary["current_val_index_one_based"] = index + 1
                guard.set_image(current_image)
                native_capture.clear()
                result, feature = capture(pre, post)
                low_logits = native_capture.take()
                bank = bundle["bank"]
                event_name = event(current_image)
                event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
                query = result["tokens"][0]
                # Preserve E0 retrieval order before executing added classifiers.
                selected_ids = {
                    "j2": select_ids(query, bank, event_id, -1, 8),
                    "j3": select_ids(query, bank, event_id, -1, 8, projector=bundle["metric"]),
                }
                for ids in selected_ids.values():
                    validate_selected(ids, bank, event_id, -1)
                support = {arm: gather_support(ids, bank) for arm, ids in selected_ids.items()}
                probabilities = dict(a1_pixel=result["old_pixel"].float().softmax(1),
                                     direct_region=result["prob"])
                for arm in ("j1", "j2", "j3"):
                    decoded = decoder_forward(bundle["decoders"][arm], arm, feature, result,
                                              support["j2"], support["j3"])
                    probabilities[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
                replay = native_replay(low_logits, result["old_pixel"], target.shape[-2:])
                native_ok &= replay.get("passed") is True
                native_max = max(native_max, float(replay.get("max_abs") or 0.0))
                if not native_ok:
                    summary["native_a1_replay_failure"] = dict(image=current_image, **replay)
                    raise ValueError("Captured native A1 replay failed: " + current_image)
                probes, probe_checks, _ = feature_branches(
                    feature.float(), result["q"], result["ids"], result["count"],
                    bundle["front"].base.clf_2, target.shape[-2:])
                probabilities.update({arm: probes[arm] for arm in PROBE_ARMS})
                _merge_checks(probe_totals, probe_checks)
                inputs = prepare(feature, result, bundle["front"].base.clf_2)
                for arm in ADAPT_ARMS:
                    probabilities[arm] = probability(arm, heads[arm], inputs, target.shape[-2:])
                checks = initial_closeness(probabilities) if initial else {}
                _merge_checks(initial_totals, checks)
                predictions = checked_predictions(
                    probabilities, ARMS, target.shape[-2:], validator=probability_prediction,
                    image=current_image, output=out,
                    context=dict(feature=feature, q=result["q"], ids=result["ids"],
                                 count=result["count"], shape=result["shape"],
                                 tokens=result["tokens"], mean_tokens=inputs["mean_tokens"],
                                 mass=inputs["mass"], native_low_logits=low_logits,
                                 heads=heads.state_dict()))
                # GT is used for counts only after predictions and support IDs exist.
                cms = counts.add(current_image, event_name, target, predictions)
                arrays = {"target": target.detach().cpu().numpy().astype(np.int16)}
                arrays.update({"prediction_" + arm: predictions[arm].detach().cpu().numpy().astype(np.uint8)
                               for arm in ARMS})
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
                    print(f"Region residual Val {counts.image_count}/{len(selected)}", flush=True)
        phase = "reproduction_and_integrity"
        checked_journal(out, selected, digest, safe_child)
        result_report = counts.finish()
        reproduction = reproduction_report(result_report["metrics"], bundle["reference"],
                                           smoke=smoke, image_count=counts.image_count)
        e0_comparison = reproduction_report(result_report["metrics"], protocol["metrics"],
                                            smoke=smoke, image_count=counts.image_count)
        immutable = verify_unchanged(bundle)
        immutable["residual_code"] = current_source == source_hash()
        current_heads = heads.state_dict()
        immutable["evaluated_heads"] = set(current_heads) == set(saved_heads) and all(
            torch.equal(value, current_heads[name]) for name, value in saved_heads.items())
        pool_ok = bool(guard.restored and not guard.failures)
        init_ok = not initial or (initialized.get("passed") is True and checks_passed(initial_totals))
        q_ok = probe_totals.get("q_normalization", {}).get("passed") is True
        complete = bool(counts.image_count == len(selected) and checks_passed(immutable)
                        and pool_ok and native_ok and init_ok and q_ok
                        and (smoke or (reproduction.get("passed") is True
                                       and e0_comparison.get("passed") is True)))
        formal = formal_gate(counts.image_count, reproduction, e0_comparison, immutable,
                             pool_ok, native_ok, init_ok, q_ok, smoke)
        report = dict(
            format=FORMAT, complete=complete, gates_passed=complete,
            full_validation_evidence=formal, **result_report,
            expected_image_count=450, smoke=smoke, initial=initial,
            initial_parameters=initialized, initial_probability_checks=initial_totals,
            initial_checks_passed=init_ok, diagnostic_probe_checks=probe_totals,
            q_checks_passed=q_ok, reproduction=reproduction, e0_metric_comparison=e0_comparison,
            immutable=immutable, native_a1_replay_ok=native_ok,
            native_a1_replay_max_abs=native_max, pool_guard=guard.report(),
            comparison_pairs=list(PAIRS), arms=list(ARMS),
            miou_delta_percentage_points=_deltas(result_report["metrics"]),
            per_event_miou_delta_percentage_points={key: _deltas(value)
                for key, value in result_report["per_event"].items()},
            protocol_path=str(bundle["protocol_path"]), protocol_sha256=bundle["protocol_sha256"],
            source_sha256=current_source, profile=protocol["profile"], pool_mode=protocol["pool_mode"],
            head_fit=bundle.get("head_fit_metadata"),
            grid_probq_rule="Original frozen probability pool/unpool; initial delta is zero",
            comparison_rule="Matched local versus region correction; no historical Memory modification",
            native_rule="Captured original A1 logits replay; no extra classifier call defines this gate",
            support_id_rule="Fresh same-pass Top-8; historical ID equality not required",
            gt_rule="Val labels used only after predictions and support selection",
            metric_rule="Fixed mean of foreground IoUs 1/2/3; background excluded from mIoU",
            test_accessed=False, training=False, elapsed_seconds=time.perf_counter() - started)
        write_json(out / "result.json", report)
        summary.update(complete=complete, full_validation_evidence=formal, gates_passed=complete,
                       reproduction_passed=reproduction.get("passed"),
                       e0_comparison_passed=e0_comparison.get("passed"), immutable=immutable,
                       native_a1_replay_ok=native_ok, initial_checks_passed=init_ok,
                       q_checks_passed=q_ok, result_file="result.json",
                       result_sha256=digest(out / "result.json"),
                       phase="complete" if complete else "failed_checks")
        write_json(out / "summary.json", summary)
        if not complete or (not smoke and not formal):
            raise RuntimeError("Residual evaluation failed reproduction/integrity/initialization; 1e-5 unchanged")
        return report
    except Exception as error:
        if hasattr(error, "detail"):
            summary["probability_failure"] = error.detail
        summary.update(complete=False, full_validation_evidence=False, gates_passed=False,
                       phase=phase, image=current_image, error=f"{type(error).__name__}: {error}")
        write_json(out / "summary.json", summary)
        raise
    finally:
        if native_capture is not None:
            native_capture.close()
        if capture is not None:
            capture.close()
        for name, module in heads.named_modules():
            module.training = training_states[name]
