"""Same-pass Val450 comparison of frozen A1/Q-region paths and J1/J2/J3.

Only the E0 Val list is loaded. The original five-arm 1e-5 reproduction gate,
source checks, and native-pool guard are retained. No model is trained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .core import (ARMS, FORMAT, PAIRS, REGION_ARMS, checked_journal,
                   checked_protocol, formal_gate, metric_deltas,
                   reproduction_metrics, source_hash)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True,
                   help="Passed legacy/native all-path E0 Val450 directory or protocol.json")
    p.add_argument("--output", required=True, help="New, empty output directory")
    p.add_argument("--limit", type=int, default=0,
                   help="First N Val images for a nonformal debug prefix; 0 = Val450")
    return p


def run(cli):
    import torch
    # E0 protocol.run uses no_grad around its entire pass.
    with torch.no_grad():
        return _run(cli)


def _run(cli):
    import numpy as np
    import torch
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, load_verified_bundle, new_output, safe_child, verify_unchanged, write_json,
    )
    from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import (
        Accumulator, evaluate_image,
    )
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, decoder_forward, event, full_image_loader,
        gather_support, parse_batch, probability_prediction, select_ids,
        upsample_probability, validate_selected,
    )
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report

    if cli.limit < 0 or cli.limit > 450:
        raise ValueError("--limit must be 0..450")
    protocol_path, protocol = checked_protocol(cli.protocol)
    output = new_output(cli.output)
    summary_path = output / "summary.json"
    source_before = source_hash()
    started = time.perf_counter()
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   image_count=0, test_accessed=False, training=False,
                   phase="initialize", source_sha256=source_before)
    write_json(summary_path, summary)
    capture, guard = None, None
    phase, image_name = "load_verified_bundle", None
    try:
        # The E0 arguments bind dataset, Val list, checkpoints, numeric profile,
        # device and native pooling; no Test argument exists in this CLI.
        args = argparse.Namespace(**protocol["arguments"])
        args.protocol, args.output = str(protocol_path), str(output)
        bundle = load_verified_bundle(args)
        names = bundle["val_names"]
        if names != protocol["selected_images"] or len(names) != 450:
            raise ValueError("Current Val450 list/order differs from E0")
        selected = names[:cli.limit] if cli.limit else names
        if not selected:
            raise ValueError("Empty Val selection")
        data = full_image_loader(args.dataset, selected, False, args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        guard = PoolGuard(output / "pool_debug", mode=args.pool_mode,
                          capture="failure", check_reference=False)
        counts = Accumulator(arms=ARMS, pairs=PAIRS)
        failed_checks = {}
        native_a1_replay_ok = True
        prediction_dir = output / "predictions"
        prediction_dir.mkdir()
        phase = "Val_same_pass_inference"
        with guard:
            with (output / "images.jsonl").open("x", encoding="utf-8") as journal:
                for index, batch in enumerate(data):
                    pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                    image_name = str(image_names[0])
                    if len(pre) != 1 or index >= len(selected) or image_name != selected[index]:
                        raise ValueError("Unexpected Val image/order/batch")
                    guard.set_image(image_name)
                    result, feature = capture(pre, post)

                    # Select Top-8 immediately after the front, exactly where
                    # E0 selected it. The four new Q probes run after E0 arms.
                    bank = bundle["bank"]
                    event_name = event(image_name)
                    event_id = (bank["event_names"].index(event_name)
                                if event_name in bank["event_names"] else -1)
                    query = result["tokens"][0]
                    selected_ids = {
                        "j2": select_ids(query, bank, event_id, -1, bundle["topk"]),
                        "j3": select_ids(query, bank, event_id, -1, bundle["topk"],
                                         projector=bundle["metric"]),
                    }
                    for ids in selected_ids.values():
                        validate_selected(ids, bank, event_id, -1)
                    support = {arm: gather_support(ids, bank)
                               for arm, ids in selected_ids.items()}
                    existing_probability = {
                        "a1_pixel": result["old_pixel"].float().softmax(1),
                        "direct_region": result["prob"],
                    }
                    for arm in ("j1", "j2", "j3"):
                        decoded = decoder_forward(bundle["decoders"][arm], arm,
                                                  feature, result,
                                                  support["j2"], support["j3"])
                        existing_probability[arm] = upsample_probability(
                            decoded["logits"], target.shape[-2:])

                    region_probability, checks = evaluate_image(
                        bundle["front"], result, feature, target.shape[-2:])
                    probability = {**region_probability,
                                   **{arm: existing_probability[arm]
                                      for arm in ("j1", "j2", "j3")}}
                    if set(probability) != set(ARMS):
                        raise RuntimeError("Unexpected Val comparison arms")
                    # Retain E0 A1/region probability objects rather than
                    # recomputing them inside the probe for reproduction.
                    probability["a1_pixel"] = existing_probability["a1_pixel"]
                    probability["region"] = existing_probability["direct_region"]
                    native_a1_replay_ok &= checks["a1_native_replay"].get("passed") is True
                    for key, row in checks.items():
                        if isinstance(row, dict) and row.get("passed") is False:
                            item = failed_checks.setdefault(key, {"count": 0, "max_abs": 0.0})
                            item["count"] += 1
                            if row.get("max_abs") is not None:
                                item["max_abs"] = max(item["max_abs"], float(row["max_abs"]))

                    # GT first enters the calculation below, after every
                    # prediction and Top-8 set has been fixed.
                    predictions = {arm: probability_prediction(probability[arm],
                                                               target.shape[-2:])
                                   for arm in ARMS}
                    image_cms = counts.add(image_name, event_name, target, predictions)
                    arrays = {"target": target.detach().cpu().numpy().astype(np.int16)}
                    for arm in ARMS:
                        arrays["prediction_" + arm] = (
                            predictions[arm].detach().cpu().numpy().astype(np.uint8))
                    for arm, ids in selected_ids.items():
                        arrays["support_" + arm] = ids.detach().cpu().numpy()
                    prediction_file = prediction_dir / f"{index:04d}.npz"
                    np.savez_compressed(prediction_file, **arrays)
                    journal.write(json.dumps(dict(
                        index=index, image=image_name, event=event_name,
                        confusion=image_cms,
                        prediction_file=prediction_file.relative_to(output).as_posix(),
                        prediction_sha256=digest(prediction_file),
                    ), ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    summary["image_count"] = counts.image_count
                    if counts.image_count % 10 == 0 or counts.image_count == len(selected):
                        print(f"Val region + Memory {counts.image_count}/{len(selected)}", flush=True)

        phase = "reproduction_and_integrity"
        checked_journal(output, selected, digest, safe_child)
        result = counts.finish()
        if result["image_count"] != len(selected):
            raise ValueError("Val image count disagrees with selected names")
        reproducing = reproduction_metrics(result["metrics"])
        smoke = len(selected) != 450
        reproduction = reproduction_report(reproducing, bundle["reference"],
                                           smoke=smoke, image_count=len(selected))
        e0_comparison = reproduction_report(reproducing, protocol["metrics"],
                                            smoke=smoke, image_count=len(selected))
        immutable = verify_unchanged(bundle)
        immutable["probe_code"] = source_before == source_hash()
        pool_ok = bool(guard.restored and not guard.failures)
        formal = formal_gate(len(selected), len(names), reproduction,
                             immutable, pool_ok, native_a1_replay_ok)
        complete = bool(
            len(selected) == summary["image_count"]
            and all(immutable.values()) and pool_ok and native_a1_replay_ok
            and (smoke or reproduction.get("passed") is True)
        )
        report = dict(
            format=FORMAT, complete=complete, full_validation_evidence=formal,
            expected_image_count=len(names),
            region_probe_arms=list(REGION_ARMS), joint_arms=["j1", "j2", "j3"],
            comparison_pairs=list(PAIRS), **result,
            miou_delta_percentage_points=metric_deltas(result["metrics"]),
            per_event_miou_delta_percentage_points={
                event_name: metric_deltas(event_metrics)
                for event_name, event_metrics in result["per_event"].items()},
            reproduction=reproduction, e0_metric_comparison=e0_comparison,
            immutable=immutable, native_a1_replay_ok=native_a1_replay_ok,
            diagnostic_numerical_failures=failed_checks,
            pool_guard=guard.report(),
            protocol_path=str(protocol_path.resolve()),
            protocol_sha256=digest(protocol_path),
            source_sha256=source_before,
            profile=protocol["profile"], pool_mode=protocol["pool_mode"],
            support_id_rule="Fresh same-pass Top-8; E0 ID equality is not required",
            gt_rule="Val GT enters only after all prediction paths and support IDs are fixed",
            interpretation="Frozen-path paired comparisons; cross-path changes are not causal Memory gain",
            test_accessed=False, training=False,
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(output / "result.json", report)
        summary.update(complete=complete, full_validation_evidence=formal,
                       reproduction_passed=reproduction.get("passed"),
                       immutable=immutable, native_a1_replay_ok=native_a1_replay_ok,
                       result_file="result.json",
                       phase="complete" if complete else "failed_reproduction_or_integrity")
        write_json(summary_path, summary)
        if not complete or (not smoke and not formal):
            raise RuntimeError("Val reproduction/immutability failed; fixed 1e-5 gate unchanged")
        return report
    except Exception as exc:
        summary.update(complete=False, full_validation_evidence=False,
                       phase=phase, image=image_name,
                       error=f"{type(exc).__name__}: {exc}")
        write_json(summary_path, summary)
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
