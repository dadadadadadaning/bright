"""Frozen same-member T128/F256 retrieval and label-evidence diagnostics.

The four label-vote paths are untrained probes, not Memory Reader improvements.
Train labels supply values; Val truth is used only after retrieval and prediction.
The native A1 classifier is called exactly once by the original front forward.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path


FORMAT = "SRAC_MEMORY_KEY_DIAGNOSE_V2"
BASE_ARMS = ("a1_pixel", "direct_region", "j1", "j2", "j3")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True,
                   help="Passed full Val450 legacy/native E0 directory or protocol.json")
    p.add_argument("--keys", required=True, help="Completed build_keys output directory")
    p.add_argument("--output", required=True, help="Fresh output directory")
    p.add_argument("--limit", type=int, default=0,
                   help="First N images for smoke only; 0 is full Val450")
    p.add_argument("--temperature", type=float, default=.1)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--policy", choices=("both", "global", "per_class"), default="both")
    return p


def formal_gate(image_count, expected_count, reproduction, e0_reproduction,
                immutable, pool_ok, native_ok):
    """A smoke prefix or failed evidence gate cannot become formal Val450."""
    return bool(image_count == expected_count == 450
                and reproduction.get("passed") is True
                and e0_reproduction.get("passed") is True
                and bool(immutable) and all(value is True for value in immutable.values())
                and pool_ok is True and native_ok is True)


def _weighted_evidence_diagnostic(evidence, distribution, mass):
    """Post-hoc Q-region statistics; these are not building accuracy metrics."""
    import torch

    posterior = evidence["posterior"].detach().float()
    supported = evidence["supported"].detach().bool()
    query, weight = distribution.detach().float(), mass.detach().float()
    if posterior.shape != query.shape or weight.shape != (len(query),):
        raise ValueError("Post-hoc evidence/query shapes disagree")
    fg = query[:, 1:].sum(-1)
    fg_purity = query[:, 1:].max(-1).values / fg.clamp_min(1e-12)
    partitions = {
        "all": weight > 0,
        "foreground_pure": (weight > 0) & (fg >= .8) & (fg_purity >= .8),
        "foreground_mixed": (weight > 0) & (fg >= .8) & (fg_purity < .8),
        "background_diluted": (weight > 0) & (fg >= .2) & (fg < .8),
        "mostly_background": (weight > 0) & (fg < .2),
    }
    pred, truth = posterior.argmax(-1), query.argmax(-1)
    dot = (posterior * query).sum(-1)
    l1 = (posterior - query).abs().sum(-1)
    result = {}
    for name, mask in partitions.items():
        usable = mask & supported
        intact, damaged = usable & (truth == 1), usable & (truth == 2)
        result[name] = dict(
            queries=int(mask.sum()), supported_queries=int(usable.sum()),
            soft_compatibility_sum=float(dot[usable].double().sum()),
            l1_sum=float(l1[usable].double().sum()),
            dominant_agree=int((usable & (pred == truth)).sum()),
            valid_mass=float(weight[mask].double().sum()),
            supported_valid_mass=float(weight[usable].double().sum()),
            mass_weighted_compatibility_sum=float((dot[usable] * weight[usable]).double().sum()),
            intact_queries=int(intact.sum()),
            intact_vote_damaged=int((intact & (pred == 2)).sum()),
            intact_vote_damaged_mass_sum=float(posterior[intact, 2].double().sum()),
            damaged_queries=int(damaged.sum()),
            damaged_vote_intact=int((damaged & (pred == 1)).sum()),
            damaged_vote_intact_mass_sum=float(posterior[damaged, 1].double().sum()),
        )
    return result


def _vote_probability(evidence, result, output_size, baseline):
    """Standalone conditional label vote; no-support pixels use grid_probq.

    Partial support is normalized by supported Q mass. That fraction is saved
    separately, so a weakly covered pixel is not presented as full coverage.
    This diagnostic replacement is deliberately not a trained correction head.
    """
    import torch
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import (
        project_label_evidence)

    conditional, supported_mass, _ = project_label_evidence(
        evidence["posterior"], evidence["supported"], result["q"], result["ids"],
        result["shape"], output_size)
    probability = torch.where((supported_mass > 1e-8)[:, None], conditional, baseline)
    return probability, supported_mass


def _record_failure(output, summary, phase, image, error):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import write_json

    trace = traceback.format_exc()
    (output / "traceback.txt").write_text(trace, encoding="utf-8")
    summary.update(complete=False, full_validation_evidence=False,
                   execution_complete=False, phase=phase, image=image,
                   error=f"{type(error).__name__}: {error}", traceback_file="traceback.txt")
    write_json(output / "summary.json", summary)


def _verify_saved_artifacts(output, names, arms, retrieval_arms, bank_count):
    """Re-open hashed maps and supports; reject missing or reordered evidence."""
    import numpy as np
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, safe_child)

    rows = [json.loads(line) for line in (output / "images.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if ([row.get("image") for row in rows] != list(names)
            or [row.get("index") for row in rows] != list(range(len(names)))):
        raise ValueError("Per-image Val journal is missing or out of order")
    for row in rows:
        artifact = safe_child(output, row["prediction_file"])
        if digest(artifact) != row["prediction_sha256"]:
            raise ValueError("Saved prediction/support SHA mismatch")
        with np.load(artifact, allow_pickle=False) as saved:
            if ("target" not in saved or saved["target"].ndim != 3
                    or saved["target"].shape[0] != 1):
                raise ValueError("Saved artifact lacks [1,H,W] target")
            for arm in arms:
                name = "prediction_" + arm
                if (name not in saved or saved[name].shape != saved["target"].shape
                        or not np.issubdtype(saved[name].dtype, np.integer)
                        or ((saved[name] < 0) | (saved[name] >= 4)).any()):
                    raise ValueError("Saved prediction shape/class mismatch: " + arm)
            for arm in retrieval_arms:
                for suffix in ("", "_64"):
                    ids = saved["support_" + arm + suffix]
                    scores = saved["scores_" + arm + suffix]
                    if (ids.ndim != 2 or ids.shape != scores.shape
                            or not np.issubdtype(ids.dtype, np.integer)
                            or ((ids < -1) | (ids >= bank_count)).any()
                            or not np.isfinite(scores[ids >= 0]).all()
                            or not np.isneginf(scores[ids < 0]).all()):
                        raise ValueError("Saved support IDs/scores invalid: " + arm)
                labels = saved["softlabels_" + arm]
                ids = saved["support_" + arm]
                if (labels.shape != (*ids.shape, 4) or not np.isfinite(labels).all()
                        or (labels < 0).any()
                        or not np.allclose(labels[ids >= 0].sum(-1), 1, atol=1e-5, rtol=0)
                        or (labels[ids < 0] != 0).any()):
                    raise ValueError("Saved support soft labels invalid: " + arm)
                evidence, quality, mass = (saved["evidence_" + arm], saved["quality_" + arm],
                                           saved["supported_mass_" + arm])
                if (evidence.shape != (len(ids), 4) or quality.shape != (len(ids), 6)
                        or mass.shape != saved["target"].shape):
                    raise ValueError("Saved label evidence shapes differ: " + arm)
                supported = (ids >= 0).any(-1)
                if (not np.isfinite(evidence).all() or not np.isfinite(quality).all()
                        or not np.isfinite(mass).all() or (evidence < 0).any()
                        or (mass < 0).any() or (mass > 1).any()
                        or not np.allclose(evidence[supported].sum(-1), 1, atol=1e-5, rtol=0)
                        or (evidence[~supported] != 0).any()):
                    raise ValueError("Saved label evidence values invalid: " + arm)
    return rows


def run(cli):
    import torch

    with torch.no_grad():
        return _run(cli)


def _run(cli):
    import numpy as np
    import torch

    from .core import select_neighbors, label_evidence, paired_retrieval_diagnostics
    from .shared import load_bundle, load_paired_bank, source_hash, verify_all
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, new_output, write_json)
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import (
        NativeA1LogitsCapture, native_grid_probq, sum_nested_counts)
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.retrieve import (
        final_support_diagnostics, support_diagnostics)
    from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import Accumulator
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import (
        mean_pool, validate_routes)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.grouped_pool import GroupedRouting
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, decoder_forward, event, full_image_loader,
        gather_support, parse_batch, probability_prediction, select_ids,
        upsample_probability, validate_selected)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import region_targets

    if cli.limit < 0 or cli.limit > 450:
        raise ValueError("--limit must be 0..450")
    if cli.topk < 1 or (cli.policy != "global" and cli.topk % 4):
        raise ValueError("--topk must be positive and divisible by four for per_class")
    if not math.isfinite(cli.temperature) or cli.temperature <= 0:
        raise ValueError("--temperature must be finite and positive")
    output = new_output(cli.output)
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   execution_complete=False, image_count=0, phase="load",
                   test_accessed=False, training=False, smoke=bool(cli.limit),
                   labelvote_is_untrained_diagnostic=True,
                   arguments=vars(cli))
    write_json(output / "summary.json", summary)
    started, own_before = time.perf_counter(), source_hash()
    phase, current_image = "load", None
    capture = native_capture = guard = None
    try:
        bundle = load_bundle(cli.protocol, output)
        args, protocol = bundle["args"], bundle["protocol"]
        if bundle["topk"] != 8:
            raise ValueError("Historical E0 J2/J3 replay requires its unchanged Top-8 Reader")
        names = bundle["val_names"]
        if names != protocol["selected_images"] or len(names) != 450:
            raise ValueError("Current Val450 image list/order differs from E0")
        selected_names = names[:cli.limit] if cli.limit else names
        key_root = Path(cli.keys).resolve()
        extra_ledger = {key_root / name: digest(key_root / name)
                        for name in ("paired_bank.pt", "manifest.json", "summary.json")}
        paired, manifest = load_paired_bank(key_root, bundle, allow_smoke=bool(cli.limit))
        token_replay = {key: manifest.get(key) for key in (
            "t128_check", "comparison_tolerance", "f256_storage",
            "t128_storage_interval_max_distance", "t128_storage_uncertain_elements",
            "t128_definite_within_tol_elements", "t128_strict_rounded_replay_failed_images")}
        token_replay["interpretation"] = "Storage compatibility is not exact historical FP32 reproduction"
        summary["support_token_replay"] = token_replay
        policies = ("global", "per_class") if cli.policy == "both" else (cli.policy,)
        retrieval_arms = tuple(f"{key}_{policy}" for key in ("t128", "f256") for policy in policies)
        vote_arms = tuple("labelvote_" + arm for arm in retrieval_arms)
        all_arms = (*BASE_ARMS, "grid_probq", *vote_arms)
        pairs = ("a1_pixel__grid_probq", *(f"grid_probq__{arm}" for arm in vote_arms),
                 *(f"labelvote_t128_{policy}__labelvote_f256_{policy}" for policy in policies))
        counts = Accumulator(arms=all_arms, pairs=pairs)
        support_totals = {arm: {} for arm in retrieval_arms}
        retrieval = {arm: {} for arm in retrieval_arms}
        retrieval64 = {arm: {} for arm in retrieval_arms}
        compatibility = {arm: {} for arm in retrieval_arms}
        support_view = dict(paired, tokens=paired["t128"])
        data = full_image_loader(args.dataset, selected_names, False, args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        native_capture = NativeA1LogitsCapture(bundle["front"])
        guard = PoolGuard(output / "pool_debug", mode=args.pool_mode,
                          capture="failure", check_reference=False)
        artifacts = output / "predictions"
        artifacts.mkdir()
        native_max_abs = q_max_abs = 0.0
        query_total = inactive_total = unlabeled_total = 0
        with guard:
            with (output / "images.jsonl").open("x", encoding="utf-8") as journal:
                for index, batch in enumerate(data):
                    phase = "front_same_pass"
                    pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                    current_image = str(image_names[0])
                    if (len(pre) != 1 or index >= len(selected_names)
                            or current_image != selected_names[index]):
                        raise ValueError("Val image/order/batch differs from E0")
                    guard.set_image(current_image)
                    native_capture.clear()
                    result, feature = capture(pre, post)
                    native_logits = native_capture.take()
                    original_bank = bundle["bank"]
                    event_name = event(current_image)
                    event_id = (original_bank["event_names"].index(event_name)
                                if event_name in original_bank["event_names"] else -1)
                    phase = "historical_replay"
                    historical_query = result["tokens"][0]
                    # E0 operation order is preserved before any new probe.
                    j2_ids = select_ids(historical_query, original_bank, event_id, -1, 8)
                    j3_ids = select_ids(historical_query, original_bank, event_id, -1, 8,
                                        projector=bundle["metric"])
                    validate_selected(j2_ids, original_bank, event_id, -1)
                    validate_selected(j3_ids, original_bank, event_id, -1)
                    j2_support, j3_support = (gather_support(j2_ids, original_bank),
                                               gather_support(j3_ids, original_bank))
                    probability = dict(a1_pixel=result["old_pixel"].float().softmax(1),
                                       direct_region=result["prob"])
                    for arm in ("j1", "j2", "j3"):
                        decoded = decoder_forward(bundle["decoders"][arm], arm,
                                                  feature, result, j2_support, j3_support)
                        probability[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
                    phase = "native_A1_identity_and_Q"
                    baseline, native_replay = native_grid_probq(
                        result, feature, native_logits, target.shape[-2:])
                    probability["grid_probq"] = baseline
                    native_max_abs = max(native_max_abs, float(native_replay["max_abs"]))
                    if native_replay["passed"] is not True:
                        summary["native_a1_replay_failure"] = dict(index=index, image=current_image,
                                                                    **native_replay)
                        raise ValueError("Same-pass native A1 replay differs: " + current_image)
                    phase = "paired_retrieval"
                    t128 = historical_query.float()
                    q_normalization = validate_routes(result["q"], result["ids"], result["count"],
                                                       result["shape"])
                    q_max_abs = max(q_max_abs, float(q_normalization["max_abs"]))
                    routing = GroupedRouting(result["ids"], result["count"])
                    f256, mean_mass = mean_pool(feature.float(), result["q"], result["ids"],
                                               result["count"], routing=routing)
                    f256, mean_mass = f256[0], mean_mass[0]
                    active = (result["valid_regions"][0] & (mean_mass > 0)
                              & torch.isfinite(t128).all(-1) & torch.isfinite(f256).all(-1)
                              & (t128.norm(dim=-1) > 1e-8) & (f256.norm(dim=-1) > 1e-8))
                    query_total += len(active)
                    inactive_total += int((~active).sum())
                    query_keys = {"t128": t128.masked_fill(~active[:, None], 0),
                                  "f256": f256.masked_fill(~active[:, None], 0)}
                    # Both query spaces and support members are jointly eligible.
                    # No F256 tensor ever passes through the historical J3 metric.
                    selected, ranked64, evidence, coverage = {}, {}, {}, {}
                    for key_name, query in query_keys.items():
                        for policy in policies:
                            arm = f"{key_name}_{policy}"
                            ids, scores = select_neighbors(query, paired, key_name=key_name,
                                                           event_id=event_id, image_id=-1,
                                                           topk=cli.topk, policy=policy)
                            ids64, scores64 = select_neighbors(query, paired, key_name=key_name,
                                                               event_id=event_id, image_id=-1,
                                                               topk=64, policy=policy)
                            selected[arm], ranked64[arm] = (ids, scores), (ids64, scores64)
                            evidence[arm] = label_evidence(paired, ids, scores,
                                                          temperature=cli.temperature)
                            probability["labelvote_" + arm], coverage[arm] = _vote_probability(
                                evidence[arm], result, target.shape[-2:], baseline)
                            sum_nested_counts(support_totals[arm], support_diagnostics(ids, support_view))
                    predictions = {arm: probability_prediction(value, target.shape[-2:])
                                   for arm, value in probability.items()}
                    # GT-dependent calls occur only after support IDs, scores,
                    # evidence and every prediction have been fixed.
                    phase = "posthoc_scoring"
                    distribution, weight = region_targets(target, result)
                    gt_labeled = weight[0] > 0
                    unlabeled_total += int((~gt_labeled).sum())
                    normalized_distribution = (distribution[0]
                                               / distribution[0].sum(-1, keepdim=True).clamp_min(1e-12))
                    diagnostic_distribution = normalized_distribution[gt_labeled]
                    for arm, (ids, scores) in selected.items():
                        sum_nested_counts(retrieval[arm], paired_retrieval_diagnostics(
                            diagnostic_distribution, paired, ids[gt_labeled], scores[gt_labeled],
                            temperature=cli.temperature))
                        ids64, scores64 = ranked64[arm]
                        sum_nested_counts(retrieval64[arm], paired_retrieval_diagnostics(
                            diagnostic_distribution, paired, ids64[gt_labeled], scores64[gt_labeled],
                            temperature=cli.temperature))
                        sum_nested_counts(compatibility[arm], _weighted_evidence_diagnostic(
                            evidence[arm], normalized_distribution, weight[0]))
                    image_cms = counts.add(current_image, event_name, target, predictions)
                    phase = "write_image_evidence"
                    arrays = dict(target=target.cpu().numpy().astype(np.int16),
                                  active_queries=active.cpu().numpy(),
                                  query_mass=mean_mass.cpu().numpy().astype(np.float32),
                                  posthoc_query_distribution=distribution[0].cpu().numpy().astype(np.float32),
                                  posthoc_query_valid_mass=weight[0].cpu().numpy().astype(np.float32),
                                  historical_support_j2=j2_ids.cpu().numpy().astype(np.int32),
                                  historical_support_j3=j3_ids.cpu().numpy().astype(np.int32))
                    for arm, prediction in predictions.items():
                        arrays["prediction_" + arm] = prediction.cpu().numpy().astype(np.uint8)
                    for arm, (ids, scores) in selected.items():
                        for suffix, (row_ids, row_scores) in (("", (ids, scores)), ("_64", ranked64[arm])):
                            arrays["support_" + arm + suffix] = row_ids.cpu().numpy().astype(np.int32)
                            arrays["scores_" + arm + suffix] = row_scores.cpu().numpy().astype(np.float32)
                        softlabels = paired["distribution"][ids.clamp_min(0)].float()
                        softlabels = softlabels.masked_fill((ids < 0)[..., None], 0)
                        arrays["softlabels_" + arm] = softlabels.cpu().numpy().astype(np.float32)
                        arrays["evidence_" + arm] = evidence[arm]["posterior"].cpu().numpy().astype(np.float32)
                        arrays["quality_" + arm] = evidence[arm]["quality"].cpu().numpy().astype(np.float32)
                        arrays["supported_mass_" + arm] = coverage[arm].cpu().numpy().astype(np.float32)
                    artifact = artifacts / f"{index:04d}.npz"
                    np.savez_compressed(artifact, **arrays)
                    journal.write(json.dumps(dict(
                        index=index, image=current_image, event=event_name,
                        confusion=image_cms, native_a1_replay=native_replay,
                        q_normalization=q_normalization,
                        active_queries=int(active.sum()), inactive_queries=int((~active).sum()),
                        prediction_file=artifact.relative_to(output).as_posix(),
                        prediction_sha256=digest(artifact)), ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    summary.update(image_count=counts.image_count, phase=phase,
                                   image=current_image, native_a1_replay_max_abs=native_max_abs)
                    write_json(output / "summary.json", summary)
                    if counts.image_count % 10 == 0 or counts.image_count == len(selected_names):
                        print(f"Paired Memory diagnostic {counts.image_count}/{len(selected_names)}", flush=True)
        phase = "reproduction_and_integrity"
        _verify_saved_artifacts(output, selected_names, all_arms, retrieval_arms, len(paired["t128"]))
        results = counts.finish()
        historical = {arm: results["metrics"][arm] for arm in BASE_ARMS}
        smoke = bool(cli.limit)
        reproduction = reproduction_report(historical, bundle["reference"], smoke=smoke,
                                           image_count=counts.image_count)
        e0_reproduction = reproduction_report(historical, protocol["metrics"], smoke=smoke,
                                              image_count=counts.image_count)
        immutable = verify_all(bundle, own_before, extra_ledger)
        pool_ok = bool(guard.restored and not guard.failures)
        formal = formal_gate(counts.image_count, len(names), reproduction, e0_reproduction,
                             immutable, pool_ok, True)
        report = dict(format=FORMAT, complete=formal, execution_complete=True,
                      full_validation_evidence=formal, smoke=smoke, **results,
                      training=False, test_accessed=False,
                      labelvote_is_untrained_diagnostic=True,
                      labelvote_fallback="grid_probq only where projected supported mass <= 1e-8",
                      labelvote_partial_coverage="conditional vote normalized by supported Q mass",
                      labelvote_claim="Neither fitted Reader improvement nor causal Memory benefit",
                      retrieval_policies=list(policies), key_spaces=["t128", "f256"],
                      temperature=cli.temperature, topk=cli.topk, top64=64,
                      top64_policy="Same retrieval policy: global Top64 or four class quotas of 16",
                      comparison_pairs=list(pairs),
                      support_diagnostics={arm: final_support_diagnostics(row)
                                           for arm, row in support_totals.items()},
                      posthoc_retrieval_topk=retrieval, posthoc_retrieval_top64=retrieval64,
                      posthoc_weighted_label_evidence=compatibility,
                      query_eligibility=dict(total=query_total, inactive_both_keys=inactive_total,
                                             active=query_total - inactive_total,
                                             unlabeled_posthoc_excluded=unlabeled_total),
                      posthoc_label_semantics="Normalize positive Q label rows; exclude zero valid-label mass",
                      reproduction=reproduction, e0_reproduction=e0_reproduction,
                      native_a1_replay=dict(same_pass_identity_gate=True, max_abs=native_max_abs,
                                            failed_count=0),
                      q_normalization=dict(passed=True, max_abs=q_max_abs, atol=2e-5, rtol=2e-5),
                      immutable=immutable, pool_guard=guard.report(),
                      protocol_sha256=bundle["protocol_sha256"],
                      paired_bank_sha256=extra_ledger[key_root / "paired_bank.pt"],
                      paired_manifest_sha256=extra_ledger[key_root / "manifest.json"],
                      paired_bank_smoke=manifest["smoke"],
                      support_token_replay=token_replay,
                      original_bank_sha256=bundle["hashes"]["bank_sha256"],
                      source_sha256=own_before, elapsed_seconds=time.perf_counter() - started)
        write_json(output / "result.json", report)
        summary.update(complete=formal, execution_complete=True, full_validation_evidence=formal,
                       smoke=smoke, image_count=counts.image_count,
                       reproduction_passed=reproduction["passed"],
                       e0_reproduction_passed=e0_reproduction["passed"], immutable=immutable,
                       native_a1_replay_max_abs=native_max_abs,
                       phase="complete_smoke" if smoke else ("complete" if formal else "failed_formal_gate"))
        write_json(output / "summary.json", summary)
        if not smoke and not formal:
            raise RuntimeError("Full Val450 reproduction/integrity failed; fixed 1e-5 gate unchanged")
        return report
    except Exception as error:
        _record_failure(output, summary, phase, current_image, error)
        raise
    finally:
        if capture is not None:
            capture.close()
        if native_capture is not None:
            native_capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
