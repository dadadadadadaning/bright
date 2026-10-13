"""New paired FP32 Memory retrieval and label-evidence validation, without fitting.

T128 and F256 query keys come from one frozen front forward. Each key uses
one global Top64 ranking; Top8 is its exact prefix. Val labels are consumed
only after every neighbor, evidence value and pixel prediction is fixed.
These untrained label votes do not measure a fitted Reader's contribution.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

FORMAT = "SRAC_MEMORY_REBUILD_DIAGNOSE_V3"
BASE_ARMS = ("a1_pixel", "direct_region", "grid_probq")
RETRIEVAL_ARMS = tuple(f"{key}_top{k}" for key in ("t128", "f256") for k in (8, 64))
ALL_ARMS = (*BASE_ARMS, *("labelvote_" + arm for arm in RETRIEVAL_ARMS))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, help="Unchanged passed E0 source anchor")
    p.add_argument("--keys", required=True, help="Completed V3 paired-key build directory")
    p.add_argument("--output", required=True, help="Fresh diagnostic output directory")
    p.add_argument("--limit", type=int, default=0, help="0 = all Val450; positive = smoke prefix")
    p.add_argument("--temperature", type=float, default=.1)
    return p


def formal_gate(image_count, expected_count, checks, *, smoke=False, bank_smoke=False):
    """A full image count alone cannot authorize a formal validation claim."""
    required = ("protocol", "bank", "split", "labels", "reference", "native",
                "pool", "artifacts", "immutable")
    return bool(not smoke and not bank_smoke and image_count == expected_count == 450
                and isinstance(checks, dict)
                and all(checks.get(name) is True for name in required))


def _weighted_evidence_diagnostic(evidence, distribution, mass):
    """Additive post-hoc soft Q-region diagnostics, separate from pixel accuracy."""
    import torch
    posterior = evidence["posterior"].detach().float()
    supported = evidence["supported"].detach().bool()
    query, weight = distribution.detach().float(), mass.detach().float()
    if posterior.shape != query.shape or weight.shape != (len(query),):
        raise ValueError("Post-hoc evidence/query shapes disagree")
    if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("Nonfinite post-hoc query labels")
    fg = query[:, 1:].sum(-1)
    purity = query[:, 1:].max(-1).values / fg.clamp_min(1e-12)
    partitions = {
        "all": weight > 0,
        "foreground_pure": (weight > 0) & (fg >= .8) & (purity >= .8),
        "foreground_mixed": (weight > 0) & (fg >= .8) & (purity < .8),
        "background_diluted": (weight > 0) & (fg >= .2) & (fg < .8),
        "mostly_background": (weight > 0) & (fg < .2),
    }
    pred, truth = posterior.argmax(-1), query.argmax(-1)
    dot = (posterior * query).sum(-1)
    l1 = (posterior - query).abs().sum(-1)
    result = {}
    for name, mask in partitions.items():
        usable = mask & supported
        intact = usable & (truth == 1)
        damaged = usable & ((truth == 2) | (truth == 3))
        result[name] = dict(
            queries=int(mask.sum()), supported_queries=int(usable.sum()),
            soft_compatibility_sum=float(dot[usable].double().sum()),
            l1_sum=float(l1[usable].double().sum()),
            dominant_agree=int((usable & (pred == truth)).sum()),
            valid_mass=float(weight[mask].double().sum()),
            supported_valid_mass=float(weight[usable].double().sum()),
            mass_weighted_compatibility_sum=float((dot[usable] * weight[usable]).double().sum()),
            intact_queries=int(intact.sum()),
            intact_vote_damaged_or_destroyed=int((intact & ((pred == 2) | (pred == 3))).sum()),
            intact_vote_damaged_or_destroyed_mass_sum=float(posterior[intact, 2:].double().sum()),
            damaged_or_destroyed_queries=int(damaged.sum()),
            damaged_or_destroyed_vote_intact=int((damaged & (pred == 1)).sum()),
            damaged_or_destroyed_vote_intact_mass_sum=float(posterior[damaged, 1].double().sum()),
        )
    return result


def _vote_probability(evidence, result, output_size, baseline):
    import torch
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import project_label_evidence
    conditional, supported_mass, _ = project_label_evidence(
        evidence["posterior"], evidence["supported"], result["q"], result["ids"],
        result["shape"], output_size)
    probability = torch.where((supported_mass > 1e-8)[:, None], conditional, baseline)
    return probability, supported_mass


def _record_failure(output, summary, phase, image, error):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import write_json
    (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    summary.update(complete=False, full_validation_evidence=False,
                   execution_complete=False, phase=phase, image=image,
                   error=f"{type(error).__name__}: {error}", traceback_file="traceback.txt")
    write_json(output / "summary.json", summary)


def _verify_saved_artifacts(output, names, bank, expected_confusions=None):
    """Re-open every hashed artifact, check values, alignment and global counts.

    Selected scores are checked against their stored FP32 keys. This checks
    score provenance; independent exact-ranking tests validate the retriever.
    Top8 must be byte-identical to the first eight columns of the Top64 table.
    """
    import numpy as np
    import torch
    from .core import label_evidence
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest, safe_child
    rows = [json.loads(line) for line in (output / "images.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if ([row.get("image") for row in rows] != list(names)
            or [row.get("index") for row in rows] != list(range(len(names)))):
        raise ValueError("Per-image Val journal is missing or out of order")
    cpu_bank = {key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in bank.items()}
    bank_count = len(cpu_bank["t128"])
    cms = {arm: np.zeros((4, 4), dtype=np.int64) for arm in ALL_ARMS}
    for row in rows:
        artifact = safe_child(output, row["prediction_file"])
        if digest(artifact) != row["prediction_sha256"]:
            raise ValueError("Saved prediction/support SHA mismatch")
        with np.load(artifact, allow_pickle=False) as saved:
            target = saved["target"]
            if target.ndim != 3 or target.shape[0] != 1 or not np.issubdtype(target.dtype, np.integer):
                raise ValueError("Saved artifact lacks integer [1,H,W] target")
            count = len(saved["query_t128"])
            active = saved["active_queries"]
            if active.shape != (count,) or active.dtype != np.bool_:
                raise ValueError("Saved active-query membership invalid")
            for key, width in (("t128", 128), ("f256", 256)):
                query = saved["query_" + key]
                if query.shape != (count, width) or query.dtype != np.float32 or not np.isfinite(query).all():
                    raise ValueError("Saved query keys must be finite paired FP32")
            for name in ("query_mass", "posthoc_query_valid_mass"):
                if (saved[name].shape != (count,) or saved[name].dtype != np.float32
                        or not np.isfinite(saved[name]).all() or (saved[name] < 0).any()):
                    raise ValueError("Saved query mass invalid")
            distribution = saved["posthoc_query_distribution"]
            labeled = saved["posthoc_query_valid_mass"] > 0
            if (distribution.shape != (count, 4) or distribution.dtype != np.float32
                    or not np.isfinite(distribution).all() or (distribution < 0).any()
                    or not np.allclose(distribution[labeled].sum(-1), 1, atol=2e-5, rtol=2e-5)
                    or (distribution[~labeled] != 0).any()):
                raise ValueError("Saved post-hoc labels invalid")
            valid_target = (target >= 0) & (target < 4)
            for arm in ALL_ARMS:
                prediction = saved["prediction_" + arm]
                if (prediction.shape != target.shape or not np.issubdtype(prediction.dtype, np.integer)
                        or ((prediction < 0) | (prediction >= 4)).any()):
                    raise ValueError("Saved prediction shape/class mismatch: " + arm)
                cm = np.bincount((target[valid_target].astype(np.int64) * 4
                                 + prediction[valid_target].astype(np.int64)), minlength=16).reshape(4, 4)
                if row.get("confusion", {}).get(arm) != cm.tolist():
                    raise ValueError("Saved per-image confusion differs from prediction")
                cms[arm] += cm
            for key in ("t128", "f256"):
                if (not np.array_equal(saved[f"support_{key}_top8"], saved[f"support_{key}_top64"][:, :8])
                        or not np.array_equal(saved[f"scores_{key}_top8"], saved[f"scores_{key}_top64"][:, :8])):
                    raise ValueError("Top8 is not the exact global Top64 prefix")
            for arm in RETRIEVAL_ARMS:
                key, suffix = arm.split("_top")
                slots = int(suffix)
                ids, scores = saved["support_" + arm], saved["scores_" + arm]
                if (ids.shape != (count, slots) or ids.shape != scores.shape
                        or not np.issubdtype(ids.dtype, np.integer) or scores.dtype != np.float32
                        or ((ids < -1) | (ids >= bank_count)).any()
                        or not np.isfinite(scores[ids >= 0]).all()
                        or not np.isneginf(scores[ids < 0]).all()
                        or (ids[~active] != -1).any()):
                    raise ValueError("Saved support IDs/scores invalid: " + arm)
                usable = ids >= 0
                if (usable[:, 1:] & ~usable[:, :-1]).any() or (scores[:, 1:] > scores[:, :-1] + 2e-5).any():
                    raise ValueError("Saved global neighbor ranking is not ordered")
                for query_ids in ids:
                    selected_ids = query_ids[query_ids >= 0]
                    if len(np.unique(selected_ids)) != len(selected_ids):
                        raise ValueError("Saved neighbor ranking contains duplicate members")
                if usable.any():
                    tensor_ids = torch.from_numpy(ids.astype(np.int64)).clamp_min(0)
                    events, images = cpu_bank["events"][tensor_ids].numpy(), cpu_bank["images"][tensor_ids].numpy()
                    if ((events[usable] == row["event_id"]).any()
                            or (images[usable] == row["image_id"]).any()):
                        raise ValueError("Saved same-event/image support exclusion violated")
                    q = torch.from_numpy(saved["query_" + key]).double()
                    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    supports = cpu_bank[key][tensor_ids].double()
                    supports = supports / supports.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    expected_scores = (q[:, None] * supports).sum(-1).clamp(-1, 1).numpy()
                    if not np.allclose(scores[usable], expected_scores[usable], atol=2e-5, rtol=2e-5):
                        raise ValueError("Saved cosine score differs from paired FP32 key")
                labels = saved["softlabels_" + arm]
                expected_labels = cpu_bank["distribution"][torch.from_numpy(ids.astype(np.int64)).clamp_min(0)].numpy()
                expected_labels = np.where(usable[..., None], expected_labels, 0)
                if labels.dtype != np.float32 or not np.array_equal(labels, expected_labels):
                    raise ValueError("Saved support labels differ from rebuilt Train bank")
                evidence, quality, mass = (saved["evidence_" + arm], saved["quality_" + arm],
                                           saved["supported_mass_" + arm])
                if (evidence.shape != (count, 4) or quality.shape != (count, 6)
                        or mass.shape != target.shape or evidence.dtype != np.float32
                        or quality.dtype != np.float32 or mass.dtype != np.float32
                        or not np.isfinite(evidence).all() or not np.isfinite(quality).all()
                        or not np.isfinite(mass).all() or (mass < 0).any() or (mass > 1 + 2e-5).any()):
                    raise ValueError("Saved label evidence shape/values invalid")
                checked = label_evidence(cpu_bank, torch.from_numpy(ids.astype(np.int64)),
                                         torch.from_numpy(scores), temperature=row["temperature"])
                if (not np.allclose(evidence, checked["posterior"].numpy(), atol=2e-5, rtol=2e-5)
                        or not np.allclose(quality, checked["quality"].numpy(), atol=2e-5, rtol=2e-5)):
                    raise ValueError("Saved evidence does not match fixed supports")
    if expected_confusions is not None:
        for arm in ALL_ARMS:
            if not np.array_equal(cms[arm], np.asarray(expected_confusions[arm])):
                raise ValueError("Whole-split confusion differs from reopened saved predictions")
    return rows


def run(cli):
    import torch
    with torch.no_grad():
        return _run(cli)


def _run(cli):
    import numpy as np
    import torch
    from .core import select_neighbors, label_evidence, paired_retrieval_diagnostics, pair_neighbor_overlap
    from .paired import extract_pair, extract_labels
    from .shared import load_bundle, load_paired_bank, source_hash, verify_all, paired_ledger, label_inventory
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest, new_output, write_json
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import (
        NativeA1LogitsCapture, native_grid_probq, sum_nested_counts)
    from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import Accumulator, sample_paths
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, event, full_image_loader, parse_batch, probability_prediction)

    if cli.limit < 0 or cli.limit > 450:
        raise ValueError("--limit must be 0..450")
    if not math.isfinite(cli.temperature) or cli.temperature != .1:
        raise ValueError("V3 protocol fixes --temperature at 0.1")
    output = new_output(cli.output)
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   execution_complete=False, image_count=0, phase="load", test_accessed=False,
                   training=False, smoke=bool(cli.limit), historical_reproduction_claim=False,
                   labelvote_is_untrained_diagnostic=True, arguments=vars(cli))
    write_json(output / "summary.json", summary)
    started, own_before = time.perf_counter(), source_hash()
    phase, current_image = "load", None
    capture = native_capture = guard = None
    try:
        bundle = load_bundle(cli.protocol, output)
        args, names = bundle["args"], bundle["val_names"]
        if len(names) != 450 or len(set(names)) != 450 or set(names) & set(bundle["train_names"]):
            raise ValueError("Expected the unchanged disjoint Train2918/Val450 split")
        selected_names = names[:cli.limit] if cli.limit else names
        key_root = Path(cli.keys).resolve()
        extra_ledger = paired_ledger(key_root)
        extra_ledger.update({key_root / name: digest(key_root / name) for name in ("summary.json", "protocol.json")})
        paired, manifest = load_paired_bank(key_root, bundle, allow_smoke=bool(cli.limit))
        val_ledger = {path: digest(path) for name in selected_names
                      for path in sample_paths(args.dataset, name).values()}
        extra_ledger.update(val_ledger)
        pairs = ("a1_pixel__grid_probq", *(f"grid_probq__labelvote_{arm}" for arm in RETRIEVAL_ARMS),
                 "labelvote_t128_top8__labelvote_f256_top8", "labelvote_t128_top64__labelvote_f256_top64")
        counts = Accumulator(arms=ALL_ARMS, pairs=pairs)
        retrieval = {arm: {} for arm in RETRIEVAL_ARMS}
        compatibility = {arm: {} for arm in RETRIEVAL_ARMS}
        overlap = {"top8": {}, "top64": {}}
        data = full_image_loader(args.dataset, selected_names, False, args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        native_capture = NativeA1LogitsCapture(bundle["front"])
        guard = PoolGuard(output / "pool_debug", mode=args.pool_mode, capture="failure", check_reference=False)
        artifacts = output / "predictions"
        artifacts.mkdir()
        native_max_abs = q_max_abs = label_max_abs = 0.0
        query_total = inactive_total = unlabeled_total = 0
        pair_checks = label_checks = True
        with guard:
            with (output / "images.jsonl").open("x", encoding="utf-8") as journal:
                for index, batch in enumerate(data):
                    phase = "front_same_pass"
                    pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                    current_image = str(image_names[0])
                    if len(pre) != 1 or index >= len(selected_names) or current_image != selected_names[index]:
                        raise ValueError("Val image/order/batch differs from the fixed split")
                    guard.set_image(current_image)
                    native_capture.clear()
                    result, feature = capture(pre, post)
                    native_logits = native_capture.take()
                    baseline, native_check = native_grid_probq(result, feature, native_logits, target.shape[-2:])
                    native_max_abs = max(native_max_abs, float(native_check["max_abs"]))
                    if native_check["passed"] is not True:
                        raise ValueError("Same-pass native A1 identity failed: " + current_image)
                    phase = "paired_query_extraction"
                    pair = extract_pair(result, feature)
                    active, query_mass = pair["active"], pair["mass"]
                    pair_checks &= pair["q_check"].get("passed") is True
                    if not pair_checks:
                        raise ValueError("Current Q/paired-key reference check failed")
                    q_max_abs = max(q_max_abs, float(pair["q_check"]["max_abs"]))
                    query_total += len(active)
                    inactive_total += int((~active).sum())
                    query_keys = {key: pair[key].masked_fill(~active[:, None], 0) for key in ("t128", "f256")}
                    event_name = event(current_image)
                    event_id = paired["event_names"].index(event_name) if event_name in paired["event_names"] else -1
                    if current_image in bundle["train_names"]:
                        raise ValueError("Val query also exists in Train")
                    image_id = -1  # Verified absent from Train; never an unknown identity.
                    probability = dict(a1_pixel=result["old_pixel"].float().softmax(1),
                                       direct_region=result["prob"], grid_probq=baseline)
                    selected, evidence, coverage = {}, {}, {}
                    phase = "fixed_global_retrieval"
                    for key, query in query_keys.items():
                        ids64, scores64 = select_neighbors(query, paired, key_name=key,
                            event_id=event_id, image_id=image_id, topk=64, policy="global")
                        for topk in (8, 64):
                            arm = f"{key}_top{topk}"
                            ids, scores = ids64[:, :topk].clone(), scores64[:, :topk].clone()
                            selected[arm] = (ids, scores)
                            evidence[arm] = label_evidence(paired, ids, scores, temperature=cli.temperature)
                            probability["labelvote_" + arm], coverage[arm] = _vote_probability(
                                evidence[arm], result, target.shape[-2:], baseline)
                    predictions = {arm: probability_prediction(value, target.shape[-2:])
                                   for arm, value in probability.items()}
                    # No GT-derived values are computed until retrieval and ALL
                    # predictions are final. Image dimensions above are metadata.
                    phase = "posthoc_labels_and_scoring"
                    label = extract_labels(target, result)
                    distribution, label_mass = label["distribution"], label["label_mass"]
                    label_checks &= label["label_check"].get("passed") is True
                    if not label_checks:
                        raise ValueError("Independent query label-reference check failed")
                    label_max_abs = max(label_max_abs, float(label["label_check"]["max_abs"]))
                    labeled = label_mass > 0
                    unlabeled_total += int((~labeled).sum())
                    for arm, (ids, scores) in selected.items():
                        sum_nested_counts(retrieval[arm], paired_retrieval_diagnostics(
                            distribution[labeled], paired, ids[labeled], scores[labeled], temperature=cli.temperature))
                        sum_nested_counts(compatibility[arm], _weighted_evidence_diagnostic(
                            evidence[arm], distribution, label_mass))
                    for topk in (8, 64):
                        sum_nested_counts(overlap[f"top{topk}"], pair_neighbor_overlap(
                            selected[f"t128_top{topk}"][0], selected[f"f256_top{topk}"][0]))
                    image_cms = counts.add(current_image, event_name, target, predictions)
                    phase = "write_image_evidence"
                    arrays = dict(target=target.cpu().numpy().astype(np.int16),
                                  active_queries=active.cpu().numpy(), query_mass=query_mass.cpu().numpy().astype(np.float32),
                                  query_t128=pair["t128"].cpu().numpy().astype(np.float32),
                                  query_f256=pair["f256"].cpu().numpy().astype(np.float32),
                                  posthoc_query_distribution=distribution.cpu().numpy().astype(np.float32),
                                  posthoc_query_valid_mass=label_mass.cpu().numpy().astype(np.float32))
                    for arm, prediction in predictions.items():
                        arrays["prediction_" + arm] = prediction.cpu().numpy().astype(np.uint8)
                    for arm, (ids, scores) in selected.items():
                        arrays["support_" + arm] = ids.cpu().numpy().astype(np.int32)
                        arrays["scores_" + arm] = scores.cpu().numpy().astype(np.float32)
                        softlabels = paired["distribution"][ids.clamp_min(0)].float().masked_fill((ids < 0)[..., None], 0)
                        arrays["softlabels_" + arm] = softlabels.cpu().numpy().astype(np.float32)
                        arrays["evidence_" + arm] = evidence[arm]["posterior"].cpu().numpy().astype(np.float32)
                        arrays["quality_" + arm] = evidence[arm]["quality"].cpu().numpy().astype(np.float32)
                        arrays["supported_mass_" + arm] = coverage[arm].cpu().numpy().astype(np.float32)
                    artifact = artifacts / f"{index:04d}.npz"
                    np.savez_compressed(artifact, **arrays)
                    journal.write(json.dumps(dict(index=index, image=current_image, event=event_name,
                        event_id=event_id, image_id=image_id, temperature=cli.temperature,
                        confusion=image_cms, native_a1_identity=native_check,
                        q_check=pair["q_check"], label_reference_check=label["label_check"],
                        active_queries=int(active.sum()), inactive_queries=int((~active).sum()),
                        prediction_file=artifact.relative_to(output).as_posix(),
                        prediction_sha256=digest(artifact)), ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    summary.update(image_count=counts.image_count, phase=phase, image=current_image)
                    write_json(output / "summary.json", summary)
                    if counts.image_count % 10 == 0 or counts.image_count == len(selected_names):
                        print(f"New paired Memory diagnostic {counts.image_count}/{len(selected_names)}", flush=True)
        phase = "artifact_and_integrity_validation"
        confusions = {arm: cm.tolist() for arm, cm in counts.cms.items()}
        _verify_saved_artifacts(output, selected_names, paired, confusions)
        immutable = verify_all(bundle, own_before, extra_ledger)
        pool_ok = bool(guard.restored and not guard.failures)
        checks = dict(protocol=True, bank=True, split=True, labels=bool(label_checks),
                      reference=bool(pair_checks and label_checks), native=True, pool=pool_ok,
                      artifacts=True, immutable=bool(immutable) and all(value is True for value in immutable.values()))
        smoke = bool(cli.limit)
        formal = formal_gate(counts.image_count, len(names), checks, smoke=smoke, bank_smoke=manifest["smoke"])
        report = dict(format=FORMAT, complete=formal, execution_complete=True,
            full_validation_evidence=formal, smoke=smoke, **counts.finish(), confusion=confusions,
            training=False, test_accessed=False, historical_reproduction_claim=False,
            labelvote_is_untrained_diagnostic=True,
            labelvote_claim="Untrained retrieval label evidence, not fitted Reader improvement or causal Memory benefit",
            labelvote_fallback="grid_probq where projected supported mass <= 1e-8",
            labelvote_partial_coverage="Conditional vote normalized by supported Q mass; coverage saved separately",
            key_spaces=["t128", "f256"], key_storage="FP32", retrieval_policy="global",
            query_forward_count=counts.image_count, historical_reader_forward_count=0,
            bank_label_inventory=label_inventory(paired),
            coverage_scope="Fixed original Train support members; does not measure full-Train candidate-selection losses",
            topk=[8, 64], temperature=cli.temperature, top8_is_top64_prefix=True,
            comparison_pairs=list(pairs), posthoc_retrieval=retrieval,
            posthoc_weighted_label_evidence=compatibility, paired_neighbor_overlap=overlap,
            overlap_semantics="Jaccard sum divided by all query_count; unsupported queries contribute zero",
            query_eligibility=dict(total=query_total, inactive_both_keys=inactive_total,
                active=query_total - inactive_total, unlabeled_posthoc_excluded=unlabeled_total),
            validation=checks, immutable=immutable, pool_guard=guard.report(),
            native_a1_identity=dict(same_pass=True, max_abs=native_max_abs, passed=True),
            q_reference=dict(passed=bool(pair_checks), max_abs=q_max_abs),
            label_reference=dict(passed=bool(label_checks), max_abs=label_max_abs),
            source_anchor_protocol_sha256=bundle["protocol_sha256"],
            val_source_image_fingerprints={str(path): value for path, value in val_ledger.items()},
            rebuild_protocol_sha256=manifest["rebuild_protocol_sha256"],
            paired_bank_sha256=extra_ledger[key_root / "paired_bank.pt"],
            paired_manifest_sha256=extra_ledger[key_root / "manifest.json"],
            paired_bank_smoke=manifest["smoke"], source_sha256=own_before,
            elapsed_seconds=time.perf_counter() - started)
        write_json(output / "result.json", report)
        summary.update(complete=formal, execution_complete=True, full_validation_evidence=formal,
            validation=checks, image_count=counts.image_count, immutable=immutable,
            rebuild_protocol_sha256=manifest["rebuild_protocol_sha256"],
            phase="complete_smoke" if smoke else ("complete" if formal else "failed_formal_gate"))
        write_json(output / "summary.json", summary)
        if not smoke and not formal:
            raise RuntimeError("New paired Val450 protocol/integrity gate failed")
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
