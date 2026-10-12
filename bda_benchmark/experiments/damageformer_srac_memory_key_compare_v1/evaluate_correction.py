"""Independent same-pass Val450 evaluation of matched pixel Memory corrections.

The five historical paths retain the original absolute 1e-5 reproduction
gate. Val GT first enters scoring after every prediction and neighbor list
exists. Optional interventions reuse each trained head without retraining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .correction import (EVAL_FORMAT, FIT_FORMAT, FORMAL_STEPS, SMOKE_STEPS,
                         TRAIN_ARMS, assert_evaluation, correction_inputs, make_heads)

BASE_ARMS = ("a1_pixel", "direct_region", "j1", "j2", "j3", "grid_probq")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--keys", required=True, help="Completed paired Train support-key bank")
    p.add_argument("--fit", required=True, help="Completed equal-budget correction fit")
    p.add_argument("--output", required=True, help="Fresh independent Val directory")
    p.add_argument("--limit", type=int, default=0, help="0=Val450; 2=smoke/debug prefix")
    p.add_argument("--interventions", action="store_true", help="Normal/random/shuffled-label/off fixed-weight diagnostics")
    return p


def bank_ledger(keys):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
    root = Path(keys).resolve()
    return {path: digest(path) for path in
            (root / "manifest.json", root / "paired_bank.pt", root / "summary.json")}


@torch.no_grad()
def retrieve_inputs(feature, result, base_probability, bank, bundle, image_name,
                    *, temperature=.1, topk=8, policy="global", intervention="normal", seed=3407):
    from .core import select_neighbors, label_evidence
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import mean_pool
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import event
    mean, mass = mean_pool(feature.float(), result["q"], result["ids"], result["count"])
    t128, f256 = result["tokens"][0].float(), mean[0]
    valid = result["valid_regions"][0].bool() & (mass[0] > 0)
    valid &= torch.isfinite(t128).all(-1) & torch.isfinite(f256).all(-1)
    valid &= (t128.norm(dim=-1) > 1e-8) & (f256.norm(dim=-1) > 1e-8)
    if not bool(torch.isfinite(t128).all() and torch.isfinite(f256).all()):
        raise ValueError("Nonfinite frozen query representation")
    t128, f256 = t128 * valid[:, None], f256 * valid[:, None]
    event_name = event(image_name)
    event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
    image_id = bundle["train_names"].index(image_name) if image_name in bundle["train_names"] else -1
    inputs, support, evidence = {}, {}, {}
    for key, query in (("t128", t128), ("f256", f256)):
        ids, scores = select_neighbors(query, bank, key_name=key, event_id=event_id,
                                       image_id=image_id, topk=topk, policy=policy)
        evidence_bank = bank
        if intervention == "random":
            from .core import matched_random_ids
            ids, scores = matched_random_ids(bank, ids, scores, event_id=event_id,
                                             image_id=image_id, seed=seed, policy=policy)
        elif intervention == "shuffle_labels":
            from .core import shuffled_label_bank
            evidence_bank = shuffled_label_bank(bank, seed=seed)
        elif intervention not in ("normal", "off"):
            raise ValueError("Unknown support intervention")
        item = label_evidence(evidence_bank, ids, scores, temperature=temperature)
        # Empty Q regions are never allowed to provide supported coverage.
        present = valid
        item["supported"] = item["supported"] & present
        item["posterior"] = item["posterior"] * present[:, None]
        item["quality"] = item["quality"] * present[:, None]
        evidence[key] = item
        support[key] = dict(ids=ids, scores=scores)
        inputs[key + "_memory"] = correction_inputs(feature, result, base_probability, item)
    # The capacity control keeps the exact F256-supported coverage mask but
    # zeros all ten label/quality channels. It is not the historical J1 model.
    inputs["query_only"] = correction_inputs(feature, result, base_probability,
                                              evidence["f256"], query_only=True)
    return inputs, support


def _pairs(interventions):
    pairs = ["grid_probq__" + arm for arm in TRAIN_ARMS]
    pairs += ["query_only__t128_memory", "query_only__f256_memory", "t128_memory__f256_memory"]
    if interventions:
        for arm in TRAIN_ARMS[1:]:
            pairs.extend(arm + "__" + arm + "_" + mode for mode in ("random", "shuffle_labels", "off"))
    return tuple(pairs)


def evaluate_correction_heads(bundle, bank, heads, output, *, limit=0, initial=False,
                              temperature=.1, topk=8, policy="global",
                              interventions=False, extra_ledger=None):
    """Run a fresh evaluation directory; a prefix is explicitly nonformal."""
    import numpy as np
    from .shared import source_hash, verify_all
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, new_output, safe_child, write_json)
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import NativeA1LogitsCapture, native_grid_probq
    from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import Accumulator
    from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.core import checked_journal
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, decoder_forward, event, full_image_loader, gather_support,
        parse_batch, probability_prediction, select_ids, upsample_probability, validate_selected)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report
    if type(limit) is not int or not 0 <= limit < 450:
        raise ValueError("limit must be 0=Val450 or a diagnostic prefix 1..449")
    if set(heads) != set(TRAIN_ARMS):
        raise ValueError("Exactly three matched correction heads are required")
    names = bundle["val_names"]
    if len(names) != 450 or list(names) != bundle["protocol"]["selected_images"]:
        raise ValueError("Val order differs from the passed E0")
    selected = names[:limit] if limit else names
    smoke = bool(limit)
    out = new_output(output)
    (out / "predictions").mkdir()
    own_before = source_hash()
    summary = dict(format=EVAL_FORMAT, complete=False, full_validation_evidence=False,
                   image_count=0, smoke=smoke, initial=bool(initial), test_accessed=False,
                   training=False, phase="initialize")
    write_json(out / "summary.json", summary)
    arms = (*BASE_ARMS, *TRAIN_ARMS)
    if interventions:
        arms += tuple(arm + "_" + mode for arm in TRAIN_ARMS[1:]
                      for mode in ("random", "shuffle_labels", "off"))
    counts = Accumulator(arms=arms, pairs=_pairs(interventions))
    saved_heads = {name: value.detach().cpu().clone() for name, value in heads.state_dict().items()}
    training_states = {name: module.training for name, module in heads.named_modules()}
    heads.eval()
    capture = native_capture = guard = None
    phase, current_image = "initialize", None
    native_ok, native_max, identity_ok = True, 0.0, True
    initial_max = {arm: 0.0 for arm in TRAIN_ARMS}
    started = time.perf_counter()
    try:
        args = bundle["args"]
        data = full_image_loader(args.dataset, selected, False, args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        native_capture = NativeA1LogitsCapture(bundle["front"])
        guard = PoolGuard(out / "pool_debug", mode=args.pool_mode, capture="failure", check_reference=False)
        phase = "Val_same_pass"
        with torch.no_grad(), guard, (out / "images.jsonl").open("x", encoding="utf-8") as journal:
            for index, batch in enumerate(data):
                pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                current_image = str(image_names[0])
                if len(pre) != 1 or index >= len(selected) or current_image != selected[index]:
                    raise ValueError("Val image/order/batch differs from E0")
                guard.set_image(current_image)
                native_capture.clear()
                result, feature = capture(pre, post)
                low_logits = native_capture.take()
                old_bank = bundle["bank"]
                event_name = event(current_image)
                event_id = old_bank["event_names"].index(event_name) if event_name in old_bank["event_names"] else -1
                # Preserve historical retrieval/decoder invocation order before
                # introducing additional pool/retrieval/head computations.
                old_ids = {"j2": select_ids(result["tokens"][0], old_bank, event_id, -1, 8),
                           "j3": select_ids(result["tokens"][0], old_bank, event_id, -1, 8,
                                            projector=bundle["metric"])}
                for ids in old_ids.values():
                    validate_selected(ids, old_bank, event_id, -1)
                old_support = {arm: gather_support(ids, old_bank) for arm, ids in old_ids.items()}
                probabilities = dict(a1_pixel=result["old_pixel"].float().softmax(1), direct_region=result["prob"])
                for arm in ("j1", "j2", "j3"):
                    decoded = decoder_forward(bundle["decoders"][arm], arm, feature, result,
                                              old_support["j2"], old_support["j3"])
                    probabilities[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
                base, replay = native_grid_probq(result, feature, low_logits, target.shape[-2:])
                native_ok &= replay.get("passed") is True
                native_max = max(native_max, float(replay.get("max_abs") or 0.0))
                if not native_ok:
                    raise RuntimeError("Same-pass native A1 replay failed on " + current_image)
                probabilities["grid_probq"] = base
                inputs, support = retrieve_inputs(feature, result, base, bank, bundle, current_image,
                                                  temperature=temperature, topk=topk, policy=policy)
                row_stats = {}
                for arm in TRAIN_ARMS:
                    row = heads[arm](**inputs[arm])
                    probabilities[arm] = row["probability"]
                    row_stats[arm] = dict(mean_gate=float(row["gate"].mean()),
                                          supported_coverage=float(inputs[arm]["coverage"].mean()))
                    if initial:
                        same = torch.equal(probabilities[arm], base)
                        identity_ok &= same
                        initial_max[arm] = max(initial_max[arm], float((probabilities[arm] - base).abs().max()))
                if initial and not identity_ok:
                    raise RuntimeError("An initialized correction changed grid_probq")
                if interventions:
                    for mode in ("random", "shuffle_labels"):
                        altered, _ = retrieve_inputs(feature, result, base, bank, bundle, current_image,
                                                      temperature=temperature, topk=topk, policy=policy,
                                                      intervention=mode, seed=args.seed + index)
                        for arm in TRAIN_ARMS[1:]:
                            probabilities[arm + "_" + mode] = heads[arm](**altered[arm])["probability"]
                    for arm in TRAIN_ARMS[1:]:
                        off = heads[arm](**inputs[arm], force_off=True)["probability"]
                        if off is not base:
                            raise RuntimeError("Memory-off must return the original baseline tensor")
                        probabilities[arm + "_off"] = off
                predictions = {arm: probability_prediction(probabilities[arm], target.shape[-2:]) for arm in arms}
                # Val labels first enter any statistic here, after retrieval
                # and predictions (including interventions) are complete.
                cms = counts.add(current_image, event_name, target, predictions)
                arrays = {"target": target.detach().cpu().numpy().astype(np.int16)}
                arrays.update({"prediction_" + arm: predictions[arm].cpu().numpy().astype(np.uint8) for arm in arms})
                arrays.update({"support_" + key: value["ids"].cpu().numpy().astype(np.int32)
                               for key, value in support.items()})
                artifact = out / "predictions" / f"{index:04d}.npz"
                np.savez_compressed(artifact, **arrays)
                journal.write(json.dumps(dict(index=index, image=current_image, event=event_name,
                                               confusion=cms, correction=row_stats, native_a1_replay=replay,
                                               prediction_file=artifact.relative_to(out).as_posix(),
                                               prediction_sha256=digest(artifact)), allow_nan=False) + "\n")
                journal.flush()
                summary["image_count"] = counts.image_count
                if counts.image_count % 10 == 0 or counts.image_count == len(selected):
                    print(f"Memory correction Val {counts.image_count}/{len(selected)}", flush=True)
        phase = "reproduction_and_integrity"
        checked_journal(out, selected, digest, safe_child)
        result_report = counts.finish()
        reproduction = reproduction_report(result_report["metrics"], bundle["reference"],
                                            smoke=smoke, image_count=counts.image_count)
        e0_reproduction = reproduction_report(result_report["metrics"], bundle["protocol"]["metrics"],
                                               smoke=smoke, image_count=counts.image_count)
        immutable = verify_all(bundle, own_before, extra_ledger or {})
        immutable["correction_heads"] = all(torch.equal(value, heads.state_dict()[name].detach().cpu())
                                            for name, value in saved_heads.items())
        pool_ok = bool(guard.restored and not guard.failures)
        complete = bool(counts.image_count == len(selected) and all(immutable.values()) and pool_ok
                        and native_ok and (not initial or identity_ok)
                        and (smoke or (reproduction.get("passed") is True
                                       and e0_reproduction.get("passed") is True)))
        formal = bool(complete and not smoke and counts.image_count == 450)
        report = dict(format=EVAL_FORMAT, complete=complete, full_validation_evidence=formal,
                      **result_report, expected_image_count=450, smoke=smoke, initial=bool(initial),
                      initial_identity_passed=identity_ok if initial else None,
                      initial_probability_max_abs=initial_max if initial else None,
                      reproduction=reproduction, e0_metric_comparison=e0_reproduction,
                      immutable=immutable, native_a1_replay_ok=native_ok, native_a1_replay_max_abs=native_max,
                      pool_ok=pool_ok, pool_guard=guard.report(), source_sha256=own_before,
                      protocol_sha256=bundle["protocol_sha256"],
                      retrieval=dict(topk=topk, policy=policy, temperature=temperature),
                      arms=list(arms), comparison_pairs=list(_pairs(interventions)),
                      interventions=bool(interventions), training=False, test_accessed=False,
                      elapsed_seconds=time.perf_counter() - started,
                      rules=dict(base="Native same-pass A1 probabilities -> same-Q pool/unpool -> resize",
                                 query_only="Matched head, zero evidence, same F256-supported coverage; not historical J1",
                                 off="Exact grid_probq tensor; no probability normalization roundtrip",
                                 random="Matched support count and original cosine-score weights; random eligible identities",
                                 gt="Val GT used only for post-prediction scoring",
                                 correction="Frozen pixel F256/base/evidence -> gated bounded pixel delta"))
        write_json(out / "result.json", report)
        summary.update(complete=complete, full_validation_evidence=formal, phase="complete" if complete else "failed_checks",
                       result_file="result.json", result_sha256=digest(out / "result.json"),
                       reproduction_passed=reproduction.get("passed"), immutable=immutable)
        write_json(out / "summary.json", summary)
        if not complete:
            raise RuntimeError("Correction evaluation failed reproduction/native/integrity/initial identity checks")
        return report
    except Exception as error:
        summary.update(complete=False, full_validation_evidence=False, phase=phase,
                       image=current_image, error=f"{type(error).__name__}: {error}")
        write_json(out / "summary.json", summary)
        raise
    finally:
        if native_capture is not None:
            native_capture.close()
        if capture is not None:
            capture.close()
        for name, module in heads.named_modules():
            module.training = training_states[name]


def load_fit(fit, bundle, keys_ledger, *, allow_smoke=False):
    from .shared import source_hash
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest, safe_child
    root = Path(fit).resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("format") != FIT_FORMAT or summary.get("complete") is not True
            or summary.get("test_accessed") is not False or type(summary.get("smoke")) is not bool
            or (summary["smoke"] and not allow_smoke)):
        raise ValueError("Correction fit is incomplete, Test-accessing or a smoke supplied for formal evaluation")
    steps = SMOKE_STEPS if summary["smoke"] else FORMAL_STEPS
    if (summary.get("steps") != steps
            or summary.get("updates_per_arm") != {arm: steps for arm in TRAIN_ARMS}
            or summary.get("checkpoint_selection") != "last_equal_budget_only"
            or summary.get("protocol_sha256") != bundle["protocol_sha256"]
            or summary.get("source_sha256") != source_hash()
            or summary.get("key_files_sha256") != {str(p): sha for p, sha in keys_ledger.items()}):
        raise ValueError("Correction fit budget/source/keys differ from the current independent evaluation")
    assert_evaluation(summary["initial"], smoke=summary["smoke"], initial=True)
    assert_evaluation(summary["final"], smoke=summary["smoke"])
    diagnostic_ledger = {Path(path): sha for path, sha in summary.get("diagnostic_files_sha256", {}).items()}
    if not summary["smoke"] and not diagnostic_ledger:
        raise ValueError("Formal fit lacks completed first-stage diagnostic provenance")
    if any(digest(path) != sha for path, sha in diagnostic_ledger.items()):
        raise ValueError("First-stage diagnostic evidence changed after correction fitting")
    path = safe_child(root, summary.get("checkpoint_file", ""))
    if path.name != "heads_last.pth" or digest(path) != summary.get("checkpoint_sha256"):
        raise ValueError("Only the unchanged equal-budget heads_last.pth is accepted")
    manifest_path = root / "manifest.json"
    if digest(manifest_path) != summary.get("manifest_sha256"):
        raise ValueError("Correction fit manifest changed")
    state = torch.load(path, map_location=bundle["device"], weights_only=False)
    for key, expected in dict(format=FIT_FORMAT, complete=True, completed_steps=steps,
                              test_accessed=False,
                              smoke=summary["smoke"], protocol_sha256=bundle["protocol_sha256"],
                              source_sha256=source_hash(), manifest_sha256=summary["manifest_sha256"],
                              key_files_sha256=summary["key_files_sha256"],
                              diagnostic_files_sha256=summary.get("diagnostic_files_sha256", {}),
                              architecture=summary["architecture"], retrieval=summary["retrieval"]).items():
        if state.get(key) != expected:
            raise ValueError("Correction checkpoint provenance differs: " + key)
    architecture = summary["architecture"]
    heads = make_heads(architecture["hidden"], architecture["bound"]).to(bundle["device"])
    heads.load_state_dict(state["heads"], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in heads.state_dict().values()):
        raise ValueError("Nonfinite fitted correction parameters")
    ledger = {path: digest(path), manifest_path: digest(manifest_path), root / "summary.json": digest(root / "summary.json"),
              **diagnostic_ledger}
    return heads, summary, ledger


def run(cli):
    from .shared import load_bundle, load_paired_bank
    if cli.limit not in (0, 2):
        raise ValueError("Independent evaluator supports full Val450 or --limit 2 smoke only")
    bundle = load_bundle(cli.protocol, cli.output)
    bank, _manifest = load_paired_bank(cli.keys, bundle, allow_smoke=bool(cli.limit))
    ledger = bank_ledger(cli.keys)
    heads, summary, fit_ledger = load_fit(cli.fit, bundle, ledger, allow_smoke=bool(cli.limit))
    config = summary["retrieval"]
    return evaluate_correction_heads(bundle, bank, heads, cli.output, limit=cli.limit,
                                     interventions=cli.interventions, extra_ledger={**ledger, **fit_ledger},
                                     **config)


if __name__ == "__main__":
    run(parser().parse_args())
