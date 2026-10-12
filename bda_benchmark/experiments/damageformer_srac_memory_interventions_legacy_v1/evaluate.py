"""Full-Val same-weight Memory interventions under a passed legacy/native E0/E1.

Only Train-bank support payloads change. The source front, bank, retrieval metric,
and trained J1/J2/J3 readers are loaded through the original immutable E0 gate.
No query target reaches retrieval or intervention construction.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
    add_source_arguments, append_json, artifact_identity, check_artifact, digest,
    fail_summary, load_verified_bundle, new_output, read_json, verify_unchanged,
    write_json,
)
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import (
    random_matched_ids, reproduction_report, shuffle_support_labels,
    transition_counts, transition_report, validate_selected,
)


BASE_ARMS = ("a1_pixel", "direct_region", "j1", "j2", "j3")
MEMORY_ARMS = ("j2", "j3")
MODES = ("random", "shuffle_labels", "off")


def own_code_hash():
    root = Path(__file__).parent
    checksum = hashlib.sha256()
    for name in ("evaluate.py", "launch.py", "run.sh"):
        checksum.update(name.encode("utf-8"))
        checksum.update((root / name).read_bytes())
    return checksum.hexdigest()


def intervention_supports(selected, support, bank, event_id, image_id, generator):
    """Construct three interventions without changing the original bank or support."""
    from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import gather_support

    random_ids = random_matched_ids(selected, bank, event_id, image_id, generator)
    validate_selected(random_ids, bank, event_id, image_id)
    token, distribution, valid, metadata = support
    randomized = gather_support(random_ids, bank)
    shuffled_labels = shuffle_support_labels(distribution, valid, generator)
    return {
        "random": randomized,
        "shuffle_labels": (token, shuffled_labels, valid, metadata),
        "off": (token, distribution, torch.zeros_like(valid), metadata),
    }, {
        "random_changed_slots": int(((random_ids != selected) & valid).sum()),
        "random_valid_slots": int(valid.sum()),
        "shuffle_changed_slots": int(((shuffled_labels != distribution).any(-1) & valid).sum()),
        "shuffle_valid_slots": int(valid.sum()),
    }


def validate_reference(record, bundle):
    """Require the exact prior full E1 under the same checked E0 and source files."""
    check_artifact(record, bundle)
    if (record.get("format") != "SRAC_HARDQUERY_EVIDENCE_V1"
            or record.get("image_count") != 450
            or record.get("full_validation_evidence") is not True
            or record.get("reproduction", {}).get("passed") is not True
            or record.get("smoke") is not False):
        raise ValueError("Requires a complete, reproduced, full-Val formal E1")
    required = ("files", "inherited_code", "new_code", "frozen_models", "runtime", "protocol")
    if not all(record.get("immutable", {}).get(key) is True for key in required):
        raise ValueError("E1 immutability ledger is incomplete")
    if not all(arm in record.get("metrics", {}) for arm in BASE_ARMS):
        raise ValueError("E1 lacks one of the five original reference arms")


def validate_baselines(actual, bundle, evidence, count):
    """Every original confusion matrix must agree with E1, beyond scalar mIoU."""
    reproduction = reproduction_report(actual, bundle["reference"], smoke=False, image_count=count)
    mismatches = [arm for arm in BASE_ARMS
                  if actual[arm]["confusion"] != evidence["metrics"][arm]["confusion"]]
    if count != 450 or reproduction.get("passed") is not True or mismatches:
        raise RuntimeError("Original predictions do not match passed E0/E1: " + repr(mismatches))
    return reproduction


def parser():
    p = add_source_arguments(argparse.ArgumentParser(__doc__))
    p.add_argument("--protocol", required=True, help="Passed legacy/native all-path E0 directory or JSON")
    p.add_argument("--evidence", required=True, help="Passed full-Val E1 directory or JSON")
    p.add_argument("--intervention_seed", type=int, default=3407)
    return p


@torch.no_grad()
def run(args):
    from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture
    from bda_benchmark.experiments.damageformer_srac_joint123_v1.train import full_image_loader
    from bda_benchmark.experiments.damageformer_srac_joint123_v1.joint_runtime import (
        decoder_forward, upsample_probability,
    )
    from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import gather_support, select_ids
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
        confusion_update, event, parse_batch,
    )
    from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard

    if args.smoke or args.profile != "legacy" or args.pool_mode != "native":
        raise ValueError("This formal comparison requires legacy/native and no smoke")
    if args.intervention_seed < 0:
        raise ValueError("intervention_seed must be nonnegative")
    output, capture = new_output(args.output), None
    try:
        bundle = load_verified_bundle(args)
        evidence_path = Path(args.evidence)
        if evidence_path.is_dir():
            evidence_path = evidence_path / "evidence.json"
        evidence = read_json(evidence_path)
        validate_reference(evidence, bundle)
        own_code_before = own_code_hash()
        evidence_hash = digest(evidence_path)
        capture = FrozenFrontCapture(bundle["front"])
        data = full_image_loader(args.dataset, bundle["val_names"], False, args.workers, args.seed)
        generator = torch.Generator(device="cpu").manual_seed(args.intervention_seed)
        confusion, per_event, transitions, per_event_transitions = {}, {}, {}, {}
        intervention_counts = {arm: {key: 0 for key in (
            "random_changed_slots", "random_valid_slots", "shuffle_changed_slots", "shuffle_valid_slots")}
            for arm in MEMORY_ARMS}
        names = bundle["val_names"]
        processed_count = 0
        with PoolGuard(output / "pool_debug", mode=args.pool_mode) as guard:
            for image_index, batch in enumerate(data):
                pre, post, _, target, batch_names = parse_batch(batch, bundle["device"])
                name = str(batch_names[0])
                if len(pre) != 1 or image_index >= len(names) or name != names[image_index]:
                    raise ValueError("Val image/order/batch differs from checked source")
                guard.set_image(name)
                front_result, feature = capture(pre, post)
                event_name = event(name)
                bank = bundle["bank"]
                event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
                query = front_result["tokens"][0]
                selected = {
                    "j2": select_ids(query, bank, event_id, -1, bundle["topk"]),
                    "j3": select_ids(query, bank, event_id, -1, bundle["topk"], projector=bundle["metric"]),
                }
                for ids in selected.values():
                    validate_selected(ids, bank, event_id, -1)
                supports = {arm: gather_support(ids, bank) for arm, ids in selected.items()}
                probabilities = {
                    "a1_pixel": front_result["old_pixel"].float().softmax(1),
                    "direct_region": front_result["prob"],
                }
                for arm in ("j1", *MEMORY_ARMS):
                    decoded = decoder_forward(bundle["decoders"][arm], arm, feature, front_result,
                                              supports["j2"], supports["j3"])
                    probabilities[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
                for arm in MEMORY_ARMS:
                    variants, counts = intervention_supports(
                        selected[arm], supports[arm], bank, event_id, -1, generator)
                    for key, value in counts.items():
                        intervention_counts[arm][key] += value
                    for mode in MODES:
                        # The same trained J2 or J3 decoder is reused. Off invokes
                        # that decoder's learned-slot fallback, not the separately
                        # trained J1 decoder or a guaranteed return to A1.
                        decoded = decoder_forward(bundle["decoders"][arm], arm, feature,
                                                  front_result, variants[mode], variants[mode])
                        probabilities[f"{arm}_{mode}"] = upsample_probability(decoded["logits"], target.shape[-2:])
                predictions, per_image = {}, {}
                for arm, probability in probabilities.items():
                    if (not torch.isfinite(probability).all()
                            or not torch.allclose(probability.sum(1), torch.ones_like(probability[:, 0]),
                                                  atol=2e-5, rtol=0)):
                        raise ValueError("Invalid probability: " + arm)
                    pred = probability.argmax(1)
                    predictions[arm] = pred
                    cm = torch.zeros(4, 4, dtype=torch.long)
                    confusion_update(cm, target, pred)
                    confusion.setdefault(arm, torch.zeros_like(cm)).add_(cm)
                    per_event.setdefault(event_name, {}).setdefault(arm, torch.zeros_like(cm)).add_(cm)
                    per_image[arm] = cm.tolist()
                for arm in MEMORY_ARMS:
                    normal_label = f"j1_to_{arm}"
                    normal_transition = transition_counts(target, predictions["j1"], predictions[arm])
                    transitions.setdefault(normal_label, torch.zeros_like(normal_transition)).add_(normal_transition)
                    per_event_transitions.setdefault(event_name, {}).setdefault(
                        normal_label, torch.zeros_like(normal_transition)).add_(normal_transition)
                    for mode in MODES:
                        key = f"{arm}_{mode}"
                        for base in ("j1", arm):
                            label = f"{base}_to_{key}"
                            transition = transition_counts(target, predictions[base], predictions[key])
                            transitions.setdefault(label, torch.zeros_like(transition)).add_(transition)
                            per_event_transitions.setdefault(event_name, {}).setdefault(
                                label, torch.zeros_like(transition)).add_(transition)
                append_json(output / "images.jsonl", dict(index=image_index, name=name,
                                                          event=event_name, confusion=per_image))
                processed_count += 1
                if (image_index + 1) % 25 == 0:
                    print(f"Memory intervention {image_index + 1}/{len(names)}", flush=True)
        actual = {arm: metrics(cm) for arm, cm in confusion.items()}
        write_json(output / "diagnostic_unverified.json", dict(
            complete=False, image_count=processed_count, metrics=actual,
            meaning="Retained for mismatch diagnosis only; never a formal comparison"))
        reproduction = validate_baselines(actual, bundle, evidence, processed_count)
        immutable = verify_unchanged(bundle)
        if own_code_hash() != own_code_before or digest(evidence_path) != evidence_hash:
            raise RuntimeError("Intervention code or E1 evidence changed during evaluation")
        result = dict(
            format="SRAC_MEMORY_INTERVENTIONS_LEGACY_V1", complete=True,
            full_validation_evidence=True, image_count=processed_count, smoke=False,
            profile=args.profile, pool_mode=args.pool_mode, intervention_seed=args.intervention_seed,
            **artifact_identity(bundle), evidence_sha256=evidence_hash,
            own_source_sha256=own_code_before, immutable=immutable,
            metrics=actual, reproduction=reproduction,
            per_event={event_name: {arm: metrics(cm) for arm, cm in arms.items()}
                       for event_name, arms in per_event.items()},
            transitions={key: transition_report(value) for key, value in transitions.items()},
            per_event_transitions={event_name: {key: transition_report(value)
                                                for key, value in items.items()}
                                   for event_name, items in per_event_transitions.items()},
            intervention_counts=intervention_counts,
            reference="Original five arms must exactly match the passed E1 confusion matrices",
            off_definition="Same trained J2/J3 reader with all support slots invalid; learned-slot fallback, not J1",
            random_definition="Uniform eligible Train rows, same valid-slot count, cross-event/image exclusion",
            shuffle_definition="Valid label vectors shuffled within each image; support keys and metadata stay fixed",
            interpretation_limit="Random support and shuffled labels change the trained reader input distribution; their score drops measure dependence, not independent retrieval gains or retrained potential",
            test_accessed=False, training=False,
        )
        write_json(output / "summary.json", result)
        return result
    except Exception as error:
        fail_summary(output, error)
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
