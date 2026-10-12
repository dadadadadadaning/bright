"""Same-pass, matched-weight support interventions on the passed legacy Val450.

For each image one front forward fixes Q and both Top-8 sets. All five normal
predictions and every intervention are decoded from that same result. E0/E1
identify the source and protocol; historical scalar/class reproduction at 1e-5
is the formal result gate. E1's integer confusion matrices are informative,
not an extra equality gate for a separate native-CUDA replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
    append_json, artifact_identity, check_artifact, digest, load_verified_bundle,
    new_output, read_json, safe_child, verify_unchanged, write_json,
)
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import (
    eligible_mask, reproduction_report,
    transition_counts, transition_report, validate_selected,
)
from .gates import formal_gate


BASE_ARMS = ("a1_pixel", "direct_region", "j1", "j2", "j3")
MEMORY_ARMS = ("j2", "j3")


def own_source_hash():
    root = Path(__file__).parent
    paths = [p for p in root.glob("*.py") if not p.name.startswith("test_")]
    paths += list(root.glob("*.sh"))
    checksum = hashlib.sha256()
    for path in sorted(paths):
        checksum.update(path.name.encode("utf-8"))
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def validate_reference_e1(record, bundle):
    """E1 must be the passed, immutable, full-Val result for this exact E0."""
    check_artifact(record, bundle)
    required = ("files", "inherited_code", "new_code", "frozen_models", "runtime", "protocol")
    if (record.get("format") != "SRAC_HARDQUERY_EVIDENCE_V1"
            or record.get("image_count") != 450
            or record.get("full_validation_evidence") is not True
            or record.get("reproduction", {}).get("passed") is not True
            or record.get("smoke") is not False
            or not all(record.get("immutable", {}).get(k) is True for k in required)
            or not all(arm in record.get("metrics", {}) for arm in BASE_ARMS)):
        raise ValueError("Requires a complete, reproduced, immutable formal Val450 E1")


def check_seeds(random_seeds, shuffle_seed):
    values = tuple(int(s) for s in random_seeds)
    if not values or len(values) != len(set(values)) or any(s < 0 or s >= 2**63 - 1000000 for s in values):
        raise ValueError("Random seeds must be distinct nonnegative 64-bit integers")
    if int(shuffle_seed) < 0 or int(shuffle_seed) >= 2**63 - 1000000:
        raise ValueError("Shuffle seed must be a nonnegative 64-bit integer")
    return values


def uniform_matched_ids(selected, bank, event_id, image_id, generator):
    """Exact eligible-uniform draws without replacement, using O(rows * topk).

    Repeatedly drawing for a slot until its index is different from prior slots
    is uniform over the remaining eligible rows. Rejection is rare for Top-8
    and a normal Memory bank, but still works for a pool as small as topk.
    All draws occur on CPU and consume the supplied generator only.
    """
    validate_selected(selected, bank, event_id, image_id)
    if generator.device.type != "cpu":
        raise ValueError("Uniform support sampler requires a CPU Generator")
    eligible = eligible_mask(bank, event_id, image_id).nonzero().flatten().cpu()
    original = selected.detach().cpu()
    valid = original >= 0
    counts = valid.sum(1)
    count = int(len(eligible))
    if bool((counts > count).any()):
        raise ValueError("Cannot preserve support count under event/image exclusion")
    output = torch.full_like(original, -1)
    if count == 0 or not bool(valid.any()):
        return output.to(selected.device)
    rows, slots = original.shape
    chosen = torch.full((rows, slots), -1, dtype=torch.long)
    for slot in range(int(counts.max())):
        active = (counts > slot).nonzero().flatten()
        draw = torch.randint(count, (len(active),), generator=generator)
        if slot:
            clash = (chosen[active, :slot] == draw[:, None]).any(1)
            while bool(clash.any()):
                draw[clash] = torch.randint(count, (int(clash.sum()),), generator=generator)
                clash = (chosen[active, :slot] == draw[:, None]).any(1)
        chosen[active, slot] = draw
    position = valid.long().cumsum(1).sub(1).clamp_min(0)
    compact = chosen.gather(1, position)
    output[valid] = eligible[compact[valid]]
    return output.to(selected.device)


def bank_consistent_shuffled_labels(selected, distribution, valid, bank, generator):
    """Permute labels once per unique retrieved bank row within this image.

    A bank row can occur in many query Top-8 lists. All of its occurrences get
    the same replacement label, avoiding inconsistent key/label pairings. The
    multiset of soft labels across *unique retrieved rows* is preserved.
    """
    if generator.device.type != "cpu":
        raise ValueError("Label shuffling requires a CPU Generator")
    if not torch.equal(valid, selected >= 0):
        raise ValueError("Support validity differs from selected row IDs")
    output = distribution.clone()
    if not bool(valid.any()):
        return output
    ids_cpu = selected.detach().cpu().long()
    unique = torch.unique(ids_cpu[ids_cpu >= 0], sorted=True)
    permutation = torch.randperm(len(unique), generator=generator)
    source_ids = unique[permutation].to(bank["distribution"].device)
    replacement = bank["distribution"].index_select(0, source_ids).to(output.device)
    position = torch.searchsorted(unique, ids_cpu.clamp_min(0)).to(output.device)
    if not torch.equal(distribution[valid],
                       bank["distribution"][selected[valid]].to(distribution.device)):
        raise ValueError("Support labels differ from immutable bank")
    output[valid] = replacement[position[valid]]
    return output


def intervention_supports(selected, support, bank, event_id, image_id,
                          random_generators, shuffle_generator):
    """Change support payload only; never use query truth or modify source bank."""
    from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import gather_support

    token, distribution, valid, metadata = support
    variants, counts = {}, {}
    for seed, generator in random_generators.items():
        random_ids = uniform_matched_ids(selected, bank, event_id, image_id, generator)
        validate_selected(random_ids, bank, event_id, image_id)
        randomized = gather_support(random_ids, bank)
        if not torch.equal(randomized[2], valid):
            raise ValueError("Random support changed valid-slot mask")
        key = f"random_{seed}"
        variants[key] = randomized
        counts[key] = dict(changed_slots=int(((random_ids != selected) & valid).sum()),
                           valid_slots=int(valid.sum()))
    shuffled_labels = bank_consistent_shuffled_labels(
        selected, distribution, valid, bank, shuffle_generator)
    variants["shuffle_labels"] = (token, shuffled_labels, valid, metadata)
    counts["shuffle_labels"] = dict(
        changed_slots=int(((shuffled_labels != distribution).any(-1) & valid).sum()),
        valid_slots=int(valid.sum()))
    variants["off"] = (token, distribution, torch.zeros_like(valid), metadata)
    counts["off"] = dict(changed_slots=int(valid.sum()), valid_slots=int(valid.sum()))
    return variants, counts


def _add_counts(dest, source):
    for key, row in source.items():
        target = dest.setdefault(key, {name: 0 for name in row})
        for name, value in row.items():
            target[name] += value


def _add_transition(dest, event_dest, event_name, name, value):
    dest.setdefault(name, torch.zeros_like(value)).add_(value)
    event_dest.setdefault(event_name, {}).setdefault(name, torch.zeros_like(value)).add_(value)


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
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import probability_prediction
    from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard

    seeds = check_seeds(args.random_seeds, args.shuffle_seed)
    if args.smoke or args.profile != "legacy" or args.pool_mode != "native":
        raise ValueError("Formal comparison requires legacy/native and full Val450")
    output = new_output(args.output)
    capture = None
    phase, current_image, processed_count = "load_verified_sources", None, 0
    summary = dict(format="SRAC_MEMORY_INTERVENTIONS_SAMEPASS_V1", complete=False,
                   full_validation_evidence=False, image_count=0, phase=phase,
                   random_seeds=list(seeds), shuffle_seed=int(args.shuffle_seed),
                   test_accessed=False, training=False)
    write_json(output / "summary.json", summary)
    try:
        bundle = load_verified_bundle(args)
        evidence_path = Path(args.evidence)
        if evidence_path.is_dir():
            evidence_path = evidence_path / "evidence.json"
        evidence = read_json(evidence_path)
        validate_reference_e1(evidence, bundle)
        evidence_hash = digest(evidence_path)
        source_before = own_source_hash()
        names = bundle["val_names"]
        if names != bundle["protocol"].get("selected_images") or len(names) != 450:
            raise ValueError("Current Val names/order differs from formal E0")
        # E0 uses seed+2 for its validation DataLoader; no Test data is opened.
        data = full_image_loader(args.dataset, names, False, args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        random_generators = {
            arm: {seed: torch.Generator(device="cpu").manual_seed(seed + 1000000 * i)
                  for seed in seeds}
            for i, arm in enumerate(MEMORY_ARMS)
        }
        shuffle_generators = {
            arm: torch.Generator(device="cpu").manual_seed(int(args.shuffle_seed) + 1000000 * i)
            for i, arm in enumerate(MEMORY_ARMS)
        }
        confusions, per_event, transitions, per_event_transitions = {}, {}, {}, {}
        intervention_counts = {arm: {} for arm in MEMORY_ARMS}
        artifact_dir = output / "image_evidence"
        artifact_dir.mkdir()
        phase = "same_pass_validation"
        with PoolGuard(output / "pool_debug", mode=args.pool_mode,
                       capture="failure", check_reference=False) as guard:
            for index, batch in enumerate(data):
                pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                current_image = str(image_names[0])
                if len(pre) != 1 or index >= len(names) or current_image != names[index]:
                    raise ValueError("Val image/order/batch differs from E0")
                guard.set_image(current_image)
                front_result, feature = capture(pre, post)
                bank = bundle["bank"]
                event_name = event(current_image)
                event_id = (bank["event_names"].index(event_name)
                            if event_name in bank["event_names"] else -1)
                query = front_result["tokens"][0]
                selected = {
                    "j2": select_ids(query, bank, event_id, -1, bundle["topk"]),
                    "j3": select_ids(query, bank, event_id, -1, bundle["topk"],
                                     projector=bundle["metric"]),
                }
                for ids in selected.values():
                    validate_selected(ids, bank, event_id, -1)
                support = {arm: gather_support(ids, bank) for arm, ids in selected.items()}
                probability = {
                    "a1_pixel": front_result["old_pixel"].float().softmax(1),
                    "direct_region": front_result["prob"],
                }
                for arm in ("j1", *MEMORY_ARMS):
                    decoded = decoder_forward(bundle["decoders"][arm], arm, feature,
                                              front_result, support["j2"], support["j3"])
                    probability[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
                for arm in MEMORY_ARMS:
                    variants, counts = intervention_supports(
                        selected[arm], support[arm], bank, event_id, -1,
                        random_generators[arm], shuffle_generators[arm])
                    _add_counts(intervention_counts[arm], counts)
                    for mode, payload in variants.items():
                        decoded = decoder_forward(bundle["decoders"][arm], arm,
                                                  feature, front_result, payload, payload)
                        probability[f"{arm}_{mode}"] = upsample_probability(
                            decoded["logits"], target.shape[-2:])
                predictions, image_confusions = {}, {}
                for arm, value in probability.items():
                    pred = probability_prediction(value, target.shape[-2:])
                    predictions[arm] = pred
                    cm = torch.zeros(4, 4, dtype=torch.long)
                    confusion_update(cm, target, pred)
                    confusions.setdefault(arm, torch.zeros_like(cm)).add_(cm)
                    per_event.setdefault(event_name, {}).setdefault(
                        arm, torch.zeros_like(cm)).add_(cm)
                    image_confusions[arm] = cm.tolist()
                image_transitions = {}
                for arm in MEMORY_ARMS:
                    normal = transition_counts(target, predictions["j1"], predictions[arm])
                    _add_transition(transitions, per_event_transitions, event_name,
                                    f"j1_to_{arm}", normal)
                    image_transitions[f"j1_to_{arm}"] = normal
                    for mode in (f"random_{seed}" for seed in seeds):
                        candidate = f"{arm}_{mode}"
                        for baseline in ("j1", arm):
                            changed = transition_counts(target, predictions[baseline],
                                                        predictions[candidate])
                            label = f"{baseline}_to_{candidate}"
                            _add_transition(transitions, per_event_transitions, event_name,
                                            label, changed)
                            image_transitions[label] = changed
                    for mode in ("shuffle_labels", "off"):
                        candidate = f"{arm}_{mode}"
                        for baseline in ("j1", arm):
                            changed = transition_counts(target, predictions[baseline],
                                                        predictions[candidate])
                            label = f"{baseline}_to_{candidate}"
                            _add_transition(transitions, per_event_transitions, event_name,
                                            label, changed)
                            image_transitions[label] = changed
                arrays = {
                    "support_j2": selected["j2"].detach().cpu().numpy().astype(np.int32),
                    "support_j3": selected["j3"].detach().cpu().numpy().astype(np.int32),
                }
                arrays.update({"confusion_" + name: np.asarray(value, dtype=np.int64)
                               for name, value in image_confusions.items()})
                arrays.update({"transition_" + name: value.numpy()
                               for name, value in image_transitions.items()})
                artifact = artifact_dir / f"{index:04d}.npz"
                np.savez_compressed(artifact, **arrays)
                append_json(output / "images.jsonl", dict(
                    index=index, name=current_image, event=event_name,
                    confusion=image_confusions,
                    artifact_file=artifact.relative_to(output).as_posix(),
                    artifact_sha256=digest(artifact)))
                processed_count = index + 1
                summary.update(image_count=processed_count, last_image=current_image)
                if processed_count % 25 == 0:
                    write_json(output / "summary.json", summary)
                    print(f"Memory same-pass {processed_count}/{len(names)}", flush=True)
        phase = "reproduction_and_integrity"
        rows = [json.loads(line) for line in
                (output / "images.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if (len(rows) != processed_count
                or [row.get("name") for row in rows] != names
                or [row.get("index") for row in rows] != list(range(processed_count))):
            raise ValueError("Per-image evidence journal is incomplete or out of order")
        replayed_confusions = {arm: torch.zeros_like(cm) for arm, cm in confusions.items()}
        for row in rows:
            artifact = safe_child(output, row["artifact_file"])
            if digest(artifact) != row["artifact_sha256"]:
                raise ValueError("Per-image evidence artifact hash mismatch: " + str(artifact))
            with np.load(artifact, allow_pickle=False) as saved:
                for arm, raw in row["confusion"].items():
                    value = np.asarray(raw, dtype=np.int64)
                    if value.shape != (4, 4) or not np.array_equal(
                            saved["confusion_" + arm], value):
                        raise ValueError("Per-image confusion disagrees with evidence artifact")
                    replayed_confusions[arm].add_(torch.from_numpy(value))
        if any(not torch.equal(replayed_confusions[arm], cm)
               for arm, cm in confusions.items()):
            raise ValueError("Per-image evidence does not sum to Val confusion")
        journal_hash = digest(output / "images.jsonl")
        actual = {arm: metrics(cm) for arm, cm in confusions.items()}
        # Keep the complete diagnostic even if native replay misses a formal gate.
        write_json(output / "diagnostic_unverified.json", dict(
            complete=False, image_count=processed_count, metrics=actual,
            intervention_counts=intervention_counts,
            reason="Diagnostic only until all formal gates pass"))
        reproduction = reproduction_report(actual, bundle["reference"],
                                           smoke=False, image_count=processed_count)
        e1_metric_comparison = reproduction_report(actual, evidence["metrics"],
                                                    smoke=False, image_count=processed_count)
        immutable = verify_unchanged(bundle)
        source_ok = own_source_hash() == source_before
        evidence_ok = digest(evidence_path) == evidence_hash
        pool_ok = bool(guard.restored and not guard.failures)
        formal = formal_gate(processed_count, len(names), reproduction, immutable,
                             pool_ok, source_ok, evidence_ok)
        report = dict(
            format="SRAC_MEMORY_INTERVENTIONS_SAMEPASS_V1", complete=formal,
            full_validation_evidence=formal, image_count=processed_count,
            random_seeds=list(seeds), shuffle_seed=int(args.shuffle_seed),
            random_generator_rule="For J3, generator seed is listed seed + 1000000; for J2 it is listed seed",
            shuffle_generator_rule="For J3, generator seed is shuffle seed + 1000000; for J2 it is shuffle seed",
            **artifact_identity(bundle), evidence_sha256=evidence_hash,
            image_journal_sha256=journal_hash,
            own_source_sha256=source_before, own_source_unchanged=source_ok,
            evidence_unchanged=evidence_ok, pool_guard=guard.report(), immutable=immutable,
            metrics=actual, reproduction=reproduction,
            e1_metric_comparison_informative_only=e1_metric_comparison,
            per_event={name: {arm: metrics(cm) for arm, cm in values.items()}
                       for name, values in per_event.items()},
            transitions={name: transition_report(value)
                         for name, value in transitions.items()},
            per_event_transitions={name: {key: transition_report(value)
                                          for key, value in values.items()}
                                   for name, values in per_event_transitions.items()},
            intervention_counts=intervention_counts,
            reference="E0 and E1 identity verified; original five arms checked against historical 1e-5 metric gate",
            normal_definition="Original Top-8 from the same forward as every intervention",
            off_definition="Same trained J2/J3 Reader with invalid supports and learned-slot fallback; not J1 or A1",
            random_definition="Uniform eligible Train rows, no same event/image, same valid-slot count; a single seed is only an initial diagnostic",
            shuffle_definition="Soft labels permuted among unique retrieved bank rows within each image; repeated row IDs get one consistent replacement; keys and metadata stay fixed",
            causal_limit="Changes under random/shuffle/off test a frozen Reader's dependence on input; they do not estimate gains from retraining or isolate retrieval quality from distribution shift",
            test_accessed=False, training=False,
        )
        write_json(output / "audit.json", report)
        summary.update(complete=formal, full_validation_evidence=formal,
                       reproduction_passed=reproduction["passed"],
                       phase="complete" if formal else "failed_reproduction_or_integrity",
                       audit_file="audit.json")
        write_json(output / "summary.json", summary)
        if not formal:
            raise RuntimeError("Same-pass reproduction/immutability failed; 1e-5 unchanged")
        return report
    except Exception as error:
        summary.update(complete=False, full_validation_evidence=False,
                       image_count=processed_count, phase=phase, last_image=current_image,
                       error=type(error).__name__ + ": " + str(error))
        write_json(output / "summary.json", summary)
        raise
    finally:
        if capture is not None:
            capture.close()
