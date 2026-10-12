"""Pure, read-only Memory diagnostics. No optimizer, query GT in retrieval, or Test."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def transition_counts(target, old, new, mask=None):
    target, old, new = (v.detach().cpu().long() for v in (target, old, new))
    if target.shape != old.shape or target.shape != new.shape:
        raise ValueError("Transition shapes disagree")
    valid = (target >= 0) & (target < 4)
    if mask is not None:
        valid &= mask.detach().cpu().bool()
    if bool(((old[valid] < 0) | (old[valid] > 3) | (new[valid] < 0) | (new[valid] > 3)).any()):
        raise ValueError("Invalid prediction class")
    return torch.bincount(target[valid] * 16 + old[valid] * 4 + new[valid], minlength=64).reshape(4, 4, 4)


def transition_report(counts):
    counts = torch.as_tensor(counts, dtype=torch.long)
    rows = []
    for c in range(4):
        harm = int(counts[c, c].sum() - counts[c, c, c])
        rescue = int(counts[c, :, c].sum() - counts[c, c, c])
        fp_old = int(counts[:, c, :].sum() - counts[c, c, :].sum())
        fp_new = int(counts[:, :, c].sum() - counts[c, :, c].sum())
        rows.append(dict(gt_count=int(counts[c].sum()), harm=harm, rescue=rescue,
                         net_correct=rescue - harm,
                         false_positive_old=fp_old, false_positive_new=fp_new,
                         delta_false_positive=fp_new - fp_old,
                         old_correct=int(counts[c, c].sum()), new_correct=int(counts[c, :, c].sum())))
    return dict(gt_old_new=counts.tolist(), classes=rows,
                damaged_to_intact_harm=int(counts[2, 2, 1]),
                intact_prediction_to_damaged_rescue=int(counts[2, 1, 2]))


def eligible_mask(bank, event_id, image_id):
    """Identical strict event/image exclusion to the existing J2/J3 selector."""
    return (bank["events"] != int(event_id)) & (bank["images"] != int(image_id))


def validate_selected(selected, bank, event_id, image_id):
    if selected.ndim != 2 or selected.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected integral [region, topk] IDs")
    if bool(((selected < -1) | (selected >= len(bank["tokens"]))).any()):
        raise ValueError("Illegal Memory row ID")
    valid = selected >= 0
    if bool(valid.any()) and not bool(eligible_mask(bank, event_id, image_id)[selected[valid]].all()):
        raise ValueError("Retrieved forbidden same-image/event support")
    sentinel = len(bank["tokens"])
    ordered = selected.masked_fill(~valid, sentinel).sort(dim=-1).values
    duplicate = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] < sentinel)
    if bool(duplicate.any()):
        raise ValueError("Duplicate support row in a query")


def random_matched_ids(selected, bank, event_id, image_id, generator):
    """Uniform eligible rows without replacement; exact valid-slot mask retained."""
    validate_selected(selected, bank, event_id, image_id)
    eligible = eligible_mask(bank, event_id, image_id).nonzero().flatten().cpu()
    selected_cpu = selected.detach().cpu()
    output = torch.full_like(selected_cpu, -1)
    for row in range(len(selected_cpu)):
        positions = (selected_cpu[row] >= 0).nonzero().flatten()
        count = len(positions)
        if count > len(eligible):
            raise ValueError("Cannot preserve support count under exclusion")
        if count:
            choices = eligible[torch.randperm(len(eligible), generator=generator)[:count]]
            output[row, positions] = choices
    return output.to(selected.device)


def shuffle_support_labels(distribution, valid, generator):
    """Shuffle valid label payloads globally; keys/IDs/metadata/validity stay fixed."""
    output = distribution.clone()
    labels = distribution[valid]
    if len(labels):
        order = torch.randperm(len(labels), generator=generator).to(labels.device)
        output[valid] = labels[order]
    return output


@torch.no_grad()
def label_evidence(query, selected, bank, temperature=0.1, projector=None):
    """Cosine/projected-cosine label evidence, NOT Reader attention or calibrated posterior.

    Unsupported rows are all zeros, never a uniform fake prediction. Query GT is
    intentionally absent from the API and is only used by later diagnostics.
    """
    from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import project_batches
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Evidence temperature must be positive and finite")
    if len(query) != len(selected) or len(bank["tokens"]) == 0:
        raise ValueError("Query/support shape mismatch or empty bank")
    if bool(((selected < -1) | (selected >= len(bank["tokens"]))).any()):
        raise ValueError("Invalid support IDs")
    query_key, bank_key = project_batches(projector, query, bank["tokens"])
    valid = selected >= 0
    safe = selected.clamp_min(0).long()
    scores = (query_key[:, None] * bank_key[safe]).sum(-1) / temperature
    supported = valid.any(-1)
    scores = scores.masked_fill(~valid, -torch.inf)
    scores = torch.where(supported[:, None], scores, torch.zeros_like(scores))
    weight = scores.softmax(-1) * valid.to(scores.dtype)
    distribution = bank["distribution"][safe].float()
    evidence = (weight[..., None] * distribution).sum(1)
    if not torch.isfinite(evidence).all():
        raise ValueError("Non-finite label evidence")
    return evidence, supported, weight


def project_label_evidence(evidence, supported, q, ids, shape, output_size):
    """Return supported label mass through Q, retaining unsupported mass explicitly."""
    from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import unpool_region_tokens
    values = torch.cat((evidence.float(), supported.float()[:, None]), dim=-1)[None]
    field = unpool_region_tokens(values, q.float(), ids, shape)
    field = F.interpolate(field, size=output_size, mode="bilinear", align_corners=False)
    label_mass, supported_mass = field[:, :4], field[:, 4]
    if not torch.allclose(label_mass.sum(1), supported_mass, atol=2e-5, rtol=0):
        raise ValueError("Projected evidence mass disagrees with support coverage")
    conditional = label_mass / supported_mass[:, None].clamp_min(1e-12)
    prediction = conditional.argmax(1)
    prediction = prediction.masked_fill(supported_mass <= 1e-8, -1)
    return conditional, supported_mass.clamp(0, 1), prediction


def cohort_evidence(target, reference, candidate, evidence_prediction, supported_mass, label_probability):
    """Pixel-domain evidence/recovery counts on fixed J1 cohorts, not token accuracy."""
    target, reference, candidate, evidence_prediction = [x.detach().cpu() for x in
                                                        (target, reference, candidate, evidence_prediction)]
    mass = supported_mass.detach().cpu()
    probs = label_probability.detach().cpu()
    cohorts = {"damaged_j1_wrong": (target == 2) & (reference != 2),
               "damaged_j1_correct": (target == 2) & (reference == 2),
               "intact_j1_correct": (target == 1) & (reference == 1),
               "destroyed_j1_correct": (target == 3) & (reference == 3)}
    output = {}
    for name, mask in cohorts.items():
        covered = mask & (mass > 1e-8)
        evidence_d = covered & (evidence_prediction == 2)
        n = int(mask.sum())
        output[name] = dict(
            count=n, any_supported_pixels=int(covered.sum()),
            support_mass_sum=float(mass[mask].double().sum()),
            unsupported_mass_sum=float((1 - mass[mask]).double().sum()),
            evidence_damaged=int(evidence_d.sum()),
            evidence_damaged_and_final_damaged=int((evidence_d & (candidate == 2)).sum()),
            final_damaged=int((mask & (candidate == 2)).sum()),
            conditional_damaged_probability_sum=float(probs[:, 2][covered].double().sum()),
        )
    return output


def add_nested_counts(total, row):
    for key, value in row.items():
        if isinstance(value, dict):
            add_nested_counts(total.setdefault(key, {}), value)
        elif isinstance(value, list):
            old = total.setdefault(key, [0] * len(value))
            for i, x in enumerate(value):
                old[i] += x
        else:
            total[key] = total.get(key, 0) + value


def region_retrieval_report(selected, bank, query_distribution, query_valid_mass):
    """GT diagnostic called only AFTER retrieval; separates dominant and soft Damaged."""
    valid = selected >= 0
    neighbors = bank["distribution"][selected.clamp_min(0)].float()
    dominant = query_distribution.argmax(-1)
    usable = query_valid_mass > 0
    output = {}
    for c in range(4):
        mask = usable & (dominant == c)
        same = valid & (neighbors.argmax(-1) == c)
        output[str(c)] = dict(queries=int(mask.sum()), supported_queries=int((mask & valid.any(-1)).sum()),
                             hit_at_k_count=int((mask & same.any(-1)).sum()),
                             valid_slots=int(valid[mask].sum()), same_class_slots=int(same[mask].sum()))
    damaged_mass = query_distribution[:, 2] * query_valid_mass
    d_hit = (valid & (neighbors.argmax(-1) == 2)).any(-1)
    output["all_damaged_label_mass"] = dict(total=float(damaged_mass.double().sum()),
        with_any_support=float(damaged_mass[valid.any(-1)].double().sum()),
        with_damaged_hit=float(damaged_mass[d_hit].double().sum()),
        in_non_damaged_dominant_queries=float(damaged_mass[dominant != 2].double().sum()))
    return output


def distribution_inventory(distribution, mask=None):
    distribution = distribution.detach().cpu().float()
    if mask is not None:
        distribution = distribution[mask.detach().cpu().bool()]
    if distribution.ndim != 2 or distribution.shape[1] != 4:
        raise ValueError("Expected four-class soft distributions")
    dominant = distribution.argmax(-1)
    return dict(regions=len(distribution), dominant_counts=torch.bincount(dominant, minlength=4).tolist(),
                soft_label_sum=distribution.double().sum(0).tolist(),
                damaged_positive_regions=int((distribution[:, 2] > 0).sum()),
                damaged_soft_sum_non_damaged_dominant=float(distribution[dominant != 2, 2].double().sum()))


def reproduction_report(actual, expected, smoke=False, image_count=0):
    """Fixed absolute 1e-5 checks for all scalar/class metrics; never relax on failure."""
    if smoke:
        return dict(passed=None, reason="Two-image smoke is not full-Val reproduction", tolerance=1e-5)
    checks = []
    for name in ("a1_pixel", "direct_region", "j1", "j2", "j3"):
        if name not in actual or name not in expected:
            raise ValueError("Missing reference metric: " + name)
        for field in ("valid_miou", "iou", "precision", "recall", "f1"):
            a, b = actual[name][field], expected[name][field]
            a, b = ([a], [b]) if field == "valid_miou" else (a, b)
            if len(a) != len(b):
                raise ValueError("Reference class count mismatch")
            for index, (x, y) in enumerate(zip(a, b)):
                if x is None or y is None:
                    checks.append(dict(name=f"{name}/{field}/{index}", actual=x, expected=y,
                                       absolute_error=None, passed=x is None and y is None))
                    continue
                error = abs(float(x) - float(y))
                checks.append(dict(name=f"{name}/{field}/{index}", actual=x, expected=y,
                                   absolute_error=error, passed=math.isfinite(error) and error <= 1e-5))
    return dict(passed=image_count == 450 and all(x["passed"] for x in checks),
                image_count=image_count, tolerance=1e-5, checks=checks)
