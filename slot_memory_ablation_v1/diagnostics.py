"""Read-only, same-forward raw slot and retrieved-support mechanism audit.

The runner retains the most recent ``super().extract(batch)`` return value as
``runtime.last_extract``. Required keys are ``name``, ``result`` (shape, count,
q, ids), ``building``, ``slots`` (at least soft), and ``pair`` (active). This
module never calls the front, a Reader, retrieval, or a support constructor.
Query Train/Val GT is used only AFTER support preparation for reported labels
and strata. Every returned object is ordinary JSON data, never a Tensor.

No runtime imports happen at module import. ``runtime.diagnostics_slot_ops``
may supply the original slots module; otherwise the already-loaded ``slots``
module is imported when audit_example is called.
"""
from __future__ import annotations

import importlib
import math

SCHEMA = "slot_memory_mechanism_audit_v1"
CLASS_NAMES = ("background", "intact", "damaged", "destroyed")
DEFINITIONS = {
    "GT_usage": "posthoc Train/Val analysis only; never used to select or construct supports",
    "query_labels": "original slot_labels: area-resize valid one-hot GT, route-weighted label mass",
    "parent_GT": "sum two child valid-label counts, normalize by their total",
    "damaged": "class 2 only; damage_any denotes classes 2+3",
    "argmax": "largest soft GT mass; ties use lowest class ID; denominator is GT-valid active queries",
    "internal": "parent GT largest class mass >= 0.90; a semantic purity proxy, not geometry",
    "mixed": "parent GT largest class mass < 0.90, with positive valid-label mass",
    "border": "Q-weighted fraction >= 0.10 of feature-grid cells touching a valid 4-neighbor GT class change; nearest-resized GT",
    "low_b": "Q-weighted building probability mean <= 0.10",
    "hardcase": "GT mixed OR baseline parent argmax differs from GT argmax OR damaged GT mass >= 0.10 with low B",
    "damaged_present": "parent class-2 GT mass >= 0.10",
    "cosine": "raw pooled F256 slot features, both positive route mass and nonzero feature norms",
    "normalized_l2": "Euclidean distance between L2-normalized raw slot features",
    "interpretation": "cosine near 1 in pure regions is expected and is not a failure verdict; soft support labels are not scored as better merely for high purity",
    "n_eff": "(sum pixel weights)^2 / sum squared pixel weights; duplicate routes to the same region at one pixel are combined first",
    "entropy_and_JS": "base-2 entropy; child GT Jensen-Shannon divergence in [0,1]",
    "memory_weights": "valid child route fractions, normalized across selected supports; composition proxy, NOT learned attention",
    "effective_parent_k": "Kish effective count of selected parent valid-child fractions after merging repeated parent IDs; NOT Reader attention ESS",
    "effective_event_and_image_k": "same composition weights grouped by event/image; repeated sources merge",
    "dominant_label_rates": "unweighted fractions of valid child support distributions, argmax tie lowest class ID",
    "hard_negative_proxy": "intact-dominant parent soft label with frozen parent-key cosine >= 0.80; descriptive proxy, particularly relevant on damaged-query strata",
    "case_selection": "stable region-ID order, damaged hardcases then damaged-present then hardcases then other active queries; at most 32",
    "aggregation": "query-weighted pooled sufficient statistics n/sum/sum_sq; no unweighted mean of image means",
}


def _torch():
    return importlib.import_module("torch")


def _safe_ratio(numerator, denominator):
    torch = _torch()
    return numerator / torch.where(denominator > 0, denominator, torch.ones_like(denominator))


def _entropy(probability):
    """Zero mass contributes exactly zero, including completely empty rows."""
    return -(probability * probability.clamp_min(1e-30).log2()).sum(-1)


def effective_route_counts(route_q, route_ids, count):
    """Return mass and Kish n_eff [B,count], without a region-by-pixel array."""
    torch = _torch()
    q = route_q.detach().float()
    ids = route_ids.detach()
    if ids.ndim == 2:
        ids = ids.unsqueeze(0).expand(q.shape[0], -1, -1)
    if (q.ndim != 3 or ids.shape != q.shape or ids.dtype != torch.long
            or count < 1 or bool(((ids < 0) | (ids >= count)).any())
            or bool((q < 0).any()) or not bool(torch.isfinite(q).all())):
        raise ValueError("Expected finite nonnegative [B,S,N] routes and legal IDs")
    # There can be repeated candidate IDs at an image edge. They represent one
    # pixel's combined weight, not independent observations.
    sorted_ids, order = ids.sort(dim=1)
    sorted_q = q.gather(1, order)
    starts = torch.ones_like(sorted_ids, dtype=torch.bool)
    starts[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    group = starts.long().cumsum(1) - 1
    pixel_weights = torch.zeros_like(q).scatter_add_(1, group, sorted_q)
    pixel_ids = torch.zeros_like(ids).scatter_(1, group, sorted_ids)
    mass = q.new_zeros(q.shape[0], count)
    square_mass = torch.zeros_like(mass)
    mass.scatter_add_(1, pixel_ids.flatten(1), pixel_weights.flatten(1))
    square_mass.scatter_add_(1, pixel_ids.flatten(1), pixel_weights.square().flatten(1))
    return mass, _safe_ratio(mass.square(), square_mass)


def grouped_effective_count(ids, weights):
    """Unique count and Kish count [R] after merging equal IDs in each row."""
    torch = _torch()
    if ids.ndim != 2 or weights.shape != ids.shape:
        raise ValueError("Expected matching [queries,candidates] identity/weight arrays")
    valid = (ids >= 0) & (weights > 0)
    safe_ids = torch.where(valid, ids, -1)
    sorted_ids, order = safe_ids.sort(dim=1)
    sorted_weights = torch.where(valid, weights, 0).gather(1, order)
    starts = torch.ones_like(sorted_ids, dtype=torch.bool)
    starts[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    group = starts.long().cumsum(1) - 1
    combined = torch.zeros_like(weights).scatter_add_(1, group, sorted_weights)
    unique_count = (combined > 0).sum(-1)
    return unique_count, _safe_ratio(combined.sum(-1).square(), combined.square().sum(-1))


def _summary(values, mask=None):
    torch = _torch()
    values = values.detach().cpu().double()
    if mask is not None:
        values = values[mask.detach().cpu().bool()]
    values = values.reshape(-1)
    values = values[torch.isfinite(values)]
    n = values.numel()
    if not n:
        return dict(n=0, sum=0.0, sum_sq=0.0, mean=None, std=None, min=None, max=None)
    total, square = float(values.sum()), float(values.square().sum())
    mean = total / n
    return dict(n=n, sum=total, sum_sq=square, mean=mean,
                std=math.sqrt(max(0.0, square / n - mean * mean)),
                min=float(values.min()), max=float(values.max()))


def _stratified(metrics, masks, validity=None):
    """Each field carries its own finite/valid observation denominator."""
    cpu_metrics = {key: value.detach().cpu() for key, value in metrics.items()}
    validity = {key: value.detach().cpu().bool() for key, value in (validity or {}).items()}
    result = {}
    for name, mask in masks.items():
        mask = mask.detach().cpu().bool()
        result[name] = {
            key: _summary(value, mask & validity.get(key, mask))
            for key, value in cpu_metrics.items()
        }
    return result


def _slot_comparison(slot, labels, label_mass):
    torch = _torch()
    feature = slot["slot_f"].detach().float()[0]
    norms = feature.norm(dim=-1)
    pair_valid = slot["slot_valid"].detach()[0].all(-1) & (norms > 0).all(-1)
    unit = torch.nn.functional.normalize(feature, dim=-1)
    cosine = (unit[:, 0] * unit[:, 1]).sum(-1).clamp(-1, 1)
    normal_l2 = (unit[:, 0] - unit[:, 1]).norm(dim=-1)
    mass, n_eff = effective_route_counts(slot["route_q"], slot["route_ids"],
                                         int(slot["slot_count"]))
    mass = mass[0].reshape(-1, 2)
    n_eff = n_eff[0].reshape(-1, 2)
    p = labels[0]
    joint = (p[:, 0] + p[:, 1]) * 0.5
    js = (_entropy(joint) - 0.5 * (_entropy(p[:, 0]) + _entropy(p[:, 1]))).clamp(0, 1)
    label_pair_valid = (label_mass[0] > 0).all(-1)
    fraction = slot["slot_fraction"].detach()[0]
    metrics = dict(raw_cosine=cosine, normalized_l2=normal_l2,
                   cosine_ge_0_99=(cosine >= 0.99).float(), gt_JS=js,
                   slot0_mass=mass[:, 0], slot1_mass=mass[:, 1],
                   slot0_fraction=fraction[:, 0], slot1_fraction=fraction[:, 1],
                   slot0_n_eff=n_eff[:, 0], slot1_n_eff=n_eff[:, 1],
                   slot0_gt_damaged_mass=p[:, 0, 2], slot1_gt_damaged_mass=p[:, 1, 2],
                   slot1_minus_slot0_gt_damaged=p[:, 1, 2] - p[:, 0, 2])
    validity = {key: pair_valid for key in ("raw_cosine", "normalized_l2", "cosine_ge_0_99")}
    validity.update({key: label_pair_valid for key in ("gt_JS", "slot1_minus_slot0_gt_damaged")})
    for role in (0, 1):
        validity[f"slot{role}_gt_damaged_mass"] = label_mass[0, :, role] > 0
        validity[f"slot{role}_n_eff"] = mass[:, role] > 0
    return metrics, validity, p, mass, n_eff


def _border_grid(target, shape):
    """Nearest semantic-edge proxy; no target/region Cartesian product."""
    torch = _torch()
    grid = torch.nn.functional.interpolate(target[:, None].float(), size=shape,
                                          mode="nearest")[:, 0].long()
    edge = torch.zeros_like(grid, dtype=torch.bool)
    vertical = (grid[:, 1:] != grid[:, :-1]) & (grid[:, 1:] != 255) & (grid[:, :-1] != 255)
    horizontal = (grid[:, :, 1:] != grid[:, :, :-1]) & (grid[:, :, 1:] != 255) & (grid[:, :, :-1] != 255)
    edge[:, 1:] |= vertical
    edge[:, :-1] |= vertical
    edge[:, :, 1:] |= horizontal
    edge[:, :, :-1] |= horizontal
    return edge[:, None].float()


def _memory_metrics_chunk(example, runtime, support, row_start, row_stop):
    """Return per-query composition, identity and frozen similarity metrics."""
    torch = _torch()
    ids = support["support_ids"].detach()[row_start:row_stop]
    if ids.ndim != 2:
        raise ValueError("Memory support_ids must be original [R,K] table row IDs")
    if ids.shape[1] == 0:
        raise ValueError("Memory candidates must include at least one padded or real column")
    child_valid = support["support_valid"].detach()[ids.clamp_min(0)] & (ids >= 0)[..., None]
    fraction = support["support_fraction"].detach()[ids.clamp_min(0)].float()
    weights = torch.where(child_valid, fraction, 0)
    labels = support["support_labels"].detach()[ids.clamp_min(0)].float()
    parent_weights = weights.sum(-1)
    parent_valid = parent_weights > 0
    total_weight = parent_weights.sum(-1)
    selected_children = child_valid.sum((-1, -2))
    selected_parents = parent_valid.sum(-1)
    parent_ids = support.get("support_parent_ids", support["support_ids"]).detach()
    if parent_ids.ndim == 1:
        parent_ids = parent_ids[ids.clamp_min(0)]
    else:
        parent_ids = parent_ids[row_start:row_stop]
    if parent_ids.shape != ids.shape:
        raise ValueError("support_parent_ids must map table rows or have shape [R,K]")
    parent_ids = torch.where(ids >= 0, parent_ids, -1)
    unique_parents, parent_eff = grouped_effective_count(parent_ids, parent_weights)
    distribution = _safe_ratio((labels * weights[..., None]).sum((1, 2)), total_weight[:, None])
    metrics = dict(selected_parent_count=selected_parents.float(),
                   unique_parent_count=unique_parents.float(), effective_parent_k=parent_eff,
                   valid_child_count=selected_children.float(),
                   softlabel_damaged_mass=distribution[:, 2],
                   softlabel_damage_any_mass=distribution[:, 2:].sum(-1),
                   softlabel_intact_mass=distribution[:, 1],
                   softlabel_entropy=_entropy(distribution))
    validity = {key: total_weight > 0 for key in
                ("softlabel_damaged_mass", "softlabel_damage_any_mass",
                 "softlabel_intact_mass", "softlabel_entropy")}
    dominant = labels.argmax(-1)
    for cls, name in enumerate(CLASS_NAMES):
        key = "dominant_child_" + name + "_rate"
        metrics[key] = _safe_ratio(((dominant == cls) & child_valid).sum((1, 2)).float(),
                                   selected_children.float())
        validity[key] = selected_children > 0
    parent_label = _safe_ratio((labels * weights[..., None]).sum(-2), parent_weights[..., None])
    intact = (parent_label.argmax(-1) == 1) & parent_valid
    metrics["intact_dominant_parent_rate"] = _safe_ratio(intact.sum(-1).float(), selected_parents.float())
    validity["intact_dominant_parent_rate"] = selected_parents > 0
    key_name = getattr(getattr(runtime, "args", None), "key_name", "t128")
    query_key = example.common["region_t" if key_name == "t128" else "region_f"][0, row_start:row_stop].detach().float()
    if key_name == "t128":
        table_key = support["support_t"].detach().float()
    else:
        # F256 parent retrieval keys must not be replaced with a child mean.
        table_key = None
    bank = getattr(runtime, "bank", None)
    identity_available = (
        isinstance(bank, dict) and "images" in bank and "events" in bank
        and (support.get("support_parent_ids") is not None
             or support["support_t"].shape[0] == len(bank["images"])))
    if identity_available:
        if bool(((parent_ids[parent_valid] < 0) |
                 (parent_ids[parent_valid] >= len(bank["images"]))).any()):
            raise ValueError("Prepared memory support parent identity is outside the original bank")
        for field, prefix in (("images", "image"), ("events", "event")):
            selected_identity = bank[field].detach()[parent_ids.clamp_min(0)]
            selected_identity = torch.where(parent_valid, selected_identity, -1)
            unique, effective = grouped_effective_count(selected_identity, parent_weights)
            metrics["unique_" + prefix + "_count"] = unique.float()
            metrics["effective_" + prefix + "_k"] = effective
        if key_name == "f256":
            selected_key = bank["f256"].detach()[parent_ids.clamp_min(0)].float()
        else:
            selected_key = table_key[ids.clamp_min(0)]
    elif table_key is not None:
        selected_key = table_key[ids.clamp_min(0)]
    else:
        selected_key = None
    if selected_key is not None:
        similarity = (torch.nn.functional.normalize(query_key, dim=-1)[:, None]
                      * torch.nn.functional.normalize(selected_key, dim=-1)).sum(-1).clamp(-1, 1)
        norms_valid = (query_key.norm(dim=-1) > 0)[:, None] & (selected_key.norm(dim=-1) > 0)
        comparable = parent_valid & norms_valid
        intact_comparable = intact & comparable
        metrics["parent_key_cosine"] = _safe_ratio(
            torch.where(comparable, similarity, 0).sum(-1), comparable.sum(-1).float())
        metrics["intact_hard_negative_rate"] = _safe_ratio(
            (intact_comparable & (similarity >= 0.80)).sum(-1).float(), comparable.sum(-1).float())
        metrics["intact_parent_max_cosine"] = similarity.masked_fill(~intact_comparable, -float("inf")).max(-1).values
        validity["parent_key_cosine"] = comparable.any(-1)
        validity["intact_hard_negative_rate"] = comparable.any(-1)
        validity["intact_parent_max_cosine"] = intact_comparable.any(-1)
    return metrics, validity, {
        "identity_available": bool(identity_available),
        "retrieval_key": key_name,
        "candidate_columns": ids.shape[1],
        "weight_basis": "valid_child_route_fraction_composition_proxy",
        "learned_attention_measured": False,
    }, parent_ids, parent_valid


def _memory_metrics(example, runtime, support):
    """Bound key/label gathers to 256 query rows, independent of image size."""
    torch = _torch()
    rows = support["support_ids"].shape[0]
    if rows < 1:
        raise ValueError("Memory diagnostics require at least one query row")
    parts = [_memory_metrics_chunk(example, runtime, support, start, min(start + 256, rows))
             for start in range(0, rows, 256)]
    metrics = {key: torch.cat([part[0][key] for part in parts]) for key in parts[0][0]}
    validity = {key: torch.cat([part[1][key] for part in parts]) for key in parts[0][1]}
    return (metrics, validity, parts[0][2], torch.cat([part[3] for part in parts]),
            torch.cat([part[4] for part in parts]))


def audit_example(example, runtime, prepared_supports, sample_limit=32):
    """Produce a compact JSON audit for one already-prepared Train/Val image."""
    torch = _torch()
    with torch.no_grad():
        return _audit_example(example, runtime, prepared_supports, sample_limit)


def _audit_example(example, runtime, prepared_supports, sample_limit):
    torch = _torch()
    F = torch.nn.functional
    if not isinstance(sample_limit, int) or sample_limit < 0:
        raise ValueError("sample_limit must be a nonnegative integer")
    sample_limit = min(sample_limit, 32)
    data = getattr(runtime, "last_extract", None)
    if not isinstance(data, dict) or data.get("name") != example.image:
        raise ValueError("runtime.last_extract must contain this image's original extract return")
    if data.get("slots", {}).get("soft") is None:
        raise ValueError("The original same-forward soft slots are required")
    ops = getattr(runtime, "diagnostics_slot_ops", None)
    if ops is None:
        ops = importlib.import_module("slots")
    result = data["result"]
    shape, count = tuple(result["shape"]), int(result["count"])
    q, ids = result["q"].detach(), result["ids"].detach()
    target = example.target.detach()
    soft = data["slots"]["soft"]
    labels, label_mass = ops.slot_labels(target, soft["route_q"].detach(),
                                        soft["route_ids"].detach(), shape, int(soft["slot_count"]))
    parent_label_mass = label_mass[0].sum(-1)
    parent_labels = _safe_ratio((labels[0] * label_mass[0, :, :, None]).sum(-2), parent_label_mass[:, None])
    active = data["pair"]["active"].detach().bool()
    if active.shape != (count,):
        raise ValueError("Expected one active flag per parent Q region")
    gt_valid = active & (parent_label_mass > 0)
    building = F.interpolate(data["building"].detach().float(), size=shape,
                             mode="bilinear", align_corners=False).clamp(0, 1)
    binary_entropy = _entropy(torch.stack((1 - building, building), -1))
    values = torch.cat((building, building.square(), binary_entropy,
                        (building <= 0.10).float(), _border_grid(target, shape)), dim=1)
    pooled, parent_mass = ops._sums(values, q, ids, count)
    pooled = _safe_ratio(pooled, parent_mass[..., None])[0]
    b_mean = pooled[:, 0]
    b_var = (pooled[:, 1] - b_mean.square()).clamp_min(0)
    baseline_grid = F.interpolate(example.common["base_probability_full"].detach().float(),
                                  size=shape, mode="bilinear", align_corners=False)
    baseline, _ = ops._sums(baseline_grid, q, ids, count)
    base_labels = _safe_ratio(baseline, parent_mass[..., None])[0]
    dominant = parent_labels.argmax(-1)
    internal = gt_valid & (parent_labels.max(-1).values >= 0.90)
    mixed = gt_valid & ~internal
    low_b = active & (b_mean <= 0.10)
    damaged_present = gt_valid & (parent_labels[:, 2] >= 0.10)
    hardcase = gt_valid & (mixed | (base_labels.argmax(-1) != dominant) | (damaged_present & low_b))
    masks = dict(all_active=active, damaged_argmax=gt_valid & (dominant == 2),
                 damaged_present=damaged_present, hardcase=hardcase,
                 internal=internal, mixed=mixed,
                 border=active & (pooled[:, 4] >= 0.10), low_b=low_b)
    context = dict(b_mean=b_mean, b_variance=b_var, b_binary_entropy=pooled[:, 2],
                   low_b_pixel_fraction=pooled[:, 3], border_pixel_fraction=pooled[:, 4],
                   gt_damaged_mass=parent_labels[:, 2],
                   gt_damage_any_mass=parent_labels[:, 2:].sum(-1),
                   gt_dominant_mass=parent_labels.max(-1).values)
    context_valid = {key: gt_valid for key in ("gt_damaged_mass", "gt_damage_any_mass", "gt_dominant_mass")}
    output = dict(
        schema=SCHEMA, image=str(example.image), event=str(example.event),
        query_split=("train" if example.image in getattr(runtime, "_train_ids", {}) else
                     "val" if example.image in getattr(runtime, "_val_set", set()) else "unverified"),
        query_GT_used_for_supports=False, source_forward_count_added=0,
        query_counts=dict(total=count, active=int(active.sum()), GT_valid_active=int(gt_valid.sum()),
                          GT_missing_active=int((active & ~gt_valid).sum()),
                          GT_argmax={name: int((gt_valid & (dominant == cls)).sum())
                                     for cls, name in enumerate(CLASS_NAMES)}),
        strata_counts={key: int(mask.sum()) for key, mask in masks.items()},
        region_context=_stratified(context, masks, context_valid),
        slots_by_rule={}, memory_by_arm={}, selected_cases=[],
    )
    all_rule_arrays = {}
    for rule in ("soft", "capacity_control"):
        slot = data["slots"].get(rule)
        if slot is None:
            continue
        if rule == "soft":
            role_labels, role_label_mass = labels, label_mass
        else:
            role_labels, role_label_mass = ops.slot_labels(
                target, slot["route_q"].detach(), slot["route_ids"].detach(), shape, int(slot["slot_count"]))
        metrics, validity, p, mass, n_eff = _slot_comparison(slot, role_labels, role_label_mass)
        output["slots_by_rule"][rule] = _stratified(metrics, masks, validity)
        all_rule_arrays[rule] = (
            {key: value.detach().cpu() for key, value in metrics.items()},
            {key: value.detach().cpu() for key, value in validity.items()},
            p.detach().cpu(), mass.detach().cpu(), n_eff.detach().cpu())
    memory_arrays = {}
    for arm, support in prepared_supports.items():
        if str(arm).lower() not in ("d", "r"):
            continue
        metrics, validity, metadata, parent_ids, parent_valid = _memory_metrics(example, runtime, support)
        output["memory_by_arm"][str(arm)] = dict(metadata=metadata, strata=_stratified(metrics, masks, validity))
        memory_arrays[str(arm)] = (
            {key: value.detach().cpu() for key, value in metrics.items()},
            {key: value.detach().cpu() for key, value in validity.items()},
            parent_ids.detach().cpu(), parent_valid.detach().cpu())
    # Only the small selected lists cross into JSON. Full tensors stay transient.
    selection = []
    for mask in (damaged_present & hardcase, damaged_present, hardcase, active):
        for row in mask.nonzero(as_tuple=False).flatten().cpu().tolist():
            if row not in selection:
                selection.append(row)
            if len(selection) >= sample_limit:
                break
        if len(selection) >= sample_limit:
            break
    if not sample_limit:
        selection = []
    cpu_context = {key: value.detach().cpu() for key, value in context.items()}
    cpu_labels = parent_labels.detach().cpu()
    cpu_masks = {key: value.detach().cpu() for key, value in masks.items()}
    cpu_dominant, cpu_gt_valid = dominant.cpu(), gt_valid.cpu()
    for row in selection:
        item = dict(region_id=row, strata=[key for key, mask in cpu_masks.items() if bool(mask[row])],
                    gt_distribution=cpu_labels[row].tolist(),
                    gt_argmax=(int(cpu_dominant[row]) if bool(cpu_gt_valid[row]) else None),
                    context={key: float(value[row]) for key, value in cpu_context.items()},
                    slots={}, memory={})
        for rule, (metrics, validity, p, mass, n_eff) in all_rule_arrays.items():
            item["slots"][rule] = dict(
                raw_cosine=(float(metrics["raw_cosine"][row]) if bool(validity["raw_cosine"][row]) else None),
                normalized_l2=(float(metrics["normalized_l2"][row]) if bool(validity["normalized_l2"][row]) else None),
                gt_JS=(float(metrics["gt_JS"][row]) if bool(validity["gt_JS"][row]) else None),
                gt_distributions=p[row].tolist(), mass=mass[row].tolist(), n_eff=n_eff[row].tolist())
        for arm, (metrics, validity, parent_ids, parent_valid) in memory_arrays.items():
            item["memory"][arm] = {
                key: (float(value[row]) if bool(validity.get(key, cpu_masks["all_active"])[row]) and math.isfinite(float(value[row])) else None)
                for key, value in metrics.items()
            }
            item["memory"][arm]["first_parent_ids"] = parent_ids[row][parent_valid[row]][:8].tolist()
        output["selected_cases"].append(item)
    return output


def _merge_tree(destination, source):
    """Accumulate only sufficient-statistic leaves, preserving exact counts."""
    for key, value in source.items():
        if isinstance(value, dict) and {"n", "sum", "sum_sq"} <= value.keys():
            target = destination.setdefault(key, dict(n=0, sum=0.0, sum_sq=0.0, min=None, max=None))
            target["n"] += value["n"]
            target["sum"] += value["sum"]
            target["sum_sq"] += value["sum_sq"]
            if value["n"]:
                target["min"] = value["min"] if target["min"] is None else min(target["min"], value["min"])
                target["max"] = value["max"] if target["max"] is None else max(target["max"], value["max"])
        elif isinstance(value, dict):
            nested = destination.setdefault(key, {})
            _merge_tree(nested, value)


def _finish_tree(tree):
    if {"n", "sum", "sum_sq"} <= tree.keys():
        mean = tree["sum"] / tree["n"] if tree["n"] else None
        tree["mean"] = mean
        tree["std"] = math.sqrt(max(0.0, tree["sum_sq"] / tree["n"] - mean * mean)) if tree["n"] else None
    else:
        for value in tree.values():
            if isinstance(value, dict):
                _finish_tree(value)


def summarize_audits(audits):
    """Aggregate an iterable/generator of audits with query-weighted moments."""
    result = dict(schema=SCHEMA, image_count=0, definitions=DEFINITIONS,
                  query_counts={}, strata_counts={}, region_context={},
                  slots_by_rule={}, memory_by_arm={})
    seen = set()
    for audit in audits:
        if audit.get("schema") != SCHEMA:
            raise ValueError("Unexpected mechanism audit schema")
        if audit["image"] in seen:
            raise ValueError("Duplicate image audit would distort denominators: " + audit["image"])
        seen.add(audit["image"])
        result["image_count"] += 1
        for key in ("total", "active", "GT_valid_active", "GT_missing_active"):
            result["query_counts"][key] = result["query_counts"].get(key, 0) + audit["query_counts"][key]
        gt_counts = result["query_counts"].setdefault("GT_argmax", {name: 0 for name in CLASS_NAMES})
        for name, value in audit["query_counts"]["GT_argmax"].items():
            gt_counts[name] += value
        for key, value in audit["strata_counts"].items():
            result["strata_counts"][key] = result["strata_counts"].get(key, 0) + value
        for key in ("region_context", "slots_by_rule"):
            _merge_tree(result[key], audit[key])
        for arm, value in audit["memory_by_arm"].items():
            dest = result["memory_by_arm"].setdefault(
                arm, dict(audited_images=0, identity_available_images=0, strata={}))
            dest["audited_images"] += 1
            dest["identity_available_images"] += int(value["metadata"]["identity_available"])
            _merge_tree(dest["strata"], value["strata"])
    _finish_tree(result)
    return result
