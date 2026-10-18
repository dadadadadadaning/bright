"""Descriptive candidate evidence diagnostics; never trains or alters retrieval.

Queries use role 1 (the soft building slot); every selected parent supplies both
roles, exactly as in the current Reader. Fixed cosine/0.1 label votes are NOT
trained Reader attention or pixel predictions. Query GT appears only in metrics
whose names start ``posthoc_``; local GT evidence is explicitly an oracle.
Missing evidence is NaN, not a zero-quality observation. Counts remain zero.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

CLASSES = ("background", "intact", "damaged", "destroyed")


def _validate_labels(name: str, labels: Tensor, valid: Tensor, *, allow_zero: bool = False) -> Tensor:
    if labels.shape != (*valid.shape, 4):
        raise ValueError(f"{name}: expected labels [N,2,4]")
    chosen = labels[valid]
    if not bool(torch.isfinite(chosen).all()) or bool((chosen < 0).any()):
        raise ValueError(f"{name}: valid labels must be finite and nonnegative")
    mass = labels.sum(-1)
    normal = (mass - 1).abs() <= 2e-5
    if allow_zero:
        normal = normal | (mass == 0)
    if not bool(normal[valid].all()):
        raise ValueError(f"{name}: valid labels must sum to one (zero only for unlabelled GT)")
    return valid & (mass > 0)


def _validate_features(name: str, features: Tensor, valid: Tensor, channels: int) -> None:
    if valid.ndim != 2 or valid.shape[1] != 2 or valid.dtype != torch.bool:
        raise ValueError(f"{name}: validity must be bool [N,2]")
    if features.shape != (*valid.shape, channels) or not features.is_floating_point():
        raise ValueError(f"{name}: feature shape/type mismatch")
    if not bool(torch.isfinite(features[valid]).all()):
        raise ValueError(f"{name}: valid features must be finite")


def _validate_ids(name: str, ids: Tensor, rows: int, members: int) -> None:
    if ids.ndim != 2 or ids.shape[0] != rows or ids.dtype != torch.long:
        raise ValueError(f"{name}: IDs must be int64 [R,K]")
    if bool(((ids < -1) | (ids >= members)).any()):
        raise ValueError(f"{name}: out of range; only -1 is padding")


def inventory(labels: Tensor, valid: Tensor, images: Sequence[str], events: Sequence[str]) -> dict:
    """Stored-bank inventory, not full-Train candidate coverage."""
    if valid.ndim != 2 or valid.shape[1] != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [M,2]")
    _validate_labels("inventory", labels, valid)
    if len(images) != valid.shape[0] or len(events) != valid.shape[0]:
        raise ValueError("one image/event identity per parent required")
    present = valid.any(-1).cpu().tolist()
    chosen = labels[valid].float()
    return {
        "scope": "stored parent bank only; not full-Train coverage",
        "stored_parent_count": valid.shape[0],
        "valid_parent_count": int(valid.any(-1).sum().item()),
        "valid_child_slot_count": int(valid.sum().item()),
        "unique_source_images": len({str(v) for v, ok in zip(images, present) if ok}),
        "unique_source_events": len({str(v) for v, ok in zip(events, present) if ok}),
        "class_order": list(CLASSES),
        "child_slot_soft_mass": chosen.sum(0).cpu().tolist(),
        "child_slot_dominant_counts": torch.bincount(chosen.argmax(-1), minlength=4).cpu().tolist(),
    }


def _table_rows(query_f: Tensor, query_gt: Tensor, query_valid: Tensor,
                ids: Tensor, table_f: Tensor, labels: Tensor, table_valid: Tensor,
                *, images=None, events=None, allow_zero=False, chunk_size=32) -> dict[str, Tensor]:
    regions, _, channels = query_f.shape
    label_valid = _validate_labels("candidate", labels, table_valid, allow_zero=allow_zero)
    query_labelled = query_valid[:, 1] & (query_gt[:, 1].sum(-1) > 0)
    result: dict[str, list[Tensor]] = {}
    nan = float("nan")

    def add(name, value):
        result.setdefault(name, []).append(value.float())

    for start in range(0, regions, chunk_size):
        end = min(regions, start + chunk_size)
        batch = end - start
        fathers = ids[start:end]
        k = fathers.shape[1]
        safe = fathers.clamp_min(0)
        if table_f.shape[0] and k:
            structural = ((fathers >= 0)[..., None] & table_valid[safe])
            observed = ((fathers >= 0)[..., None] & label_valid[safe])
            selected_f = torch.where(structural[..., None], table_f[safe], 0).float().reshape(batch, k * 2, channels)
            selected_y = torch.where(observed[..., None], labels[safe], 0).float().reshape(batch, k * 2, 4)
        else:
            structural = torch.zeros((batch, k, 2), dtype=torch.bool, device=query_f.device)
            observed = structural.clone()
            selected_f = query_f.new_zeros((batch, k * 2, channels)).float()
            selected_y = query_f.new_zeros((batch, k * 2, 4)).float()
        parent_ok = structural.any(-1)
        count = observed.flatten(1).sum(-1)
        add("valid_parent_count", parent_ok.sum(-1))
        add("valid_child_slot_count", structural.flatten(1).sum(-1))
        add("labelled_child_slot_count", count)
        add("has_candidate_labels", count > 0)
        identity_rows = fathers.cpu().tolist()
        mask_rows = parent_ok.cpu().tolist()
        selected_ids = [[i for i, ok in zip(row, mask) if ok] for row, mask in zip(identity_rows, mask_rows)]
        unique_parent = query_f.new_tensor([len(set(row)) for row in selected_ids])
        add("unique_parent_count", unique_parent)
        parent_count = parent_ok.sum(-1).float()
        add("duplicate_parent_fraction", torch.where(parent_count > 0, 1 - unique_parent / parent_count.clamp_min(1), nan))
        if images is not None:
            unique_image = query_f.new_tensor([len({str(images[i]) for i in row}) for row in selected_ids])
            unique_event = query_f.new_tensor([len({str(events[i]) for i in row}) for row in selected_ids])
            add("unique_source_images", unique_image)
            add("unique_source_events", unique_event)
            add("duplicate_image_fraction", torch.where(unique_parent > 0, 1 - unique_image / unique_parent.clamp_min(1), nan))
        mask = observed.flatten(1)
        mean_y = selected_y.sum(1) / count.clamp_min(1)[:, None]
        mean_y = torch.where((count > 0)[:, None], mean_y, nan)
        if k:
            max_y = selected_y.masked_fill(~mask[..., None], -torch.inf).amax(1)
            max_y = torch.where((count > 0)[:, None], max_y, nan)
        else:
            max_y = query_f.new_full((batch, 4), nan).float()
        for cls, name in enumerate(CLASSES):
            add("candidate_mean_" + name, mean_y[:, cls])
            add("candidate_max_" + name, max_y[:, cls])
        add("candidate_background80_fraction", torch.where(count > 0, ((selected_y[..., 0] >= .8) & mask).sum(-1) / count.clamp_min(1), nan))
        add("candidate_damaged50_fraction", torch.where(count > 0, ((selected_y[..., 2] >= .5) & mask).sum(-1) / count.clamp_min(1), nan))
        for cls, name in ((1, "intact"), (2, "damaged")):
            class_parent = ((selected_y[..., cls] >= .5) & mask).reshape(batch, k, 2).any(-1)
            add(name + "50_parent_count", class_parent.sum(-1))
            if images is not None:
                class_masks = class_parent.cpu().tolist()
                class_ids = [[i for i, ok in zip(row, flags) if ok] for row, flags in zip(identity_rows, class_masks)]
                add(name + "50_unique_images", query_f.new_tensor([len({str(images[i]) for i in row}) for row in class_ids]))
                add(name + "50_unique_events", query_f.new_tensor([len({str(events[i]) for i in row}) for row in class_ids]))
        support_fg_mass = selected_y[..., 1:].sum(-1)
        fg_mask = mask & (support_fg_mass > 0)
        support_fg = selected_y[..., 1:] / support_fg_mass.clamp_min(1e-20)[..., None]
        fg_count = fg_mask.sum(-1)
        add("foreground_labelled_slot_count", fg_count)
        for cls, name in enumerate(CLASSES[1:]):
            fg_mean = (support_fg[..., cls] * fg_mask).sum(-1) / fg_count.clamp_min(1)
            add("candidate_foreground_conditional_mean_" + name, torch.where(fg_count > 0, fg_mean, nan))

        q = torch.where(query_valid[start:end, 1, None], query_f[start:end, 1], 0).float()
        qnorm = q.norm(dim=-1)
        snorm = selected_f.norm(dim=-1)
        cosine_valid = mask & (snorm > 1e-12) & (qnorm > 1e-12)[:, None]
        cosine = (selected_f * q[:, None]).sum(-1) / (snorm * qnorm[:, None]).clamp_min(1e-12)
        cosine = cosine.clamp(-1, 1)
        cos_count = cosine_valid.sum(-1)
        add("cosine_valid_slot_count", cos_count)
        add("has_cosine_evidence", cos_count > 0)
        if k:
            score = (cosine / .1).masked_fill(~cosine_valid, -1e30)
            weight = score.softmax(-1) * cosine_valid.float()
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-20)
        else:
            weight = cosine
        vote = (weight[..., None] * selected_y).sum(1)
        vote = torch.where((cos_count > 0)[:, None], vote, nan)
        for cls, name in enumerate(CLASSES):
            add("cosine_vote_" + name, vote[:, cls])
        add("cosine_vote_entropy_nats", -(vote * vote.clamp_min(1e-20).log()).sum(-1))
        for cls, name in ((1, "intact"), (2, "damaged")):
            exposure = cosine_valid & (cosine > .8) & (selected_y[..., cls] >= .5)
            add("similar_" + name + "50_fraction", torch.where(cos_count > 0, exposure.sum(-1) / cos_count.clamp_min(1), nan))
            add("has_similar_" + name + "50", torch.where(cos_count > 0, exposure.any(-1).float(), nan))
        qy = query_gt[start:end, 1].float()
        qobserved = query_labelled[start:end]
        pair_ok = qobserved & (cos_count > 0)
        add("posthoc_vote_tv_compatibility", torch.where(pair_ok, 1 - .5 * (vote - qy).abs().sum(-1), nan))
        add("posthoc_vote_soft_dot_not_accuracy", torch.where(pair_ok, (vote * qy).sum(-1), nan))
        add("posthoc_mean_soft_dot_not_accuracy", torch.where(qobserved & (count > 0), (mean_y * qy).sum(-1), nan))
        if k:
            best_tv = (1 - .5 * (selected_y - qy[:, None]).abs().sum(-1)).masked_fill(~mask, -torch.inf).amax(-1)
        else:
            best_tv = query_f.new_full((batch,), nan)
        add("posthoc_best_candidate_tv_compatibility", torch.where(qobserved & (count > 0), best_tv, nan))
        qfg = qy[:, 1:].sum(-1)
        vfg = vote[:, 1:].sum(-1)
        qcond = qy[:, 1:] / qfg.clamp_min(1e-20)[:, None]
        vcond = vote[:, 1:] / vfg.clamp_min(1e-20)[:, None]
        add("posthoc_foreground_conditional_vote_tv", torch.where(pair_ok & (qfg > 0) & (vfg > 0), 1 - .5 * (qcond - vcond).abs().sum(-1), nan))
        if k:
            best_fg_tv = (1 - .5 * (support_fg - qcond[:, None]).abs().sum(-1)).masked_fill(~fg_mask, -torch.inf).amax(-1)
        else:
            best_fg_tv = query_f.new_full((batch,), nan)
        add("posthoc_best_foreground_conditional_candidate_tv", torch.where(qobserved & (qfg > 0) & (fg_count > 0), best_fg_tv, nan))
        add("cosine_vote_foreground_mass", vfg)
        add("cosine_vote_conditional_damaged", torch.where((cos_count > 0) & (vfg > 0), vote[:, 2] / vfg.clamp_min(1e-20), nan))
    return {name: torch.cat(values) for name, values in result.items()}


@torch.no_grad()
def neighbor_diagnostics(query_f: Tensor, query_labels: Tensor, query_valid: Tensor,
                         memory_ids: Tensor, memory_f: Tensor, memory_labels: Tensor,
                         memory_valid: Tensor, memory_images: Sequence[str], memory_events: Sequence[str],
                         local_ids: Tensor, local_labels_pred: Tensor, local_valid: Tensor,
                         *, chunk_size: int = 32) -> dict:
    """Return scalar region rows plus a bank inventory and definitions.

    ``local_gt_oracle`` and ``local_other_gt_oracle`` use query GT only for an
    explicitly posthoc oracle. They never change selected IDs or real votes.
    The other-only local references remove self *without refilling*: a probe of
    dependence on self evidence, not a new equal-budget trained comparator.
    """
    if query_f.ndim != 3 or query_f.shape[1] != 2 or query_f.shape[0] < 1 or query_f.shape[2] < 1:
        raise ValueError("query_f must be nonempty [R,2,C]")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("positive integer chunk_size required")
    r, _, c = query_f.shape
    tensors = (query_labels, query_valid, memory_ids, memory_f, memory_labels, memory_valid, local_ids, local_labels_pred, local_valid)
    if any(t.device != query_f.device for t in tensors):
        raise ValueError("all tensors must share a device")
    _validate_features("query", query_f, query_valid, c)
    _validate_features("memory", memory_f, memory_valid, c)
    if local_valid.shape != query_valid.shape or local_valid.dtype != torch.bool:
        raise ValueError("local_valid must be bool [R,2]")
    if bool((local_valid & ~query_valid).any()):
        raise ValueError("local_valid cannot include absent query slots")
    _validate_labels("query GT", query_labels, query_valid, allow_zero=True)
    _validate_labels("memory", memory_labels, memory_valid)
    _validate_labels("local prediction", local_labels_pred, local_valid)
    _validate_ids("memory", memory_ids, r, memory_f.shape[0])
    _validate_ids("local", local_ids, r, r)
    if memory_ids.shape[1] != local_ids.shape[1]:
        raise ValueError("memory/local parent budgets must match")
    if len(memory_images) != memory_f.shape[0] or len(memory_events) != memory_f.shape[0]:
        raise ValueError("memory image/event metadata mismatch")
    rows: dict[str, Tensor] = {}
    specs = [
        ("memory", memory_ids, memory_f, memory_labels, memory_valid, dict(images=memory_images, events=memory_events)),
        ("local_pred", local_ids, query_f, local_labels_pred, local_valid, {}),
        ("local_gt_oracle", local_ids, query_f, query_labels, local_valid, dict(allow_zero=True)),
    ]
    other_ids = local_ids.clone()
    self_mask = local_ids == torch.arange(r, device=query_f.device)[:, None]
    other_ids[self_mask] = -1
    specs += [
        ("local_other_pred", other_ids, query_f, local_labels_pred, local_valid, {}),
        ("local_other_gt_oracle", other_ids, query_f, query_labels, local_valid, dict(allow_zero=True)),
    ]
    for prefix, ids, features, labels, valid, options in specs:
        measured = _table_rows(query_f, query_labels, query_valid, ids, features, labels, valid, chunk_size=chunk_size, **options)
        rows.update({prefix + "_" + key: value for key, value in measured.items()})
    rows["local_self_parent_occurrences"] = self_mask.sum(-1).float()
    for cls in CLASSES:
        rows["memory_minus_local_pred_vote_" + cls] = rows["memory_cosine_vote_" + cls] - rows["local_pred_cosine_vote_" + cls]
    rows["memory_minus_local_pred_posthoc_tv"] = rows["memory_posthoc_vote_tv_compatibility"] - rows["local_pred_posthoc_vote_tv_compatibility"]
    return {
        "region_rows": rows,
        "inventory": inventory(memory_labels, memory_valid, memory_images, memory_events),
        "definitions": {
            "query_role": 1,
            "support_roles": [0, 1],
            "parent_budget": memory_ids.shape[1],
            "temperature": .1,
            "vote": "fixed raw-feature cosine-weighted diagnostic; NOT trained Reader attention or pixel IoU",
            "posthoc": "query GT affects only compatibility statistics and explicitly named local GT oracle",
            "local_gt_oracle": "oracle reference-label quality only; GT is never a model input; may contain self",
            "local_other": "self excluded without refill; diagnostic, not an equal-budget trained comparator",
            "missing": "undefined evidence is NaN; counts are zero; report finite denominator separately",
            "similar_thresholds": "cosine > 0.8 and candidate class mass >= 0.5; descriptive exposure, not a mining rule",
            "duplicate_image_fraction": "1 - unique source images / unique valid parent cases",
            "class_values": "unweighted child-slot distribution unless explicitly called cosine_vote; not area-weighted",
            "foreground_conditional": "background mass removed for this statistic only; always interpret together with foreground mass and valid slot counts",
        },
    }
