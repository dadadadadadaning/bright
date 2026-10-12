from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from . import AUDIT_PROTOCOL, PROTOCOL
from .core import (
    balanced_memory_indices, binary_auroc, composite_building_key,
    cross_event_topk, macro_f1, stable_string_hash, standardize,
)


LOCAL_GROUPS = ("spatial", "stage2_role", "rgb_edge", "sar_texture", "structure_relation")


def load_cards(path: str) -> dict:
    """Load either the original packed cache or the memory-safe sharded cache."""
    manifest = torch.load(path, map_location="cpu", weights_only=False)
    if not manifest.get("sharded", False):
        return manifest
    shards = manifest.get("shards", [])
    if not shards:
        raise RuntimeError("sharded card manifest contains no shards")
    tensor_fields = (
        "image_index", "region_label", "areas", "bboxes", "geometry", "labels",
        "label_purity", "class_fractions", "p3_predictions", "p3_probabilities",
        "role_tokens", "role_token_mask", "role_fractions",
    )
    columns = {name: [] for name in tensor_fields}; key_columns = {}; modality_columns = {}
    image_names, image_events, image_offset = [], [], 0
    for shard_path in shards:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for name in tensor_fields:
            value = shard[name]
            if name == "image_index": value = value.long() + image_offset
            columns[name].append(value)
        for name, value in shard["key_variants"].items(): key_columns.setdefault(name, []).append(value)
        for name, value in shard["modality_keys"].items(): modality_columns.setdefault(name, []).append(value)
        image_names.extend(shard["image_names"]); image_events.extend(shard["image_events"])
        image_offset += len(shard["image_names"])
    return {
        **{k: v for k, v in manifest.items() if k not in ("shards", "sharded")},
        **{name: torch.cat(values) for name, values in columns.items()},
        "key_variants": {name: torch.cat(values) for name, values in key_columns.items()},
        "modality_keys": {name: torch.cat(values) for name, values in modality_columns.items()},
        "image_names": image_names, "image_events": image_events,
    }


def _events(cache: dict) -> list[str]:
    return [cache["image_events"][int(i)] for i in cache["image_index"].long().tolist()]


def semantic_memory_posterior(train: dict, query: dict, args) -> tuple[torch.Tensor, dict]:
    train_key, statistics = composite_building_key(train)
    query_key, _ = composite_building_key(query, statistics)
    memory_rows = balanced_memory_indices(
        train["labels"].long(), _events(train),
        maximum_per_event_class=args.maximum_per_event_class, seed=args.seed,
    )
    retrieval = cross_event_topk(
        query_key, train_key[memory_rows], _events(query),
        [_events(train)[i] for i in memory_rows.tolist()],
        topk=args.topk, maximum_per_event=args.maximum_per_event, device=args.device,
    )
    labels = train["labels"].long()[memory_rows][retrieval.indices.clamp_min(0)]
    weights = torch.softmax(retrieval.scores / args.temperature, 1)
    weights *= torch.isfinite(retrieval.scores)
    posterior = torch.zeros(len(query_key), 4)
    posterior.scatter_add_(1, labels, weights)
    posterior /= posterior.sum(1, keepdim=True).clamp_min(1e-8)
    posterior[~retrieval.coverage] = .25
    neighbour_valid = torch.isfinite(retrieval.scores)
    purity = ((labels == query["labels"].long()[:, None]) & neighbour_valid).sum(1).float() / neighbour_valid.sum(1).clamp_min(1)
    diagnostics = {
        "memory_rows": int(memory_rows.numel()),
        "coverage": retrieval.coverage,
        "neighbour_purity": purity,
        "indices": retrieval.indices,
        "scores": retrieval.scores,
    }
    return posterior, diagnostics


def _semantic_metrics(train: dict, query: dict, args) -> dict:
    posterior, retrieval_diagnostics = semantic_memory_posterior(train, query, args)
    target = query["labels"].long()
    p3 = query["p3_probabilities"].float()
    blend = (1.0 - args.semantic_blend) * p3 + args.semantic_blend * posterior
    p3_prediction = p3.argmax(1); blend_prediction = blend.argmax(1)
    auc_by_class = {str(label): binary_auroc(target == label, posterior[:, label]) for label in range(4)}
    valid_auc = [v for v in auc_by_class.values() if v == v]
    event_names = _events(query); event_gain = {}
    for event in sorted(set(event_names)):
        mask = torch.tensor([item == event for item in event_names])
        event_gain[event] = macro_f1(target[mask], blend_prediction[mask]) - macro_f1(target[mask], p3_prediction[mask])
    coverage = retrieval_diagnostics["coverage"]
    purity = retrieval_diagnostics["neighbour_purity"]
    return {
        "memory_rows": retrieval_diagnostics["memory_rows"], "queries": len(target),
        "coverage": float(coverage.float().mean()),
        "mean_neighbour_purity": float(purity[coverage].mean()) if coverage.any() else 0.0,
        "macro_ovr_auroc": sum(valid_auc) / max(len(valid_auc), 1),
        "affected_ovr_auroc": (auc_by_class["2"] + auc_by_class["3"]) / 2,
        "auroc_by_class": auc_by_class,
        "p3_macro_f1": macro_f1(target, p3_prediction),
        "memory_blend_macro_f1": macro_f1(target, blend_prediction),
        "macro_f1_gain": macro_f1(target, blend_prediction) - macro_f1(target, p3_prediction),
        "nonnegative_events": sum(value >= 0 for value in event_gain.values()),
        "event_total": len(event_gain), "events": event_gain,
    }


def _child_matrix(cache: dict) -> torch.Tensor:
    missing = [name for name in LOCAL_GROUPS if name not in cache["features"]]
    if missing:
        raise RuntimeError(f"D1 cache misses groups: {missing}")
    return torch.cat([cache["features"][name].float() for name in LOCAL_GROUPS], 1)


def _card_lookup(card: dict) -> dict[tuple[str, int], int]:
    names = [card["image_names"][int(i)] for i in card["image_index"].long().tolist()]
    return {(name, int(region)): row for row, (name, region) in enumerate(zip(names, card["region_label"]))}


def _prepare_children(d1: dict, card: dict, mean=None, std=None) -> tuple[dict, torch.Tensor, torch.Tensor]:
    lookup = _card_lookup(card); image_names = d1["image_names"]
    parent, kept = [], []
    net = d1["utility"]["net"].float(); changed = d1["utility"]["changed"].float()
    eligible = (changed > 0) & (net != 0)
    for row in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
        key = (image_names[int(d1["image_index"][row])], int(d1["region_label"][row]))
        if key in lookup:
            kept.append(row); parent.append(lookup[key])
    if not kept:
        raise RuntimeError("no D1 children joined complete building cards")
    kept_t = torch.tensor(kept, dtype=torch.long)
    local, mean, std = standardize(_child_matrix(d1)[kept_t], mean, std)
    event_names = [card["image_events"][int(card["image_index"][p])] for p in parent]
    return {
        "key": F.normalize(local, dim=1, eps=1e-8), "parent": torch.tensor(parent),
        "event_names": event_names, "event": stable_string_hash(event_names),
        "direction": d1["direction_code"].long()[kept_t],
        "label": (net[kept_t] > 0).long(),
        "quality": (net[kept_t].abs() / changed[kept_t].clamp_min(1)).clamp(0, 1),
    }, mean, std


@torch.no_grad()
def _utility_retrieve(train_child, query_child, train_building_key, query_building_key, args) -> dict:
    probabilities, purity, coverage, neighbour_events = [], [], [], []
    train_event = train_child["event"]
    for row in range(len(query_child["key"])):
        parent = int(query_child["parent"][row]); query_event = query_child["event"][row]
        coarse = query_building_key[parent] @ train_building_key.T
        building_events = torch.full((len(train_building_key),), -1, dtype=torch.long)
        building_events[train_child["parent"]] = train_event
        coarse[(building_events < 0) | (building_events == query_event)] = float("-inf")
        count = min(args.coarse_buildings, int(torch.isfinite(coarse).sum()))
        routed = coarse.topk(count).indices if count else torch.empty(0, dtype=torch.long)
        candidate = torch.isin(train_child["parent"], routed) & (train_event != query_event)
        ids = torch.nonzero(candidate, as_tuple=False).flatten()
        if not ids.numel():
            probabilities.append(.5); purity.append(0.); coverage.append(False); neighbour_events.append(0); continue
        local_score = train_child["key"][ids] @ query_child["key"][row]
        local_score += args.direction_bonus * (train_child["direction"][ids] == query_child["direction"][row]).float()
        local_score += args.parent_score_weight * coarse[train_child["parent"][ids]]
        order = local_score.argsort(descending=True, stable=True)
        chosen, per_event = [], {}
        for position in order.tolist():
            item = int(ids[position]); event = int(train_event[item])
            if per_event.get(event, 0) >= args.maximum_per_event:
                continue
            chosen.append(item); per_event[event] = per_event.get(event, 0) + 1
            if len(chosen) == args.topk:
                break
        chosen_t = torch.tensor(chosen, dtype=torch.long)
        scores = train_child["key"][chosen_t] @ query_child["key"][row]
        weights = torch.softmax(scores / args.temperature, 0) * train_child["quality"][chosen_t]
        probability = (weights * train_child["label"][chosen_t]).sum() / weights.sum().clamp_min(1e-8)
        probabilities.append(float(probability)); coverage.append(True)
        purity.append(float((train_child["label"][chosen_t] == query_child["label"][row]).float().mean()))
        neighbour_events.append(len(per_event))
    score = torch.tensor(probabilities); valid = torch.tensor(coverage)
    event_auc = {}
    for event in sorted(set(query_child["event_names"])):
        mask = torch.tensor([name == event for name in query_child["event_names"]])
        value = binary_auroc(query_child["label"][mask], score[mask])
        if value == value:
            event_auc[event] = value
    return {
        "utility_auroc": binary_auroc(query_child["label"], score),
        "event_macro_auroc": sum(event_auc.values()) / max(len(event_auc), 1),
        "correct_events": sum(value > .5 for value in event_auc.values()),
        "evaluable_events": len(event_auc), "coverage": float(valid.float().mean()),
        "mean_neighbour_purity": float(torch.tensor(purity)[valid].mean()) if valid.any() else 0.0,
        "mean_neighbour_events": float(torch.tensor(neighbour_events)[valid].float().mean()) if valid.any() else 0.0,
        "events": event_auc,
    }


def _validate(card: dict, split: str) -> None:
    if card.get("protocol") != PROTOCOL or card.get("split") != split:
        raise RuntimeError(f"expected {PROTOCOL} {split}")
    if card.get("entry_filter_uses_gt", True):
        raise RuntimeError("card entry eligibility must be label-free")
    if any(card.get(name, True) for name in (
        "meta_confirm_accessed", "val450_accessed",
        "official_instances_accessed", "test_accessed",
    )):
        raise RuntimeError("development audit refuses confirm/Val/test data")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Decide which task complete-building Memory can serve")
    parser.add_argument("--train-cards", required=True); parser.add_argument("--calibration-cards", required=True)
    parser.add_argument("--train-d1", required=True); parser.add_argument("--calibration-d1", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=8); parser.add_argument("--coarse-buildings", type=int, default=32)
    parser.add_argument("--maximum-per-event-class", type=int, default=256)
    parser.add_argument("--maximum-per-event", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.10); parser.add_argument("--semantic-blend", type=float, default=.25)
    parser.add_argument("--direction-bonus", type=float, default=.10); parser.add_argument("--parent-score-weight", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    train = load_cards(args.train_cards)
    calibration = load_cards(args.calibration_cards)
    _validate(train, "meta_train"); _validate(calibration, "meta_calibration")
    semantic = _semantic_metrics(train, calibration, args)
    train_key, stats = composite_building_key(train); calibration_key, _ = composite_building_key(calibration, stats)
    train_d1 = torch.load(args.train_d1, map_location="cpu", weights_only=False)
    calibration_d1 = torch.load(args.calibration_d1, map_location="cpu", weights_only=False)
    if train_d1.get("split") != "meta_train" or calibration_d1.get("split") != "meta_calibration":
        raise RuntimeError("utility audit requires meta_train/meta_calibration D1 caches")
    forbidden = ("meta_confirm_accessed", "val450_accessed", "test_accessed")
    if any(train_d1.get(name, False) or calibration_d1.get(name, False) for name in forbidden):
        raise RuntimeError("utility audit refuses confirm/Val/test D1 caches")
    train_child, mean, std = _prepare_children(train_d1, train)
    query_child, _, _ = _prepare_children(calibration_d1, calibration, mean, std)
    utility = _utility_retrieve(train_child, query_child, train_key, calibration_key, args)
    semantic_checks = {
        "macro_ovr_auroc_ge_0p65": semantic["macro_ovr_auroc"] >= .65,
        "affected_ovr_auroc_ge_0p65": semantic["affected_ovr_auroc"] >= .65,
        "neighbour_purity_ge_0p60": semantic["mean_neighbour_purity"] >= .60,
        "macro_f1_gain_ge_0p01": semantic["macro_f1_gain"] >= .01,
        "nonnegative_events_ge_10": semantic["nonnegative_events"] >= min(10, semantic["event_total"]),
        "coverage_ge_0p80": semantic["coverage"] >= .80,
    }
    utility_checks = {
        "utility_auroc_ge_0p60": utility["utility_auroc"] >= .60,
        "event_macro_auroc_ge_0p58": utility["event_macro_auroc"] >= .58,
        "correct_events_ge_10": utility["correct_events"] >= min(10, utility["evaluable_events"]),
        "neighbour_purity_ge_0p60": utility["mean_neighbour_purity"] >= .60,
        "coverage_ge_0p80": utility["coverage"] >= .80,
        "neighbour_events_ge_3": utility["mean_neighbour_events"] >= 3,
    }
    semantic_pass = all(semantic_checks.values()); utility_pass = all(utility_checks.values())
    if semantic_pass and utility_pass:
        decision = "MEMORY_SUITABLE_FOR_BOTH_USE_SEPARATE_BANKS"
    elif semantic_pass:
        decision = "MEMORY_SUITABLE_FOR_P3_SEMANTIC_CLASSIFICATION_ONLY"
    elif utility_pass:
        decision = "MEMORY_SUITABLE_FOR_S2_MODIFICATION_VERIFICATION_ONLY"
    else:
        decision = "STOP_MEMORY_NEITHER_TASK_HAS_CROSS_EVENT_EVIDENCE"
    report = {
        "protocol": AUDIT_PROTOCOL, "configuration": vars(args),
        "semantic_task": semantic, "semantic_checks": semantic_checks,
        "utility_task": utility, "utility_checks": utility_checks,
        "decision": decision,
        "interpretation": {
            "semantic": "whole building -> GT damage class; may support P3 logits",
            "utility": "parent building routed child modification -> keep or revert S2",
            "shared_decision_forbidden": True,
        },
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"semantic": semantic, "semantic_checks": semantic_checks,
                      "utility": utility, "utility_checks": utility_checks,
                      "decision": decision}, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
