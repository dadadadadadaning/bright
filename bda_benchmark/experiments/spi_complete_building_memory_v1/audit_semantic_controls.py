from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .audit_suitability import _events, _validate, load_cards
from .core import (
    balanced_memory_indices, binary_auroc, composite_building_key,
    cross_event_topk, macro_f1, standardize,
)


def _role_key(cache: dict) -> torch.Tensor:
    role = cache["role_tokens"].float()
    valid = cache["role_token_mask"].float()[..., None]
    role = (role * valid).sum(2) / valid.sum(2).clamp_min(1)
    return F.normalize(role.flatten(1), dim=1, eps=1e-8)


def _key_pair(train: dict, query: dict, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "complete":
        train_key, statistics = composite_building_key(train)
        query_key, _ = composite_building_key(query, statistics)
        return train_key, query_key
    if name == "geometry":
        raw_train = torch.cat((train["geometry"].float(), train["role_fractions"].float()), 1)
        raw_query = torch.cat((query["geometry"].float(), query["role_fractions"].float()), 1)
        train_key, mean, std = standardize(raw_train)
        query_key, _, _ = standardize(raw_query, mean, std)
    elif name == "p3_global":
        fields = ("mean", "hard_critical", "global_exist")
        train_key = torch.cat([F.normalize(train["key_variants"][field].float(), dim=1) for field in fields], 1)
        query_key = torch.cat([F.normalize(query["key_variants"][field].float(), dim=1) for field in fields], 1)
    elif name in ("rgb_only", "sar_only"):
        modality = name.split("_", 1)[0]
        train_key = train["modality_keys"][modality].float()
        query_key = query["modality_keys"][modality].float()
    elif name == "rgb_sar_roles":
        train_rgb = F.normalize(train["modality_keys"]["rgb"].float(), dim=1)
        train_sar = F.normalize(train["modality_keys"]["sar"].float(), dim=1)
        query_rgb = F.normalize(query["modality_keys"]["rgb"].float(), dim=1)
        query_sar = F.normalize(query["modality_keys"]["sar"].float(), dim=1)
        train_key = torch.cat((train_rgb, train_sar, F.normalize(torch.abs(train_rgb - train_sar), dim=1), _role_key(train)), 1)
        query_key = torch.cat((query_rgb, query_sar, F.normalize(torch.abs(query_rgb - query_sar), dim=1), _role_key(query)), 1)
    else:
        raise ValueError(f"unknown key view: {name}")
    return F.normalize(train_key, dim=1, eps=1e-8), F.normalize(query_key, dim=1, eps=1e-8)


def _score(
    train: dict, query: dict, train_key: torch.Tensor, query_key: torch.Tensor,
    memory_rows: torch.Tensor, args, *, shuffled: bool = False,
) -> dict:
    train_events = _events(train); query_events = _events(query)
    retrieval = cross_event_topk(
        query_key, train_key[memory_rows], query_events,
        [train_events[index] for index in memory_rows.tolist()],
        topk=args.topk, maximum_per_event=args.maximum_per_event, device=args.device,
    )
    memory_labels = train["labels"].long()[memory_rows].clone()
    if shuffled:
        generator = torch.Generator().manual_seed(args.seed + 991)
        memory_labels = memory_labels[torch.randperm(len(memory_labels), generator=generator)]
    labels = memory_labels[retrieval.indices.clamp_min(0)]
    weights = torch.softmax(retrieval.scores / args.temperature, 1)
    valid_neighbour = torch.isfinite(retrieval.scores)
    weights *= valid_neighbour
    posterior = torch.zeros(len(query_key), 4)
    posterior.scatter_add_(1, labels, weights)
    posterior /= posterior.sum(1, keepdim=True).clamp_min(1e-8)
    posterior[~retrieval.coverage] = .25
    target = query["labels"].long(); p3 = query["p3_probabilities"].float()
    prediction = ((1.0 - args.semantic_blend) * p3 + args.semantic_blend * posterior).argmax(1)
    auc_by_class = {str(label): binary_auroc(target == label, posterior[:, label]) for label in range(4)}
    valid_auc = [value for value in auc_by_class.values() if value == value]
    event_gain = {}
    for event in sorted(set(query_events)):
        mask = torch.tensor([value == event for value in query_events])
        event_gain[event] = macro_f1(target[mask], prediction[mask]) - macro_f1(target[mask], p3[mask].argmax(1))
    purity = ((labels == target[:, None]) & valid_neighbour).sum(1).float() / valid_neighbour.sum(1).clamp_min(1)
    covered = retrieval.coverage
    return {
        "macro_ovr_auroc": sum(valid_auc) / max(len(valid_auc), 1),
        "affected_ovr_auroc": (auc_by_class["2"] + auc_by_class["3"]) / 2,
        "auroc_by_class": auc_by_class,
        "mean_neighbour_purity": float(purity[covered].mean()) if covered.any() else 0.0,
        "coverage": float(covered.float().mean()),
        "macro_f1": macro_f1(target, prediction),
        "macro_f1_gain_vs_p3": macro_f1(target, prediction) - macro_f1(target, p3.argmax(1)),
        "nonnegative_events": sum(value >= 0 for value in event_gain.values()),
        "event_total": len(event_gain), "events": event_gain,
    }


def _prior_control(train: dict, query: dict, memory_rows: torch.Tensor, args) -> dict:
    labels = train["labels"].long()[memory_rows]
    prior = torch.bincount(labels, minlength=4).float(); prior /= prior.sum().clamp_min(1)
    posterior = prior[None].repeat(len(query["labels"]), 1)
    target = query["labels"].long(); p3 = query["p3_probabilities"].float()
    prediction = ((1.0 - args.semantic_blend) * p3 + args.semantic_blend * posterior).argmax(1)
    auc_by_class = {str(label): binary_auroc(target == label, posterior[:, label]) for label in range(4)}
    events = _events(query); gains = {}
    for event in sorted(set(events)):
        mask = torch.tensor([value == event for value in events])
        gains[event] = macro_f1(target[mask], prediction[mask]) - macro_f1(target[mask], p3[mask].argmax(1))
    return {
        "macro_ovr_auroc": .5, "affected_ovr_auroc": .5, "auroc_by_class": auc_by_class,
        "mean_neighbour_purity": float("nan"), "coverage": 1.0,
        "macro_f1": macro_f1(target, prediction),
        "macro_f1_gain_vs_p3": macro_f1(target, prediction) - macro_f1(target, p3.argmax(1)),
        "nonnegative_events": sum(value >= 0 for value in gains.values()),
        "event_total": len(gains), "events": gains, "class_prior": prior.tolist(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Complete-building semantic Memory shortcut controls")
    parser.add_argument("--train-cards", required=True); parser.add_argument("--calibration-cards", required=True)
    parser.add_argument("--suitability-json", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--maximum-per-event-class", type=int, default=256)
    parser.add_argument("--maximum-per-event", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.10); parser.add_argument("--semantic-blend", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    if args.topk != 8 or args.maximum_per_event != 2 or args.semantic_blend != .25:
        raise RuntimeError("control protocol freezes topk=8, max/event=2 and beta=0.25")
    prerequisite = json.load(open(args.suitability_json, encoding="utf-8"))
    if prerequisite.get("decision") != "MEMORY_SUITABLE_FOR_P3_SEMANTIC_CLASSIFICATION_ONLY":
        raise RuntimeError("semantic Memory E0 prerequisite did not pass")
    train = load_cards(args.train_cards); query = load_cards(args.calibration_cards)
    _validate(train, "meta_train"); _validate(query, "meta_calibration")
    memory_rows = balanced_memory_indices(
        train["labels"].long(), _events(train),
        maximum_per_event_class=args.maximum_per_event_class, seed=args.seed,
    )
    results = {}
    for variant, view in (
        ("c0_normal_complete", "complete"), ("c1_shuffled_labels", "complete"),
        ("c3_geometry_only", "geometry"), ("c4_p3_global_only", "p3_global"),
        ("c5_rgb_only", "rgb_only"), ("c6_sar_only", "sar_only"),
        ("c7_rgb_sar_roles", "rgb_sar_roles"),
    ):
        train_key, query_key = _key_pair(train, query, view)
        results[variant] = _score(
            train, query, train_key, query_key, memory_rows, args,
            shuffled=variant == "c1_shuffled_labels",
        )
        row = results[variant]
        print(f"{variant:28s} AUC={row['macro_ovr_auroc']:.4f} F1gain={100*row['macro_f1_gain_vs_p3']:+.3f}pp purity={row['mean_neighbour_purity']:.4f}", flush=True)
    results["c2_class_prior"] = _prior_control(train, query, memory_rows, args)
    full = results["c0_normal_complete"]
    checks = {
        "shuffled_macro_auroc_le_0p55": results["c1_shuffled_labels"]["macro_ovr_auroc"] <= .55,
        "complete_beats_geometry_auc_by_0p02": full["macro_ovr_auroc"] >= results["c3_geometry_only"]["macro_ovr_auroc"] + .02,
        "complete_beats_p3_global_auc_by_0p01": full["macro_ovr_auroc"] >= results["c4_p3_global_only"]["macro_ovr_auroc"] + .01,
        "complete_not_below_best_single_modality": full["macro_ovr_auroc"] >= max(results["c5_rgb_only"]["macro_ovr_auroc"], results["c6_sar_only"]["macro_ovr_auroc"]),
        "complete_cross_event_nonnegative_events_ge_10": full["nonnegative_events"] >= 10,
    }
    decision = "ADMIT_COMPLETE_BUILDING_MEMORY_FOR_PIXEL_WRITEBACK_E1" if all(checks.values()) else "STOP_MEMORY_SHORTCUT_CONTROLS_FAILED"
    report = {
        "protocol": "SPI_COMPLETE_BUILDING_MEMORY_SEMANTIC_CONTROLS_C0_C7_v1",
        "configuration": vars(args), "results": results, "checks": checks, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"Checks: {checks}"); print(f"Decision: {decision}"); print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
