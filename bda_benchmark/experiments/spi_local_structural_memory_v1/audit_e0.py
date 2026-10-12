from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import PROTOCOL
from .core import (affected_target, balanced_event_class_rows, binary_auroc, cross_event_topk,
                   event_auc, normalise_from_train, weighted_binary_vote)


ROW_TENSORS = (
    "image_index", "region_id", "labels", "label_purity", "areas", "geometry",
    "p3_predictions", "p3_probabilities", "rgb_key", "sar_key", "fused_key", "structure_key",
)


def _manifest(path: str) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("protocol") != PROTOCOL:
        raise RuntimeError(f"not an LSM cache: {path}")
    return value


def load_metadata(path: str) -> dict:
    """Read only fields needed to select a bounded Memory bank."""
    value = _manifest(path)
    if not value.get("sharded"):
        return value
    output = {key: value[key] for key in value if key not in ("shards",)}
    parts = [torch.load(item, map_location="cpu", weights_only=False) for item in value["shards"]]
    output["labels"] = torch.cat([part["labels"] for part in parts])
    output["label_purity"] = torch.cat([part["label_purity"] for part in parts])
    output["p3_predictions"] = torch.cat([part["p3_predictions"] for part in parts])
    output["row_events"] = [event for part in parts for event in part["row_events"]]
    return output


def load(path: str, rows: torch.Tensor | None = None) -> dict:
    """Load all rows, or only globally indexed rows from a sharded manifest."""
    value = _manifest(path)
    if not value.get("sharded"):
        if rows is None:
            return value
        output = dict(value)
        total = len(value["labels"])
        for key in ROW_TENSORS:
            output[key] = value[key][rows]
        output["row_events"] = [value["row_events"][int(i)] for i in rows.tolist()]
        return output
    wanted = None if rows is None else set(rows.tolist())
    pieces: dict[str, list] = {key: [] for key in ROW_TENSORS}
    events: list[str] = []
    offset = 0
    for item in value["shards"]:
        part = torch.load(item, map_location="cpu", weights_only=False)
        count = len(part["labels"])
        if wanted is None:
            local = torch.arange(count)
        else:
            local = torch.tensor(sorted(i - offset for i in wanted if offset <= i < offset + count), dtype=torch.long)
        if len(local):
            for key in ROW_TENSORS:
                pieces[key].append(part[key][local])
            events.extend(part["row_events"][int(i)] for i in local.tolist())
        offset += count
    if not pieces["labels"]:
        raise RuntimeError("requested zero rows from LSM cache")
    output = {key: value[key] for key in value if key not in ("shards",)}
    output.update({key: torch.cat(values) for key, values in pieces.items()})
    output["row_events"] = events
    return output


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def train_key(train: dict, query: dict, names: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    train_parts = [train[name].float() for name in names]
    query_parts = [query[name].float() for name in names]
    train_raw, query_raw = torch.cat(train_parts, 1), torch.cat(query_parts, 1)
    key, statistics = normalise_from_train(train_raw)
    query_key, _ = normalise_from_train(query_raw, statistics)
    return key, query_key


def recall_at_k(target: torch.Tensor, retrieval, memory_target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = retrieval.indices >= 0
    found = ((memory_target[retrieval.indices.clamp_min(0)] == target[:, None]) & valid).any(1)
    return float(found[mask & retrieval.coverage].float().mean()) if bool((mask & retrieval.coverage).any()) else 0.


def report(name: str, retrieval, query: dict, memory: dict, *, shuffled_values: torch.Tensor | None = None) -> dict:
    target = affected_target(query["labels"]); memory_target = affected_target(memory["labels"])
    if shuffled_values is not None:
        memory_target = shuffled_values
    score = weighted_binary_vote(retrieval, memory_target)
    coverage = retrieval.coverage
    events = query["row_events"]
    gate_miss = (query["p3_predictions"].long() <= 1) & (target == 1)
    event = event_auc(target, score, events, coverage)
    pure = (((memory_target[retrieval.indices.clamp_min(0)] == target[:, None]) & (retrieval.indices >= 0)).sum(1).float()
            / (retrieval.indices >= 0).sum(1).clamp_min(1))
    return {
        "name": name, "affected_auroc": binary_auroc(target[coverage], score[coverage]),
        "event_macro_auroc": sum(event.values()) / max(len(event), 1), "correct_events": sum(v > .5 for v in event.values()),
        "coverage": float(coverage.float().mean()), "neighbour_purity": float(pure[coverage].mean()),
        "affected_recall_at_k": recall_at_k(target, retrieval, memory_target, target == 1),
        "gate_miss_recall_at_k": recall_at_k(target, retrieval, memory_target, gate_miss),
        "gate_miss_count": int(gate_miss.sum()), "events": event,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("LSM-E0: local structural retrieval audit")
    parser.add_argument("--train-cache", required=True); parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--maximum-per-event", type=int, default=2); parser.add_argument("--maximum-per-event-class", type=int, default=256)
    parser.add_argument("--minimum-purity", type=float, default=.60)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    train_meta = load_metadata(args.train_cache)
    # Cache construction never filters with GT.  This mask is used only to obtain
    # an interpretable evaluation target with one semantic label per region.
    train_rows = balanced_event_class_rows(train_meta["labels"], train_meta["row_events"],
                                           maximum_per_event_class=args.maximum_per_event_class, seed=args.seed,
                                           eligible=train_meta["label_purity"].float() >= args.minimum_purity)
    memory, query = load(args.train_cache, train_rows), load(args.calibration_cache)
    variants = {
        "e0_a_p3_local": (),
        "e0_b_rgb_structure": ("rgb_key", "structure_key"),
        "e0_c_sar_structure": ("sar_key", "structure_key"),
        "e0_d_local_rgb_sar": ("rgb_key", "sar_key", "fused_key", "structure_key"),
    }
    results, retrievals, query_keys, memory_keys = {}, {}, {}, {}
    target = affected_target(query["labels"]); p3_score = query["p3_probabilities"].float()[:, 2:].sum(1)
    p3_events = event_auc(target, p3_score, query["row_events"], torch.ones(len(target), dtype=torch.bool))
    results["e0_a_p3_local"] = {
        "affected_auroc": binary_auroc(target, p3_score), "event_macro_auroc": sum(p3_events.values()) / max(len(p3_events), 1),
        "correct_events": sum(v > .5 for v in p3_events.values()), "coverage": 1., "neighbour_purity": None,
        "affected_recall_at_k": None, "gate_miss_recall_at_k": None,
        "gate_miss_count": int(((query["p3_predictions"] <= 1) & (target == 1)).sum()), "events": p3_events,
    }
    for name, fields in variants.items():
        if not fields:
            continue
        memory_key, query_key = train_key(memory, query, fields)
        retrieval = cross_event_topk(query_key, memory_key, query["row_events"], memory["row_events"], topk=args.topk,
                                     maximum_per_event=args.maximum_per_event, device=args.device)
        retrievals[name] = retrieval; results[name] = report(name, retrieval, query, memory)
        query_keys[name], memory_keys[name] = query_key, memory_key
        print(json.dumps(results[name], ensure_ascii=False), flush=True)
    selected = "e0_d_local_rgb_sar"
    g = torch.Generator().manual_seed(args.seed + 81)
    shuffled = affected_target(memory["labels"])[torch.randperm(len(memory["labels"]), generator=g)]
    shuffled_report = report("e0_e_shuffled_labels", retrievals[selected], query, memory, shuffled_values=shuffled)
    permutation = torch.randperm(len(query["labels"]), generator=torch.Generator().manual_seed(args.seed + 82))
    random = cross_event_topk(query_keys[selected][permutation], memory_keys[selected],
                              query["row_events"], memory["row_events"], topk=args.topk,
                              maximum_per_event=args.maximum_per_event, device=args.device)
    random_report = report("e0_f_random_cross_event", random, query, memory)
    candidate = results[selected]
    checks = {
        "local_key_beats_p3_by_0p02_auc": candidate["affected_auroc"] >= results["e0_a_p3_local"]["affected_auroc"] + .02,
        "local_key_beats_shuffled_by_0p02_auc": candidate["affected_auroc"] >= shuffled_report["affected_auroc"] + .02,
        "local_key_beats_random_by_0p02_auc": candidate["affected_auroc"] >= random_report["affected_auroc"] + .02,
        "gate_miss_recall_at_8_ge_0p60": candidate["gate_miss_recall_at_k"] >= .60,
        "event_macro_auroc_ge_0p60": candidate["event_macro_auroc"] >= .60,
        "correct_events_ge_10": candidate["correct_events"] >= 10,
        "coverage_ge_0p80": candidate["coverage"] >= .80,
    }
    output = {
        "protocol": PROTOCOL, "stage": "LSM_E0_LOCAL_RETRIEVAL_AUDIT", "configuration": vars(args),
        "memory_rows": int(len(memory["labels"])), "query_rows": int(len(query["labels"])), "reports": {**results, "e0_e_shuffled_labels": shuffled_report, "e0_f_random_cross_event": random_report},
        "selected_variant": selected, "checks": checks,
        "decision": "ADMIT_LSM_E1_TRUE_NEIGHBOUR_DEPENDENCY" if all(checks.values()) else "STOP_LSM_AFTER_E0_NO_LOCAL_RETRIEVAL_SIGNAL",
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    save(Path(args.output), output); print(json.dumps({"decision": output["decision"], "checks": checks}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
