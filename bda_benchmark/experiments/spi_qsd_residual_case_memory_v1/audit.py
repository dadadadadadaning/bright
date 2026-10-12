from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_independent_building_memory_v1.audit import (
    compare_report, load_independent, subset,
)
from bda_benchmark.experiments.spi_independent_building_memory_v1.core import multiclass_report

from . import PROTOCOL
from .core import (FactorizedCaseMetric, bounded_observable_memory_rows,
                   correction_direction, memory_probability, neighbour_diagnostics,
                   prepare_groups, reliability_gates, retrieve, row_events)


def save(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def event_purity(retrieval, query_direction, memory_direction, query_events):
    safe = retrieval.indices.clamp_min(0); valid = retrieval.indices >= 0
    same = (memory_direction[safe] == query_direction[:, None]) & valid
    purity = same.sum(1).float() / valid.sum(1).clamp_min(1)
    hard = (query_direction // 4) != (query_direction % 4)
    output = {}
    for event in sorted(set(query_events)):
        mask = torch.tensor([name == event for name in query_events]) & hard & retrieval.coverage
        if mask.any(): output[event] = float(purity[mask].mean())
    return output


def main(argv=None):
    parser = argparse.ArgumentParser("QSD residual case Memory D0/E0/E1/E2 audit")
    parser.add_argument("--train-cards", required=True); parser.add_argument("--calibration-cards", required=True)
    parser.add_argument("--metric-checkpoint", required=True); parser.add_argument("--result-root", required=True)
    parser.add_argument("--conditional-result", default=""); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capacity-per-event-source", type=int, default=256)
    parser.add_argument("--coarse-k", type=int, default=64); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--maximum-per-event", type=int, default=2); parser.add_argument("--quality-limit", type=float, default=3.0)
    parser.add_argument("--area-ratio-limit", type=float, default=2.0); parser.add_argument("--alpha", type=float, default=.10)
    parser.add_argument("--minimum-similarity", type=float, default=.20)
    parser.add_argument("--minimum-consensus", type=float, default=.75)
    parser.add_argument("--minimum-events", type=int, default=3); parser.add_argument("--minimum-ess", type=float, default=3.0)
    parser.add_argument("--maximum-quality-distance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    result_root = Path(args.result_root); result_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train = load_independent(args.train_cards); query = load_independent(args.calibration_cards)
    train_events_all, query_events = row_events(train), row_events(query)
    rows = bounded_observable_memory_rows(train, args.capacity_per_event_source, args.seed)
    memory = subset(train, rows); memory_events = [train_events_all[i] for i in rows.tolist()]
    groups = prepare_groups(train, query)
    ts, qs = groups["train_structure"][rows], groups["query_structure"]
    td, qd = groups["train_damage"][rows], groups["query_damage"]
    tq, qq = groups["train_quality"][rows], groups["query_quality"]
    query_source = query["p3_predictions"].long(); memory_source = memory["p3_predictions"].long()
    query_area = query["areas"].float(); memory_area = memory["areas"].float()
    query_probability = query["p3_probabilities"].float(); memory_probability_base = memory["p3_probabilities"].float()
    query_margin = query_probability.topk(2, 1).values.diff(1).abs().squeeze(1)
    memory_margin = memory_probability_base.topk(2, 1).values.diff(1).abs().squeeze(1)
    query_direction = correction_direction(query); memory_direction = correction_direction(memory)

    checkpoint = torch.load(args.metric_checkpoint, map_location="cpu", weights_only=False)
    model = FactorizedCaseMetric(checkpoint["structure_dim"], checkpoint["damage_dim"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["model_state_dict"]); model.to(device).eval()
    with torch.no_grad():
        ms_parts, md_parts, qs_parts, qd_parts = [], [], [], []
        for start in range(0, len(ts), 2048):
            a, b = model.encode(ts[start:start+2048].to(device), td[start:start+2048].to(device))
            ms_parts.append(a.cpu()); md_parts.append(b.cpu())
        for start in range(0, len(qs), 2048):
            a, b = model.encode(qs[start:start+2048].to(device), qd[start:start+2048].to(device))
            qs_parts.append(a.cpu()); qd_parts.append(b.cpu())
        learned_ms, learned_md = torch.cat(ms_parts), torch.cat(md_parts)
        learned_qs, learned_qd = torch.cat(qs_parts), torch.cat(qd_parts)

    common = dict(
        query_quality=qq, memory_quality=tq, query_events=query_events, memory_events=memory_events,
        query_source=query_source, memory_source=memory_source, query_area=query_area, memory_area=memory_area,
        query_margin=query_margin, memory_margin=memory_margin, coarse_k=args.coarse_k, topk=args.topk,
        maximum_per_event=args.maximum_per_event, quality_limit=args.quality_limit,
        area_ratio_limit=args.area_ratio_limit, seed=args.seed,
        similarity_device=str(device),
    )
    specifications = {
        "r0_random_cross_event": ("random", qs, ts, qd, td),
        "r1_flat_concat": ("flat", qs, ts, qd, td),
        "r2_structure_only": ("structure", qs, ts, qd, td),
        "r3_damage_only": ("damage", qs, ts, qd, td),
        "r4_structure_then_damage": ("hierarchical", qs, ts, qd, td),
        "r5_quality_compatible": ("quality", qs, ts, qd, td),
        "r6_observable_route": ("route", qs, ts, qd, td),
        "r7_learned_factorized_route": ("learned", learned_qs, learned_ms, learned_qd, learned_md),
    }
    retrievals, retrieval_reports = {}, {}
    for name, (mode, q_structure, m_structure, q_damage, m_damage) in specifications.items():
        print(f"QSD retrieval {name}", flush=True)
        found = retrieve(q_structure, m_structure, q_damage, m_damage, **common, mode=mode)
        retrievals[name] = found
        row = neighbour_diagnostics(found, query_direction, memory_direction, memory_events)
        row["events"] = event_purity(found, query_direction, memory_direction, query_events)
        retrieval_reports[name] = row
        print(json.dumps({"variant": name, **{k: v for k, v in row.items() if k != "events"}}), flush=True)
    selected_name = "r7_learned_factorized_route"
    selected = retrievals[selected_name]
    random_events = retrieval_reports["r0_random_cross_event"]["events"]
    selected_events = retrieval_reports[selected_name]["events"]
    comparable_events = sorted(set(random_events) & set(selected_events))
    better_events = sum(selected_events[e] > random_events[e] for e in comparable_events)
    d0_checks = {
        "p3_error_direction_purity_ge_0p65": retrieval_reports[selected_name]["direction_purity_p3_errors"] >= .65,
        "coverage_ge_0p80": retrieval_reports[selected_name]["coverage"] >= .80,
        "mean_neighbour_events_ge_3": retrieval_reports[selected_name]["mean_neighbour_events"] >= 3,
        "better_than_random_in_ge_10_events": better_events >= 10,
    }
    d0_payload = {
        "protocol": PROTOCOL, "memory_rows": int(len(rows)), "variants": retrieval_reports,
        "selected": selected_name, "better_than_random_events": f"{better_events}/{len(comparable_events)}",
        "checks": d0_checks, "decision": "ADMIT_QSD_VALUE_AUDIT" if all(d0_checks.values()) else "STOP_QSD_NO_RETRIEVABLE_RESIDUAL_CASES",
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    save(result_root / f"qsd_d0_e0_retrieval_seed{args.seed}.json", d0_payload)

    # E1: identical neighbours, only the Memory value changes.
    target = query["labels"].long(); p3 = query_probability
    permutation = torch.randperm(len(memory["labels"]), generator=torch.Generator().manual_seed(args.seed + 41))
    values = {
        "v0_hard_label": memory_probability(p3, selected, memory, "hard_label", args.alpha),
        "v1_soft_label": memory_probability(p3, selected, memory, "soft_label", args.alpha),
        "v2_p3_residual": memory_probability(p3, selected, memory, "residual", args.alpha),
        "v3_shuffled_residual": memory_probability(p3, selected, memory, "residual", args.alpha, permutation=permutation),
        "v4_zero": p3,
    }
    value_reports = {name: compare_report(target, p3, probability, query_events) for name, probability in values.items()}

    # E2: identical residual Memory, progressively stronger query-time abstention.
    gates, gate_diagnostics = reliability_gates(
        selected, memory_direction, memory_events, minimum_similarity=args.minimum_similarity,
        minimum_consensus=args.minimum_consensus, minimum_events=args.minimum_events,
        minimum_ess=args.minimum_ess, maximum_quality_distance=args.maximum_quality_distance,
    )
    gate_reports = {}
    for name, gate in gates.items():
        probability = memory_probability(p3, selected, memory, "residual", args.alpha, gate=gate)
        gate_reports[name] = {**compare_report(target, p3, probability, query_events),
                              "use_fraction": float(gate.float().mean()), "used_queries": int(gate.sum())}

    p3_report = multiclass_report(target, p3, query_events)
    selected_report = gate_reports["g4_full_reliability"]
    p3_only_auc = None
    if args.conditional_result and Path(args.conditional_result).is_file():
        conditional = json.loads(Path(args.conditional_result).read_text(encoding="utf-8"))
        p3_only_auc = conditional["reports"]["b1_p3_only_calibrator"]["macro_ovr_auroc"]
    final_checks = {
        "residual_beats_hard_label_auc": value_reports["v2_p3_residual"]["macro_ovr_auroc"] > value_reports["v0_hard_label"]["macro_ovr_auroc"],
        "residual_beats_shuffled_by_0p01_auc": value_reports["v2_p3_residual"]["macro_ovr_auroc"] >= value_reports["v3_shuffled_residual"]["macro_ovr_auroc"] + .01,
        "full_gate_beats_raw_p3_by_0p01_auc": selected_report["macro_ovr_auroc"] >= p3_report["macro_ovr_auroc"] + .01,
        "full_gate_beats_p3_only_by_0p01_auc": p3_only_auc is not None and selected_report["macro_ovr_auroc"] >= p3_only_auc + .01,
        "full_gate_nonnegative_events_ge_10": selected_report["nonnegative_events"] >= 10,
        "memory_use_fraction_between_0p05_and_0p40": .05 <= selected_report["use_fraction"] <= .40,
    }
    e1_payload = {
        "protocol": PROTOCOL, "p3": p3_report, "p3_only_calibrator_auc": p3_only_auc,
        "value_reports": value_reports, "gate_reports": gate_reports, "gate_diagnostics": gate_diagnostics,
        "configuration": {"alpha": args.alpha, "minimum_similarity": args.minimum_similarity,
                          "minimum_consensus": args.minimum_consensus, "minimum_events": args.minimum_events,
                          "minimum_ess": args.minimum_ess, "maximum_quality_distance": args.maximum_quality_distance},
        "checks": final_checks,
        "decision": "ADMIT_QSD_PIXEL_WRITEBACK" if all(d0_checks.values()) and all(final_checks.values()) else "STOP_QSD_BEFORE_PIXEL_WRITEBACK",
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    save(result_root / f"qsd_e1_e2_value_gate_seed{args.seed}.json", e1_payload)
    summary = {
        "protocol": PROTOCOL, "d0_e0_decision": d0_payload["decision"], "e1_e2_decision": e1_payload["decision"],
        "pixel_writeback_allowed": e1_payload["decision"] == "ADMIT_QSD_PIXEL_WRITEBACK",
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    save(result_root / f"qsd_summary_seed{args.seed}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
