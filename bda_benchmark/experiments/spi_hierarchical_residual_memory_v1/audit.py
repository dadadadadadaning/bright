from __future__ import annotations

import argparse, json
from pathlib import Path
import torch

from bda_benchmark.experiments.spi_independent_building_memory_v1.audit import load_independent, subset
from bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.core import (
    FactorizedCaseMetric, bounded_observable_memory_rows, prepare_groups, retrieve, row_events,
)
from . import PROTOCOL
from .core import (TARGET_NAMES, binary_event_report, categorical_retrieval_report,
                   dual_bank_error_score, granularity_targets)


CLASSES = {"p3_source": 4, "gt_class": 4, "affected": 2, "p3_error": 2,
           "error_family": 5, "full_direction": 16}


def save(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser("Target granularity and Correct/Residual bank audit")
    p.add_argument("--train-cards", required=True); p.add_argument("--calibration-cards", required=True)
    p.add_argument("--metric-checkpoint", required=True); p.add_argument("--result-root", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--capacity-per-event-source", type=int, default=256)
    p.add_argument("--coarse-k", type=int, default=64); p.add_argument("--topk", type=int, default=8)
    p.add_argument("--maximum-per-event", type=int, default=2); p.add_argument("--quality-limit", type=float, default=3.)
    p.add_argument("--area-ratio-limit", type=float, default=2.); p.add_argument("--score-temperature", type=float, default=.10)
    p.add_argument("--seed", type=int, default=7341); a = p.parse_args(argv)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu"); result_root = Path(a.result_root)
    train = load_independent(a.train_cards); query = load_independent(a.calibration_cards)
    train_events = row_events(train); query_events = row_events(query)
    rows = bounded_observable_memory_rows(train, a.capacity_per_event_source, a.seed)
    memory = subset(train, rows); memory_events = [train_events[i] for i in rows.tolist()]
    groups = prepare_groups(train, query)
    raw_ms, raw_md, mq = groups["train_structure"][rows], groups["train_damage"][rows], groups["train_quality"][rows]
    raw_qs, raw_qd, qq = groups["query_structure"], groups["query_damage"], groups["query_quality"]
    checkpoint = torch.load(a.metric_checkpoint, map_location="cpu", weights_only=False)
    model = FactorizedCaseMetric(checkpoint["structure_dim"], checkpoint["damage_dim"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["model_state_dict"]); model.to(device).eval()
    def encode(s, d):
        left, right = [], []
        with torch.no_grad():
            for start in range(0, len(s), 2048):
                x, y = model.encode(s[start:start+2048].to(device), d[start:start+2048].to(device))
                left.append(x.cpu()); right.append(y.cpu())
        return torch.cat(left), torch.cat(right)
    learned_ms, learned_md = encode(raw_ms, raw_md); learned_qs, learned_qd = encode(raw_qs, raw_qd)
    query_source = query["p3_predictions"].long(); memory_source = memory["p3_predictions"].long()
    query_area, memory_area = query["areas"].float(), memory["areas"].float()
    query_margin = query["p3_probabilities"].float().topk(2, 1).values.diff(1).abs().squeeze(1)
    memory_margin = memory["p3_probabilities"].float().topk(2, 1).values.diff(1).abs().squeeze(1)

    def retrieve_rows(local_rows: torch.Tensor, learned: bool):
        qs, qd = (learned_qs, learned_qd) if learned else (raw_qs, raw_qd)
        ms, md = (learned_ms[local_rows], learned_md[local_rows]) if learned else (raw_ms[local_rows], raw_md[local_rows])
        return retrieve(
            qs, ms, qd, md, qq, mq[local_rows], query_events,
            [memory_events[i] for i in local_rows.tolist()], query_source, memory_source[local_rows],
            query_area, memory_area[local_rows], query_margin, memory_margin[local_rows],
            mode="learned" if learned else "route", coarse_k=a.coarse_k, topk=a.topk,
            maximum_per_event=a.maximum_per_event, quality_limit=a.quality_limit,
            area_ratio_limit=a.area_ratio_limit, similarity_device=str(device), seed=a.seed,
        )

    all_rows = torch.arange(len(memory["labels"]))
    print("HRM-D1 rule retrieval", flush=True); rule = retrieve_rows(all_rows, False)
    print("HRM-D1 learned error-focused retrieval", flush=True); learned = retrieve_rows(all_rows, True)
    q_target, m_target = granularity_targets(query), granularity_targets(memory)
    error_mask = q_target["p3_error"].bool()
    granularity = {}
    for retrieval_name, retrieval in (("r0_rule_route", rule), ("r1_error_focused_metric", learned)):
        granularity[retrieval_name] = {}
        for name in TARGET_NAMES:
            granularity[retrieval_name][name] = {
                "all": categorical_retrieval_report(retrieval, q_target[name], m_target[name], CLASSES[name], query_events),
                "p3_errors_only": categorical_retrieval_report(retrieval, q_target[name], m_target[name], CLASSES[name], query_events, error_mask),
            }
    target_checks = {
        "affected_auroc_ge_0p65": granularity["r1_error_focused_metric"]["affected"]["all"]["macro_ovr_auroc"] >= .65,
        "p3_error_auroc_ge_0p65": granularity["r1_error_focused_metric"]["p3_error"]["all"]["macro_ovr_auroc"] >= .65,
        "p3_error_event_macro_ge_0p60": granularity["r1_error_focused_metric"]["p3_error"]["all"]["event_macro_auroc"] >= .60,
        "p3_error_correct_events_ge_10": granularity["r1_error_focused_metric"]["p3_error"]["all"]["correct_events"] >= 10,
    }
    d1 = {"protocol": PROTOCOL, "memory_rows": int(len(memory["labels"])), "query_rows": int(len(query["labels"])),
          "p3_error_queries": int(error_mask.sum()), "reports": granularity, "checks": target_checks,
          "decision": "ADMIT_CORRECT_RESIDUAL_BANK_AUDIT" if all(target_checks.values()) else "REPORT_TARGET_GRANULARITY_ONLY",
          "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    save(result_root / f"hrm_d1_target_granularity_seed{a.seed}.json", d1)

    # E0: query both historical P3-correct and historical P3-error banks.
    correct_rows = torch.nonzero(~m_target["p3_error"].bool(), as_tuple=False).flatten()
    residual_rows = torch.nonzero(m_target["p3_error"].bool(), as_tuple=False).flatten()
    print(f"HRM-E0 true banks correct={len(correct_rows)} residual={len(residual_rows)}", flush=True)
    correct_retrieval = retrieve_rows(correct_rows, True); residual_retrieval = retrieve_rows(residual_rows, True)
    coverage = correct_retrieval.coverage & residual_retrieval.coverage
    true_score, difference = dual_bank_error_score(correct_retrieval, residual_retrieval, a.score_temperature)
    true_report = binary_event_report(q_target["p3_error"], true_score, query_events, coverage)
    true_report["mean_similarity_difference"] = float(difference[coverage].mean()) if coverage.any() else 0.

    # Causal control: preserve bank sizes, but randomise which historical cases are called residual.
    permutation = torch.randperm(len(all_rows), generator=torch.Generator().manual_seed(a.seed + 91))
    shuffled_residual = permutation[:len(residual_rows)]; shuffled_correct = permutation[len(residual_rows):]
    print("HRM-E0 shuffled bank control", flush=True)
    shuffled_c = retrieve_rows(shuffled_correct, True); shuffled_r = retrieve_rows(shuffled_residual, True)
    shuffled_coverage = shuffled_c.coverage & shuffled_r.coverage
    shuffled_score, _ = dual_bank_error_score(shuffled_c, shuffled_r, a.score_temperature)
    shuffled_report = binary_event_report(q_target["p3_error"], shuffled_score, query_events, shuffled_coverage)
    uncertainty = 1 - query["p3_probabilities"].float().amax(1)
    p3_report = binary_event_report(q_target["p3_error"], uncertainty, query_events, torch.ones(len(query_source), dtype=torch.bool))
    family = categorical_retrieval_report(residual_retrieval, q_target["error_family"],
                                          m_target["error_family"][residual_rows], 5, query_events, error_mask)
    checks = {
        "dual_bank_error_auroc_ge_0p65": true_report["auroc"] >= .65,
        "dual_bank_event_macro_ge_0p60": true_report["event_macro_auroc"] >= .60,
        "dual_bank_correct_events_ge_10": true_report["correct_events"] >= 10,
        "true_beats_shuffled_by_0p03_auc": true_report["auroc"] >= shuffled_report["auroc"] + .03,
        "true_beats_p3_uncertainty_by_0p02_auc": true_report["auroc"] >= p3_report["auroc"] + .02,
        "both_bank_coverage_ge_0p80": true_report["coverage"] >= .80,
        "error_family_purity_ge_0p45": family["neighbour_purity"] >= .45,
    }
    e0 = {"protocol": PROTOCOL, "banks": {"correct_rows": int(len(correct_rows)), "residual_rows": int(len(residual_rows))},
          "true_dual_bank": true_report, "shuffled_bank_control": shuffled_report,
          "p3_uncertainty_control": p3_report, "residual_error_family": family,
          "configuration": {"topk": a.topk, "coarse_k": a.coarse_k, "score_temperature": a.score_temperature},
          "checks": checks, "decision": "ADMIT_HIERARCHICAL_MEMORY_FAMILY_DEVELOPMENT" if all(checks.values()) else "STOP_HIERARCHICAL_MEMORY_NO_INDEPENDENT_ERROR_SIGNAL",
          "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    save(result_root / f"hrm_e0_correct_residual_banks_seed{a.seed}.json", e0)
    family_allowed = e0["decision"] == "ADMIT_HIERARCHICAL_MEMORY_FAMILY_DEVELOPMENT"
    summary = {"protocol": PROTOCOL, "d1_decision": d1["decision"], "e0_decision": e0["decision"],
               "residual_family_development_allowed": family_allowed,
               "model_integration_allowed": False,
               "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    save(result_root / f"hrm_summary_seed{a.seed}.json", summary); print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
