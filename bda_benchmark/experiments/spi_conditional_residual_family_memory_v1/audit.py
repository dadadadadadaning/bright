from __future__ import annotations

import argparse, json
from pathlib import Path
import torch

from bda_benchmark.experiments.spi_independent_building_memory_v1.audit import load_independent, subset
from bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.core import (
    FactorizedCaseMetric, bounded_observable_memory_rows, prepare_groups, retrieve, row_events,
)
from . import PROTOCOL
from .core import (family_report, granularity_targets, high_risk_mask, memory_features,
                   source_quantile_thresholds, train_head)


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser("Conditioned Residual Family Memory audit")
    p.add_argument("--train-cards", required=True); p.add_argument("--calibration-cards", required=True)
    p.add_argument("--metric-checkpoint", required=True); p.add_argument("--result-root", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--risk-quantile", type=float, default=.80)
    p.add_argument("--capacity-per-event-source", type=int, default=256); p.add_argument("--coarse-k", type=int, default=64)
    p.add_argument("--topk", type=int, default=8); p.add_argument("--maximum-per-event", type=int, default=2)
    p.add_argument("--quality-limit", type=float, default=3.); p.add_argument("--area-ratio-limit", type=float, default=2.)
    p.add_argument("--head-steps", type=int, default=600); p.add_argument("--head-batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--seed", type=int, default=7341)
    p.add_argument("--predictions-output", default=None,
                   help="Optional deterministic per-candidate prediction cache for read-only D2 attribution.")
    a = p.parse_args(argv); device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    train, calibration = load_independent(a.train_cards), load_independent(a.calibration_cards)
    threshold = source_quantile_thresholds(train["p3_probabilities"], train["p3_predictions"], a.risk_quantile)
    train_mask = high_risk_mask(train["p3_probabilities"], train["p3_predictions"], threshold)
    calibration_mask = high_risk_mask(calibration["p3_probabilities"], calibration["p3_predictions"], threshold)
    query_train, query_cal = subset(train, torch.nonzero(train_mask, as_tuple=False).flatten()), subset(calibration, torch.nonzero(calibration_mask, as_tuple=False).flatten())
    train_events, cal_events = row_events(query_train), row_events(query_cal)
    rows = bounded_observable_memory_rows(train, a.capacity_per_event_source, a.seed); memory = subset(train, rows)
    memory_events = [row_events(train)[i] for i in rows.tolist()]
    groups_train = prepare_groups(train, query_train); groups_cal = prepare_groups(train, query_cal)
    checkpoint = torch.load(a.metric_checkpoint, map_location="cpu", weights_only=False)
    model = FactorizedCaseMetric(checkpoint["structure_dim"], checkpoint["damage_dim"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["model_state_dict"]); model.to(device).eval()
    def encode(s, d):
        out_s, out_d = [], []
        with torch.no_grad():
            for start in range(0, len(s), 2048):
                x, y = model.encode(s[start:start+2048].to(device), d[start:start+2048].to(device)); out_s.append(x.cpu()); out_d.append(y.cpu())
        return torch.cat(out_s), torch.cat(out_d)
    ms, md = encode(groups_train["train_structure"][rows], groups_train["train_damage"][rows])
    ts, td = encode(groups_train["query_structure"], groups_train["query_damage"])
    cs, cd = encode(groups_cal["query_structure"], groups_cal["query_damage"])
    m_targets = granularity_targets(memory); correct_rows = torch.nonzero(~m_targets["p3_error"].bool(), as_tuple=False).flatten(); residual_rows = torch.nonzero(m_targets["p3_error"].bool(), as_tuple=False).flatten()
    def build_retrieval(qcache, qevents, qs, qd, quality, local_rows):
        qprob = qcache["p3_probabilities"].float(); qmargin = qprob.topk(2, 1).values.diff(1).abs().squeeze(1)
        mprob = memory["p3_probabilities"].float(); mmargin = mprob.topk(2, 1).values.diff(1).abs().squeeze(1)
        return retrieve(qs, ms[local_rows], qd, md[local_rows], quality, groups_train["train_quality"][rows][local_rows], qevents,
                        [memory_events[i] for i in local_rows.tolist()], qcache["p3_predictions"], memory["p3_predictions"][local_rows],
                        qcache["areas"].float(), memory["areas"][local_rows].float(), qmargin, mmargin[local_rows], mode="learned",
                        coarse_k=a.coarse_k, topk=a.topk, maximum_per_event=a.maximum_per_event, quality_limit=a.quality_limit,
                        area_ratio_limit=a.area_ratio_limit, similarity_device=str(device), seed=a.seed)
    def feature_bundle(qcache, qevents, qs, qd, quality):
        correct = build_retrieval(qcache, qevents, qs, qd, quality, correct_rows)
        residual = build_retrieval(qcache, qevents, qs, qd, quality, residual_rows)
        true = memory_features(correct, residual, m_targets["error_family"][residual_rows])
        perm = torch.randperm(len(residual_rows), generator=torch.Generator().manual_seed(a.seed + 31))
        shuffled = memory_features(correct, residual, m_targets["error_family"][residual_rows][perm])
        return true, shuffled
    print(f"CRFM candidates train={len(query_train['labels'])} calibration={len(query_cal['labels'])}", flush=True)
    train_true, train_shuffled = feature_bundle(query_train, train_events, ts, td, groups_train["query_quality"])
    cal_true, cal_shuffled = feature_bundle(query_cal, cal_events, cs, cd, groups_cal["query_quality"])
    train_target, cal_target = granularity_targets(query_train)["error_family"], granularity_targets(query_cal)["error_family"]
    p3_train, p3_cal = query_train["p3_probabilities"].float(), query_cal["p3_probabilities"].float()
    variants = {"a0_p3_only": (p3_train, p3_cal), "a1_true_memory": (torch.cat((p3_train, train_true), 1), torch.cat((p3_cal, cal_true), 1)),
                "a2_shuffled_memory": (torch.cat((p3_train, train_shuffled), 1), torch.cat((p3_cal, cal_shuffled), 1)),
                "a3_zero_memory": (torch.cat((p3_train, torch.zeros_like(train_true)), 1), torch.cat((p3_cal, torch.zeros_like(cal_true)), 1))}
    reports, predictions = {}, {}
    for index, (name, (feature_train, feature_cal)) in enumerate(variants.items()):
        head = train_head(feature_train, train_target, steps=a.head_steps, batch_size=a.head_batch_size, learning_rate=a.learning_rate, seed=a.seed + index, device=device)
        with torch.no_grad(): probability = torch.softmax(head(feature_cal), 1)
        reports[name] = family_report(cal_target, probability, cal_events)
        predictions[name] = probability.cpu()
        print(json.dumps({"variant": name, **{k: v for k, v in reports[name].items() if k != "events"}}), flush=True)
    true, base, shuffled, zero = reports["a1_true_memory"], reports["a0_p3_only"], reports["a2_shuffled_memory"], reports["a3_zero_memory"]
    events = {name: reports[name]["events"] for name in reports}; nonnegative = sum(events["a1_true_memory"].get(e, -1) >= events["a0_p3_only"].get(e, 1e9) for e in events["a1_true_memory"])
    checks = {"candidate_fraction_between_0p05_and_0p30": .05 <= float(calibration_mask.float().mean()) <= .30,
              "true_memory_auc_beats_p3_by_0p02": true["macro_ovr_auroc"] >= base["macro_ovr_auroc"] + .02,
              "true_memory_f1_beats_p3_by_0p01": true["macro_f1"] >= base["macro_f1"] + .01,
              "true_memory_beats_shuffled_by_0p01_f1": true["macro_f1"] >= shuffled["macro_f1"] + .01,
              "true_memory_beats_zero_by_0p01_f1": true["macro_f1"] >= zero["macro_f1"] + .01,
              "nonnegative_events_ge_10": nonnegative >= 10}
    prediction_path = Path(a.predictions_output) if a.predictions_output else Path(a.result_root) / f"crfm_conditional_family_predictions_seed{a.seed}.pt"
    torch.save({"protocol": PROTOCOL, "target": cal_target.cpu(), "p3_source": query_cal["p3_predictions"].long().cpu(),
                "events": cal_events, "probabilities": predictions, "risk_quantile": a.risk_quantile,
                "thresholds_by_p3_source": threshold.cpu(), "meta_confirm_accessed": False,
                "val450_accessed": False, "test_accessed": False}, prediction_path)
    payload = {"protocol": PROTOCOL, "thresholds_by_p3_source": threshold.tolist(), "risk_quantile": a.risk_quantile,
               "train_candidate_fraction": float(train_mask.float().mean()), "calibration_candidate_fraction": float(calibration_mask.float().mean()),
               "train_candidates": int(train_mask.sum()), "calibration_candidates": int(calibration_mask.sum()), "reports": reports,
               "nonnegative_events_true_vs_p3": f"{nonnegative}/{len(events['a1_true_memory'])}", "checks": checks,
               "prediction_cache": str(prediction_path),
               "decision": "ADMIT_MEMORY_FAMILY_EXPERT_DEVELOPMENT" if all(checks.values()) else "STOP_CONDITIONAL_MEMORY_FAMILY",
               "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    save(Path(a.result_root) / f"crfm_conditional_family_seed{a.seed}.json", payload); print(json.dumps(payload["checks"], indent=2), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
