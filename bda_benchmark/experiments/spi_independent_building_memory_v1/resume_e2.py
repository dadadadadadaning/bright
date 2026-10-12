from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import PROTOCOL
from .audit import (compare_report, load_independent, memory_rows, random_cross_event_support,
                    role_mean, row_events, save_json, train_role_arm)
from .core import ConditionalResidualModel, group_matrix
from bda_benchmark.experiments.spi_memory_roles_v1.core import retrieval_report


def encode(model, value, probability, batch_size, device):
    output = []
    model = model.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(value), batch_size):
            stop = min(start + batch_size, len(value))
            _, embedding, _ = model(value[start:stop].to(device), probability[start:stop].to(device))
            output.append(embedding.cpu())
    return torch.cat(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Resume only bounded MI-E2 dependency audit")
    parser.add_argument("--train-independent", required=True); parser.add_argument("--calibration-independent", required=True)
    parser.add_argument("--model-checkpoint", required=True); parser.add_argument("--d0-json", required=True)
    parser.add_argument("--d1-json", required=True); parser.add_argument("--result-root", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--role-steps", type=int, default=400); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--maximum-per-event", type=int, default=2); parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train = load_independent(args.train_independent); cal = load_independent(args.calibration_independent)
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL: raise RuntimeError("MI-E1 checkpoint protocol mismatch")
    groups = tuple(checkpoint["groups"]); statistics = checkpoint["statistics"]
    train_x, _ = group_matrix(train, groups, statistics); cal_x, _ = group_matrix(cal, groups, statistics)
    config = checkpoint.get("configuration", {})
    model = ConditionalResidualModel(train_x.shape[1], int(config.get("hidden", args.hidden)),
                                     int(config.get("prototypes_per_class", 3)))
    model.load_state_dict(checkpoint["model_state_dict"])
    train_embedding = encode(model, train_x, train["p3_probabilities"].float(), args.batch_size, device)
    cal_embedding = encode(model, cal_x, cal["p3_probabilities"].float(), args.batch_size, device)
    train_events = row_events(train); cal_events = row_events(cal); target = cal["labels"].long()

    d1 = json.load(open(args.d1_json, encoding="utf-8"))
    selected_name = max(d1["results"], key=lambda key: d1["results"][key]["macro_ovr_auroc"])
    if selected_name.startswith("high_purity_"):
        method, capacity_text = "high_purity", selected_name[len("high_purity_"):]
    else:
        method, capacity_text = selected_name.split("_", 1)
    capacity = 0 if capacity_text == "all" else int(capacity_text)
    selected_rows = memory_rows(train, train_embedding, capacity, method, args.seed)
    selected_events = [train_events[i] for i in selected_rows.tolist()]
    _, _, train_local = retrieval_report(
        train_embedding, train_embedding[selected_rows], train["labels"].long(),
        train["labels"][selected_rows].long(), train_events, selected_events,
        topk=args.topk, maximum_per_event=args.maximum_per_event, device=str(device))
    _, _, cal_local = retrieval_report(
        cal_embedding, train_embedding[selected_rows], target, train["labels"][selected_rows].long(),
        cal_events, selected_events, topk=args.topk, maximum_per_event=args.maximum_per_event, device=str(device))
    true_train = selected_rows[train_local]; true_cal = selected_rows[cal_local]
    random_train = selected_rows[random_cross_event_support(selected_events, train_events, args.topk, args.seed+1)]
    random_cal = selected_rows[random_cross_event_support(selected_events, cal_events, args.topk, args.seed+2)]
    permutation = selected_rows[torch.randperm(len(selected_rows), generator=torch.Generator().manual_seed(args.seed+3))]
    self_train = torch.arange(len(train["labels"]))[:, None].repeat(1, args.topk)
    placeholders_train = torch.zeros_like(true_train); placeholders_cal = torch.zeros_like(true_cal)
    arms = {"true_cross_event": (true_train, true_cal), "random_cross_event": (random_train, random_cal),
            "shuffled_correspondence": (permutation[train_local], permutation[cal_local]),
            "self_roles": (self_train, placeholders_cal), "zero_support": (placeholders_train, placeholders_cal)}
    train_roles = role_mean(train); cal_roles = role_mean(cal); reports, gates = {}, {}
    for name, (train_support, cal_support) in arms.items():
        print(f"MI-E2 {name}: training", flush=True)
        probability, gate = train_role_arm(name, train, cal, train_roles, cal_roles,
                                            train_support, cal_support, args, device)
        reports[name] = compare_report(target, cal["p3_probabilities"].float(), probability, cal_events)
        gates[name] = gate
        print(f"MI-E2 {name}: F1={reports[name]['macro_f1']:.4f} gate={gate:+.4f}", flush=True)
    true_f1 = reports["true_cross_event"]["macro_f1"]
    checks = {"true_beats_random_by_0p50pp": true_f1 >= reports["random_cross_event"]["macro_f1"] + .005,
              "true_beats_shuffled_by_0p50pp": true_f1 >= reports["shuffled_correspondence"]["macro_f1"] + .005,
              "true_beats_self_by_0p30pp": true_f1 >= reports["self_roles"]["macro_f1"] + .003,
              "nonnegative_events_ge_10": reports["true_cross_event"]["nonnegative_events"] >= 10}
    result = Path(args.result_root); result.mkdir(parents=True, exist_ok=True)
    save_json(result/"mi_e2_role_dependency.json", {"protocol": PROTOCOL,
              "selected_d1_memory": {"variant": selected_name, "rows": int(len(selected_rows))},
              "reports": reports, "gates": gates, "checks": checks,
              "decision": "ADMIT_INDEPENDENT_MEMORY_MODEL_INTEGRATION" if all(checks.values()) else "STOP_MEMORY_ROLE_DEPENDENCY"})
    d0 = json.load(open(args.d0_json, encoding="utf-8")); d0_checks = d0["checks"]
    e1 = d0["conditional_e1"]
    summary = {"protocol": PROTOCOL, "stages": {
        "mi_d0": all(d0_checks.values()), "mi_d1": "reported", "mi_d2": "reported",
        "mi_e1": e1["gain_macro_auc"] >= .01 and e1["nonnegative_events"] >= 10,
        "mi_e2": all(checks.values())},
        "memory_model_integration_allowed": all(checks.values()) and d0_checks["beats_shuffled_by_0p01_auc"],
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    save_json(result/"independent_memory_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("Bounded MI-E2 resume complete. Meta-confirm/Val450/Test: NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
