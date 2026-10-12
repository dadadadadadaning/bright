from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_complete_building_memory_v1.audit_suitability import load_cards as load_complete_cards
from bda_benchmark.experiments.spi_complete_building_memory_v1.core import macro_f1
from bda_benchmark.experiments.spi_memory_roles_v1.core import event_ids, retrieval_report, soft_cross_entropy

from . import PROTOCOL
from .core import (ConditionalResidualModel, cross_event_contrastive, group_matrix,
                   kcenter_indices, multiclass_report, prototype_diversity)


CARD_FIELDS = ("image_index", "region_label", "areas", "geometry", "labels", "label_purity",
               "class_fractions", "p3_predictions", "p3_probabilities", "rgb_structure",
               "sar_physics", "relative", "quality", "role_tokens", "role_token_mask")
GROUPS = {
    "rgb_structure": ("rgb_structure",),
    "sar_physics": ("sar_physics",),
    "rgb_sar": ("rgb_structure", "sar_physics"),
    "rgb_sar_relative": ("rgb_structure", "sar_physics", "relative"),
    "rgb_sar_relative_quality": ("rgb_structure", "sar_physics", "relative", "quality"),
    "complete_independent": ("rgb_structure", "sar_physics", "relative", "quality", "role_tokens"),
}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True)+"\n", encoding="utf-8")


def load_independent(path: str) -> dict:
    manifest = torch.load(path, map_location="cpu", weights_only=False)
    if not manifest.get("sharded"):
        return manifest
    columns = {name: [] for name in CARD_FIELDS}; names, events, offset = [], [], 0
    for shard_path in manifest["shards"]:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for field in CARD_FIELDS:
            value = shard[field]
            if field == "image_index": value = value.long() + offset
            columns[field].append(value)
        names.extend(shard["image_names"]); events.extend(shard["image_events"]); offset += len(shard["image_names"])
    return {**{k: v for k, v in manifest.items() if k not in ("shards", "sharded")},
            **{k: torch.cat(v) for k, v in columns.items()}, "image_names": names, "image_events": events}


def row_events(cache: dict) -> list[str]:
    return [cache["image_events"][int(i)] for i in cache["image_index"].long().tolist()]


def subset(cache: dict, rows: torch.Tensor) -> dict:
    result = dict(cache)
    count = len(cache["labels"])
    for key, value in cache.items():
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == count:
            result[key] = value[rows]
    # Re-indexing images is unnecessary for row_events; retain original image table/indices.
    return result


def align_complete(independent: dict, complete: dict) -> dict:
    old_names = [complete["image_names"][int(i)] for i in complete["image_index"].long().tolist()]
    lookup = {(name, int(region)): row for row, (name, region) in enumerate(zip(old_names, complete["region_label"]))}
    new_names = [independent["image_names"][int(i)] for i in independent["image_index"].long().tolist()]
    rows = []
    for name, region in zip(new_names, independent["region_label"]):
        key = (name, int(region))
        if key not in lookup: raise RuntimeError(f"complete/independent card mismatch: {key}")
        rows.append(lookup[key])
    aligned = subset(complete, torch.tensor(rows, dtype=torch.long))
    if not torch.equal(aligned["labels"].long(), independent["labels"].long()):
        raise RuntimeError("aligned cards disagree on labels")
    return aligned


def limit(cache: dict, maximum: int) -> dict:
    return subset(cache, torch.arange(min(maximum, len(cache["labels"])))) if maximum else cache


def train_conditional(name: str, train_x: torch.Tensor, cal_x: torch.Tensor, train: dict, cal: dict,
                      args, device, *, auxiliary: bool, use_p3: bool = True) -> tuple[ConditionalResidualModel, torch.Tensor, torch.Tensor, torch.Tensor]:
    model = ConditionalResidualModel(train_x.shape[1], args.hidden, args.prototypes_per_class).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed + sum(ord(x) for x in name))
    labels = train["labels"].long(); soft = train["class_fractions"].float(); events = event_ids(row_events(train))
    train_base = train["p3_probabilities"].float() if use_p3 else torch.full_like(train["p3_probabilities"].float(), .25)
    cal_base = cal["p3_probabilities"].float() if use_p3 else torch.full_like(cal["p3_probabilities"].float(), .25)
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(labels), (min(args.batch_size, len(labels)),), generator=generator)
        logits, embedding, proto_logits = model(train_x[ids].to(device), train_base[ids].to(device))
        loss = soft_cross_entropy(logits, soft[ids].to(device), train["label_purity"][ids].float().to(device))
        if auxiliary:
            loss = loss + args.contrast_weight * cross_event_contrastive(
                embedding, labels[ids].to(device), events[ids].to(device))
            loss = loss + args.prototype_weight * F.cross_entropy(proto_logits, labels[ids].to(device))
            loss = loss + args.diversity_weight * prototype_diversity(model.prototypes)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if step == 1 or step % args.progress_every == 0:
            print(json.dumps({"stage": name, "step": step, "loss": float(loss.detach())}), flush=True)
    model.eval(); outputs, embeds = [], []
    with torch.no_grad():
        for start in range(0, len(cal_x), args.batch_size):
            stop = min(start + args.batch_size, len(cal_x))
            logits, embedding, _ = model(cal_x[start:stop].to(device), cal_base[start:stop].to(device))
            outputs.append(logits.softmax(1).cpu()); embeds.append(embedding.cpu())
        train_embed = []
        for start in range(0, len(train_x), args.batch_size):
            stop = min(start + args.batch_size, len(train_x))
            _, embedding, _ = model(train_x[start:stop].to(device), train_base[start:stop].to(device))
            train_embed.append(embedding.cpu())
    return model.cpu(), torch.cat(outputs), torch.cat(train_embed), torch.cat(embeds)


def compare_report(target, base, candidate, events):
    base_report = multiclass_report(target, base, events); report = multiclass_report(target, candidate, events)
    gains = {e: report["events"][e] - base_report["events"][e] for e in report["events"]}
    return {**report, "gain_macro_auc": report["macro_ovr_auroc"] - base_report["macro_ovr_auroc"],
            "gain_macro_f1": report["macro_f1"] - base_report["macro_f1"],
            "nonnegative_events": sum(x >= 0 for x in gains.values()), "event_gains": gains}


def memory_rows(cache: dict, key: torch.Tensor, capacity: int, method: str, seed: int) -> torch.Tensor:
    labels = cache["labels"].long(); events = row_events(cache); generator = torch.Generator().manual_seed(seed)
    selected = []
    for event in sorted(set(events)):
        event_mask = torch.tensor([x == event for x in events])
        for label in range(4):
            rows = torch.nonzero(event_mask & (labels == label), as_tuple=False).flatten()
            if not len(rows): continue
            maximum = len(rows) if capacity <= 0 else min(capacity, len(rows))
            if method == "random": local = torch.randperm(len(rows), generator=generator)[:maximum]
            elif method == "high_purity": local = cache["label_purity"][rows].argsort(descending=True, stable=True)[:maximum]
            elif method == "kcenter": local = kcenter_indices(key[rows], maximum)
            else: raise ValueError(method)
            selected.append(rows[local])
    return torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)


class RoleAdapter(nn.Module):
    def __init__(self, role_dim: int, hidden: int = 128):
        super().__init__(); self.q = nn.Linear(role_dim, hidden); self.kv = nn.Linear(role_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True); self.delta = nn.Linear(hidden, 4)
        # gate=0 gives exact P3 identity while a non-zero delta path gives the gate a first-step gradient.
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, base_probability, query_role, support_role):
        q = self.q(query_role.flatten(1, 2).float())[:, None]
        kv = self.kv(support_role.flatten(2, 3).float())
        value, _ = self.attn(q, kv, kv, need_weights=False)
        return base_probability.clamp_min(1e-6).log() + torch.tanh(self.gate) * self.delta(value[:, 0])


def role_mean(cache):
    valid = cache["role_token_mask"].float()[..., None]
    return (cache["role_tokens"].float() * valid).sum(2) / valid.sum(2).clamp_min(1)


def random_cross_event_support(train_events, query_events, topk, seed):
    generator = torch.Generator().manual_seed(seed); output = []
    for event in query_events:
        candidates = torch.tensor([i for i, x in enumerate(train_events) if x != event], dtype=torch.long)
        output.append(candidates[torch.randint(len(candidates), (topk,), generator=generator)])
    return torch.stack(output)


def train_role_arm(name, train, cal, train_roles, cal_roles, train_support, cal_support, args, device):
    torch.manual_seed(args.seed + 701)
    model = RoleAdapter(train_roles.shape[1] * train_roles.shape[2], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # Every causal arm sees identical initial weights and query batches.
    generator = torch.Generator().manual_seed(args.seed + 702)
    for step in range(1, args.role_steps + 1):
        ids = torch.randint(len(train["labels"]), (min(args.batch_size, len(train["labels"])),), generator=generator)
        if name == "self_roles":
            support = train_roles[ids, None].repeat(1, args.topk, 1, 1)
        elif name == "zero_support":
            support = torch.zeros(len(ids), args.topk, *train_roles.shape[1:])
        else:
            support = train_roles[train_support[ids]]
        logits = model(train["p3_probabilities"][ids].to(device), train_roles[ids].to(device),
                       support.to(device))
        loss = soft_cross_entropy(logits, train["class_fractions"][ids].to(device), train["label_purity"][ids].to(device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        if name == "self_roles":
            support = cal_roles[:, None].repeat(1, args.topk, 1, 1)
        elif name == "zero_support":
            support = torch.zeros(len(cal_roles), args.topk, *cal_roles.shape[1:])
        else:
            support = train_roles[cal_support]
        probability = model(cal["p3_probabilities"].to(device), cal_roles.to(device),
                            support.to(device)).softmax(1).cpu()
    return probability, float(torch.tanh(model.gate).detach().cpu())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Sequential independent-information Memory validation")
    parser.add_argument("--train-independent", required=True); parser.add_argument("--calibration-independent", required=True)
    parser.add_argument("--train-complete", required=True); parser.add_argument("--calibration-complete", required=True)
    parser.add_argument("--output-root", required=True); parser.add_argument("--result-root", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=800); parser.add_argument("--role-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--contrast-weight", type=float, default=.10); parser.add_argument("--prototype-weight", type=float, default=.10)
    parser.add_argument("--diversity-weight", type=float, default=.02); parser.add_argument("--prototypes-per-class", type=int, default=3)
    parser.add_argument("--topk", type=int, default=8); parser.add_argument("--maximum-per-event", type=int, default=2)
    parser.add_argument("--capacities", default="32,64,128,256,512,0"); parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7341); parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-calibration-rows", type=int, default=0)
    args = parser.parse_args(argv)
    if args.topk != 8 or args.maximum_per_event != 2: raise RuntimeError("protocol freezes Top-K=8 and max/event=2")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_root); result = Path(args.result_root); output.mkdir(parents=True, exist_ok=True); result.mkdir(parents=True, exist_ok=True)
    train = limit(load_independent(args.train_independent), args.max_train_rows)
    cal = limit(load_independent(args.calibration_independent), args.max_calibration_rows)
    # Alignment makes P3-global a diagnostic baseline only; it never enters independent keys.
    align_complete(train, load_complete_cards(args.train_complete)); align_complete(cal, load_complete_cards(args.calibration_complete))
    train_events = row_events(train); cal_events = row_events(cal); target = cal["labels"].long(); p3 = cal["p3_probabilities"].float()
    p3_report = multiclass_report(target, p3, cal_events)

    matrices, statistics = {}, {}
    for name, groups in GROUPS.items():
        tx, stat = group_matrix(train, groups); cx, _ = group_matrix(cal, groups, stat)
        matrices[name] = (tx, cx); statistics[name] = stat

    # MI-D0 and MI-D2: conditional information and card ablations.
    ablations = {"p3_global": p3_report}; trained = {}
    for name in ("rgb_structure", "sar_physics", "rgb_sar", "rgb_sar_relative", "rgb_sar_relative_quality", "complete_independent"):
        model, probability, train_embedding, cal_embedding = train_conditional(
            "MI-D0/"+name, *matrices[name], train, cal, args, device, auxiliary=False)
        ablations[name] = compare_report(target, p3, probability, cal_events)
        trained[name] = (model, train_embedding, cal_embedding)
    independent_model, independent_probability, _, _ = train_conditional(
        "MI-D0/independent_only", *matrices["complete_independent"], train, cal, args, device,
        auxiliary=False, use_p3=False)
    independent_only = multiclass_report(target, independent_probability, cal_events)
    full_model, full_probability, train_embedding, cal_embedding = train_conditional(
        "MI-E1/conditional_residual", *matrices["complete_independent"], train, cal, args, device, auxiliary=True)
    e1 = compare_report(target, p3, full_probability, cal_events)
    generator = torch.Generator().manual_seed(args.seed + 91); shuffle = torch.randperm(len(target), generator=generator)
    with torch.no_grad():
        full_model = full_model.to(device)
        shuffled_probability = full_model(
            matrices["complete_independent"][1][shuffle].to(device), cal["p3_probabilities"].to(device))[0].softmax(1).cpu()
        full_model = full_model.cpu()
    shuffled = compare_report(target, p3, shuffled_probability, cal_events)
    d0_checks = {
        "p3_plus_independent_auc_gain_ge_0p01": e1["gain_macro_auc"] >= .01,
        "target_auc_gain_ge_0p02": e1["gain_macro_auc"] >= .02,
        "p3_plus_independent_beats_independent_only": e1["macro_ovr_auroc"] > independent_only["macro_ovr_auroc"],
        "beats_shuffled_by_0p01_auc": e1["macro_ovr_auroc"] >= shuffled["macro_ovr_auroc"] + .01,
        "nonnegative_events_ge_10": e1["nonnegative_events"] >= 10,
    }
    save_json(result/"mi_d0_d2_independent_information.json", {"protocol": PROTOCOL, "p3": p3_report,
              "independent_only": independent_only,
              "ablations": ablations, "conditional_e1": e1, "shuffled_independent": shuffled,
              "checks": d0_checks, "decision": "ADMIT_MI_D1_E1" if d0_checks["p3_plus_independent_auc_gain_ge_0p01"] and d0_checks["beats_shuffled_by_0p01_auc"] else "STOP_NO_P3_INDEPENDENT_INFORMATION"})
    torch.save({"protocol": PROTOCOL, "model_state_dict": full_model.state_dict(), "statistics": statistics["complete_independent"],
                "groups": GROUPS["complete_independent"], "configuration": vars(args)}, output/"mi_e1_conditional_residual.pth")

    # MI-D1: same key, same Top-K, capacity and selection are the only variables.
    capacity_reports = {}
    for capacity in [int(x) for x in args.capacities.split(",")]:
        label = "all" if capacity == 0 else str(capacity)
        for method in ("random", "high_purity", "kcenter"):
            rows = memory_rows(train, train_embedding, capacity, method, args.seed)
            report, _, _ = retrieval_report(cal_embedding, train_embedding[rows], target, train["labels"][rows].long(),
                                             cal_events, [train_events[i] for i in rows.tolist()], topk=8,
                                             maximum_per_event=2, device=str(device))
            report["memory_rows"] = int(len(rows)); capacity_reports[f"{method}_{label}"] = report
            print(f"MI-D1 {method} capacity={label}: AUC={report['macro_ovr_auroc']:.4f} purity={report['mean_neighbour_purity']:.4f}", flush=True)
    save_json(result/"mi_d1_capacity_diversity.json", {"protocol": PROTOCOL, "results": capacity_reports,
              "selection_rule": "choose smallest pre-plateau capacity on meta_calibration; never Val450",
              "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False})

    # MI-E2: support-token causal controls. All arms get the same extra network.
    # Never perform an 88k x 88k train-to-train search. Reuse the best D1 bounded
    # memory, which is the actual deployable store whose dependency is being tested.
    selected_name = max(capacity_reports, key=lambda key: capacity_reports[key]["macro_ovr_auroc"])
    if selected_name.startswith("high_purity_"):
        selected_method, selected_capacity = "high_purity", selected_name[len("high_purity_"):]
    else:
        selected_method, selected_capacity = selected_name.split("_", 1)
    selected_capacity_value = 0 if selected_capacity == "all" else int(selected_capacity)
    selected_rows = memory_rows(train, train_embedding, selected_capacity_value, selected_method, args.seed)
    selected_events = [train_events[i] for i in selected_rows.tolist()]
    _, _, true_train_local = retrieval_report(
        train_embedding, train_embedding[selected_rows], train["labels"].long(),
        train["labels"][selected_rows].long(), train_events, selected_events,
        topk=8, maximum_per_event=2, device=str(device))
    _, _, true_cal_local = retrieval_report(
        cal_embedding, train_embedding[selected_rows], target, train["labels"][selected_rows].long(),
        cal_events, selected_events, topk=8, maximum_per_event=2, device=str(device))
    true_train_support = selected_rows[true_train_local]
    true_cal_support = selected_rows[true_cal_local]
    random_train = selected_rows[random_cross_event_support(selected_events, train_events, 8, args.seed+1)]
    random_cal = selected_rows[random_cross_event_support(selected_events, cal_events, 8, args.seed+2)]
    permuted_selected = selected_rows[torch.randperm(len(selected_rows), generator=torch.Generator().manual_seed(args.seed+3))]
    self_train = torch.arange(len(train["labels"]))[:, None].repeat(1, 8); self_cal = true_cal_support
    train_roles = role_mean(train); cal_roles = role_mean(cal)
    zero_train = torch.zeros_like(true_train_support); zero_cal = torch.zeros_like(true_cal_support)
    arms = {"true_cross_event": (true_train_support, true_cal_support),
            "random_cross_event": (random_train, random_cal),
            "shuffled_correspondence": (permuted_selected[true_train_local], permuted_selected[true_cal_local]),
            "self_roles": (self_train, self_cal), "zero_support": (zero_train, zero_cal)}
    role_reports, gates = {}, {}
    for name, (ts, cs) in arms.items():
        probability, gate = train_role_arm(name, train, cal, train_roles, cal_roles, ts, cs, args, device)
        role_reports[name] = compare_report(target, p3, probability, cal_events); gates[name] = gate
    true_f1 = role_reports["true_cross_event"]["macro_f1"]
    e2_checks = {"true_beats_random_by_0p50pp": true_f1 >= role_reports["random_cross_event"]["macro_f1"] + .005,
                 "true_beats_shuffled_by_0p50pp": true_f1 >= role_reports["shuffled_correspondence"]["macro_f1"] + .005,
                 "true_beats_self_by_0p30pp": true_f1 >= role_reports["self_roles"]["macro_f1"] + .003,
                 "nonnegative_events_ge_10": role_reports["true_cross_event"]["nonnegative_events"] >= 10}
    save_json(result/"mi_e2_role_dependency.json", {"protocol": PROTOCOL,
              "selected_d1_memory": {"variant": selected_name, "rows": int(len(selected_rows))},
              "reports": role_reports, "gates": gates,
              "checks": e2_checks, "decision": "ADMIT_INDEPENDENT_MEMORY_MODEL_INTEGRATION" if all(e2_checks.values()) else "STOP_MEMORY_ROLE_DEPENDENCY"})
    summary = {"protocol": PROTOCOL, "stages": {"mi_d0": all(d0_checks.values()), "mi_d1": "reported",
               "mi_d2": "reported", "mi_e1": e1["gain_macro_auc"] >= .01 and e1["nonnegative_events"] >= 10,
               "mi_e2": all(e2_checks.values())}, "memory_model_integration_allowed": all(e2_checks.values()) and d0_checks["beats_shuffled_by_0p01_auc"],
               "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    save_json(result/"independent_memory_summary.json", summary); print(json.dumps(summary, indent=2), flush=True)
    print("Independent Memory validation complete. Meta-confirm/Val450/test: NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
