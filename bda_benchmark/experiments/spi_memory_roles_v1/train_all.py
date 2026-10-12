from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bda_benchmark.experiments.spi_complete_building_memory_v1.audit_suitability import (
    _events, _validate, load_cards,
)
from bda_benchmark.experiments.spi_complete_building_memory_v1.audit_semantic_controls import _key_pair
from bda_benchmark.experiments.spi_complete_building_memory_v1.core import (
    balanced_memory_indices, binary_auroc,
)

from . import PROTOCOL
from .core import (
    HardCaseRouter, MetricTeacher, PrototypeClassifier, RoleSupportAdapter,
    balanced_weights, classification_report, cross_event_contrastive,
    event_ids, event_prototype_alignment, hard_negative_margin,
    retrieval_report, role_means, soft_cross_entropy,
)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def limit_rows(cache: dict, maximum: int) -> dict:
    if maximum <= 0 or len(cache["labels"]) <= maximum:
        return cache
    output = dict(cache); size = len(cache["labels"])
    for name, value in cache.items():
        if torch.is_tensor(value) and value.ndim and len(value) == size:
            output[name] = value[:maximum]
        elif isinstance(value, dict):
            output[name] = {
                key: item[:maximum] if torch.is_tensor(item) and item.ndim and len(item) == size else item
                for key, item in value.items()
            }
    return output


def train_metric(name, x, soft, label, purity, event, args, device,
                 *, event_balance=True, hard_negative=True, soft_target=True):
    torch.manual_seed(args.seed + sum(name.encode()))
    model = MetricTeacher(x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    sampling = balanced_weights(label, event, purity) if event_balance else torch.ones(len(x)) / len(x)
    generator = torch.Generator().manual_seed(args.seed + sum(name.encode()) + 17)
    batch_size = min(args.batch_size, len(x))
    for step in range(1, args.metric_steps + 1):
        ids = torch.multinomial(sampling, batch_size, replacement=True, generator=generator)
        bx = x[ids].to(device); by = label[ids].to(device); be = event[ids].to(device)
        logits, embedding = model(bx)
        if soft_target:
            semantic = soft_cross_entropy(logits, soft[ids].to(device), purity[ids].to(device))
        else:
            semantic = F.cross_entropy(logits, by)
        contrast = cross_event_contrastive(embedding, by, be, args.temperature)
        hard = hard_negative_margin(embedding, by, be) if hard_negative else embedding.sum() * 0
        prototype = event_prototype_alignment(embedding, by, be)
        loss = semantic + args.contrast_weight * contrast + args.hard_negative_weight * hard + args.prototype_weight * prototype
        optimizer.zero_grad(); loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step == 1 or step % args.progress_every == 0:
            print(json.dumps({"stage": "R-M1", "variant": name, "step": step,
                              "loss": float(loss.detach()), "semantic": float(semantic.detach()),
                              "contrast": float(contrast.detach()), "hard_negative": float(hard.detach()),
                              "prototype": float(prototype.detach()), "gradient_norm": float(gradient)}), flush=True)
    return model.cpu()


@torch.no_grad()
def encode(model, x, device, batch=4096):
    model = model.to(device).eval(); logits, embeddings = [], []
    for start in range(0, len(x), batch):
        row_logits, row_embedding = model(x[start:start + batch].to(device))
        logits.append(row_logits.cpu()); embeddings.append(row_embedding.cpu())
    return torch.cat(logits), torch.cat(embeddings)


def train_prototype(x, soft, label, purity, event, args, device):
    torch.manual_seed(args.seed + 211)
    model = PrototypeClassifier(x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    sampling = balanced_weights(label, event, purity); generator = torch.Generator().manual_seed(args.seed + 212)
    for step in range(1, args.prototype_steps + 1):
        ids = torch.multinomial(sampling, min(args.batch_size, len(x)), replacement=True, generator=generator)
        logits, embedding = model(x[ids].to(device)); by = label[ids].to(device); be = event[ids].to(device)
        semantic = soft_cross_entropy(logits, soft[ids].to(device), purity[ids].to(device))
        alignment = event_prototype_alignment(embedding, by, be)
        prototype_similarity = (
            F.normalize(model.prototypes, dim=1)
            @ F.normalize(model.prototypes, dim=1).T
        )
        off_diagonal = ~torch.eye(4, dtype=torch.bool, device=device)
        separation = F.relu(.20 + prototype_similarity[off_diagonal]).mean()
        loss = semantic + args.prototype_weight * alignment + .05 * separation
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step == 1 or step % args.progress_every == 0:
            print(json.dumps({"stage": "R-M2", "step": step, "loss": float(loss.detach()),
                              "alignment": float(alignment.detach()), "separation": float(separation.detach())}), flush=True)
    return model.cpu()


def train_role_adapter(base_model, train_embedding, cal_embedding, train_roles, cal_roles,
                       train_support, cal_support, soft, purity, label, event, args, device):
    model = RoleSupportAdapter(args.hidden, train_roles.shape[-1], base_model.classifier).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate * .5, weight_decay=1e-4)
    sampling = balanced_weights(label, event, purity); generator = torch.Generator().manual_seed(args.seed + 311)
    for step in range(1, args.role_steps + 1):
        ids = torch.multinomial(sampling, min(args.role_batch_size, len(label)), replacement=True, generator=generator)
        logits, _ = model(train_embedding[ids].to(device), train_roles[ids].to(device),
                          train_roles[train_support[ids]].to(device))
        loss = soft_cross_entropy(logits, soft[ids].to(device), purity[ids].to(device))
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step == 1 or step % args.progress_every == 0:
            print(json.dumps({"stage": "R-M3", "step": step, "loss": float(loss.detach()),
                              "gate": float(torch.tanh(model.gate).detach())}), flush=True)
    model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(cal_embedding), args.role_batch_size):
            stop = min(start + args.role_batch_size, len(cal_embedding)); ids = torch.arange(start, stop)
            logits, _ = model(cal_embedding[ids].to(device), cal_roles[ids].to(device),
                              train_roles[cal_support[ids]].to(device))
            output.append(logits.softmax(1).cpu())
    return model.cpu(), torch.cat(output)


def train_router(feature, error, sampling, args, device):
    model = HardCaseRouter(feature.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed + 411)
    positive = int(error.sum()); negative = len(error) - positive
    pos_weight = torch.tensor(negative / max(positive, 1), device=device)
    for step in range(1, args.router_steps + 1):
        ids = torch.multinomial(sampling, min(args.batch_size, len(error)), replacement=True, generator=generator)
        logits = model(feature[ids].to(device)); target = error[ids].float().to(device)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step == 1 or step % args.progress_every == 0:
            print(json.dumps({"stage": "R-M4", "step": step, "loss": float(loss.detach())}), flush=True)
    return model.cpu()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Five sequential complete-building Memory roles")
    parser.add_argument("--train-cards", required=True); parser.add_argument("--calibration-cards", required=True)
    parser.add_argument("--output-root", required=True); parser.add_argument("--result-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden", type=int, default=128); parser.add_argument("--metric-steps", type=int, default=1000)
    parser.add_argument("--prototype-steps", type=int, default=800); parser.add_argument("--role-steps", type=int, default=500)
    parser.add_argument("--router-steps", type=int, default=500); parser.add_argument("--sampler-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--role-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--temperature", type=float, default=.10)
    parser.add_argument("--contrast-weight", type=float, default=.20); parser.add_argument("--hard-negative-weight", type=float, default=.10)
    parser.add_argument("--prototype-weight", type=float, default=.10); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--maximum-per-event", type=int, default=2); parser.add_argument("--maximum-per-event-class", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=100); parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--max-train-rows", type=int, default=0); parser.add_argument("--max-calibration-rows", type=int, default=0)
    args = parser.parse_args(argv)
    if args.topk != 8 or args.maximum_per_event != 2:
        raise RuntimeError("protocol freezes topk=8 and maximum_per_event=2")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_root); output.mkdir(parents=True, exist_ok=True)
    result = Path(args.result_root); result.mkdir(parents=True, exist_ok=True)
    train = limit_rows(load_cards(args.train_cards), args.max_train_rows)
    calibration = limit_rows(load_cards(args.calibration_cards), args.max_calibration_rows)
    _validate(train, "meta_train"); _validate(calibration, "meta_calibration")
    full_train, full_cal = _key_pair(train, calibration, "complete")
    p3_train, p3_cal = _key_pair(train, calibration, "p3_global")
    label = train["labels"].long(); cal_label = calibration["labels"].long()
    soft = train["class_fractions"].float(); purity = train["label_purity"].float()
    train_events = _events(train); cal_events = _events(calibration)
    event = event_ids(train_events)
    memory_rows = balanced_memory_indices(label, train_events,
                                          maximum_per_event_class=args.maximum_per_event_class, seed=args.seed)
    memory_events = [train_events[index] for index in memory_rows.tolist()]

    # R-M1: fixed baselines plus four learned metric variants.
    r1_results = {}
    for name, tx, cx in (("b0_p3_global", p3_train, p3_cal), ("b1_complete_fixed", full_train, full_cal)):
        report, _, _ = retrieval_report(cx, tx[memory_rows], cal_label, label[memory_rows],
                                        cal_events, memory_events, topk=args.topk,
                                        maximum_per_event=args.maximum_per_event, device=str(device))
        r1_results[name] = report
    definitions = {
        "b2_learned_full": (True, True, True),
        "b3_no_event_balance": (False, True, True),
        "b4_no_hard_negative": (True, False, True),
        "b5_hard_majority_label": (True, True, False),
    }
    models = {}
    for name, (balance, hard, soft_target) in definitions.items():
        model = train_metric(name, full_train, soft, label, purity, event, args, device,
                             event_balance=balance, hard_negative=hard, soft_target=soft_target)
        _, train_embedding = encode(model, full_train, device); _, cal_embedding = encode(model, full_cal, device)
        report, _, _ = retrieval_report(cal_embedding, train_embedding[memory_rows], cal_label, label[memory_rows],
                                        cal_events, memory_events, topk=args.topk,
                                        maximum_per_event=args.maximum_per_event, device=str(device))
        r1_results[name] = report; models[name] = model
        torch.save({"protocol": PROTOCOL, "stage": "R-M1", "variant": name,
                    "model_state_dict": model.state_dict(), "input_dim": full_train.shape[1],
                    "hidden": args.hidden, "configuration": vars(args)}, output / f"r_m1_{name}.pth")
    # Negative control: destroy the mapping between building features and labels.
    shuffle_generator = torch.Generator().manual_seed(args.seed + 97)
    shuffle = torch.randperm(len(label), generator=shuffle_generator)
    shuffled_label = label[shuffle]
    shuffled_soft = soft[shuffle]
    shuffled_model = train_metric(
        "c1_shuffled_label", full_train, shuffled_soft, shuffled_label,
        purity, event, args, device, event_balance=True,
        hard_negative=True, soft_target=True,
    )
    _, shuffled_train_embedding = encode(shuffled_model, full_train, device)
    _, shuffled_cal_embedding = encode(shuffled_model, full_cal, device)
    shuffled_report, _, _ = retrieval_report(
        shuffled_cal_embedding, shuffled_train_embedding[memory_rows],
        cal_label, shuffled_label[memory_rows], cal_events, memory_events,
        topk=args.topk, maximum_per_event=args.maximum_per_event,
        device=str(device),
    )
    r1_results["c1_shuffled_label"] = shuffled_report
    torch.save({"protocol": PROTOCOL, "stage": "R-M1", "variant": "c1_shuffled_label",
                "model_state_dict": shuffled_model.state_dict(), "input_dim": full_train.shape[1],
                "hidden": args.hidden, "configuration": vars(args)},
               output / "r_m1_c1_shuffled_label.pth")
    main_r1 = r1_results["b2_learned_full"]; p3_r1 = r1_results["b0_p3_global"]
    r1_checks = {
        "macro_auc_beats_p3_global_by_0p02": main_r1["macro_ovr_auroc"] >= p3_r1["macro_ovr_auroc"] + .02,
        "neighbour_purity_ge_0p70": main_r1["mean_neighbour_purity"] >= .70,
        "damaged_recall_at_8_not_below_p3": main_r1["recall_at_k_by_class"][2] >= p3_r1["recall_at_k_by_class"][2],
        "destroyed_recall_at_8_not_below_p3": main_r1["recall_at_k_by_class"][3] >= p3_r1["recall_at_k_by_class"][3],
        "correct_events_ge_10": main_r1["correct_events"] >= 10,
        "shuffled_label_macro_auc_le_0p55": shuffled_report["macro_ovr_auroc"] <= .55,
    }
    r1_decision = "ADMIT_R_M2_R_M3_MODEL_INTEGRATION" if all(r1_checks.values()) else "STOP_LEARNED_RETRIEVAL_BEFORE_MODEL_INTEGRATION"
    save_json(result / "r_m1_metric_teacher.json", {"protocol": PROTOCOL, "stage": "R-M1", "results": r1_results,
              "checks": r1_checks, "decision": r1_decision})

    b2 = models["b2_learned_full"]; b2_logits_train, train_embedding = encode(b2, full_train, device)
    b2_logits_cal, cal_embedding = encode(b2, full_cal, device)

    # R-M2: external Memory compressed into trainable class prototypes.
    prototype_model = train_prototype(full_train, soft, label, purity, event, args, device)
    proto_train_logits, _ = encode(prototype_model, full_train, device)
    proto_cal_logits, _ = encode(prototype_model, full_cal, device)
    r2_baseline = classification_report(cal_label, b2_logits_cal.softmax(1), cal_events)
    r2_model = classification_report(cal_label, proto_cal_logits.softmax(1), cal_events)
    r2_nonnegative = sum(r2_model["events"][name] >= r2_baseline["events"][name] for name in r2_model["events"])
    r2_checks = {"macro_f1_gain_ge_0p01": r2_model["macro_f1"] >= r2_baseline["macro_f1"] + .01,
                 "damaged_recall_not_below": r2_model["class_recall"][2] >= r2_baseline["class_recall"][2],
                 "destroyed_recall_not_below": r2_model["class_recall"][3] >= r2_baseline["class_recall"][3],
                 "nonnegative_events_ge_10": r2_nonnegative >= 10}
    torch.save({"protocol": PROTOCOL, "stage": "R-M2", "model_state_dict": prototype_model.state_dict(),
                "input_dim": full_train.shape[1], "hidden": args.hidden}, output / "r_m2_prototype_classifier.pth")
    save_json(result / "r_m2_prototype.json", {"protocol": PROTOCOL, "stage": "R-M2", "baseline": r2_baseline,
              "prototype": r2_model, "nonnegative_events": f"{r2_nonnegative}/{len(set(cal_events))}",
              "checks": r2_checks, "decision": "ADMIT_PROTOTYPE_MODEL_INTEGRATION" if all(r2_checks.values()) else "STOP_PROTOTYPE_AFTER_CARD_SCREEN"})

    # Shared cross-event supports for R-M3/R-M4. Supports contain features only.
    _, memory_posterior, cal_support_local = retrieval_report(
        cal_embedding, train_embedding[memory_rows], cal_label, label[memory_rows], cal_events, memory_events,
        topk=args.topk, maximum_per_event=args.maximum_per_event, device=str(device))
    _, train_posterior, train_support_local = retrieval_report(
        train_embedding, train_embedding[memory_rows], label, label[memory_rows], train_events, memory_events,
        topk=args.topk, maximum_per_event=args.maximum_per_event, device=str(device))
    # Invalid neighbours are represented as -1; clamp only for safe indexing.
    # Their retrieval weight is already zero, and full-data coverage is audited.
    train_support = memory_rows[train_support_local.clamp_min(0)]
    cal_support = memory_rows[cal_support_local.clamp_min(0)]

    # R-M3: role-token feature support rather than class-posterior writeback.
    train_roles = role_means(train); cal_roles = role_means(calibration)
    role_model, role_probability = train_role_adapter(
        b2, train_embedding, cal_embedding, train_roles, cal_roles, train_support, cal_support,
        soft, purity, label, event, args, device,
    )
    r3_base = classification_report(cal_label, b2_logits_cal.softmax(1), cal_events)
    r3_model = classification_report(cal_label, role_probability, cal_events)
    r3_nonnegative = sum(r3_model["events"][name] >= r3_base["events"][name] for name in r3_model["events"])
    r3_checks = {"macro_f1_gain_ge_0p01": r3_model["macro_f1"] >= r3_base["macro_f1"] + .01,
                 "nonnegative_events_ge_10": r3_nonnegative >= 10,
                 "gate_nonzero": abs(float(torch.tanh(role_model.gate))) >= .01}
    torch.save({"protocol": PROTOCOL, "stage": "R-M3", "model_state_dict": role_model.state_dict(),
                "hidden": args.hidden, "role_dim": train_roles.shape[-1]}, output / "r_m3_role_support.pth")
    save_json(result / "r_m3_role_support.json", {"protocol": PROTOCOL, "stage": "R-M3", "baseline": r3_base,
              "role_support": r3_model, "gate": float(torch.tanh(role_model.gate)), "checks": r3_checks,
              "decision": "ADMIT_ROLE_SUPPORT_MODEL_INTEGRATION" if all(r3_checks.values()) else "STOP_ROLE_SUPPORT_AFTER_CARD_SCREEN"})

    # R-M4: Memory disagreement predicts P3 hard cases; it never directly emits a pixel label.
    train_error = train["p3_predictions"].long() != label
    cal_error = calibration["p3_predictions"].long() != cal_label
    train_router_x = torch.cat((train_embedding, train["p3_probabilities"].float(), train_posterior,
                                (train_posterior - train["p3_probabilities"].float()).abs()), 1)
    cal_router_x = torch.cat((cal_embedding, calibration["p3_probabilities"].float(), memory_posterior,
                              (memory_posterior - calibration["p3_probabilities"].float()).abs()), 1)
    router_sampling = balanced_weights(label, event, purity)
    router = train_router(train_router_x, train_error, router_sampling, args, device).to(device).eval()
    with torch.no_grad():
        router_score = torch.sigmoid(router(cal_router_x.to(device))).cpu()
    router_auc = binary_auroc(cal_error, router_score)
    p3_probability = calibration["p3_probabilities"].float(); specialist_probability = b2_logits_cal.softmax(1)
    routed = router_score >= .5; routed_probability = p3_probability.clone(); routed_probability[routed] = specialist_probability[routed]
    r4_base = classification_report(cal_label, p3_probability, cal_events)
    r4_model = classification_report(cal_label, routed_probability, cal_events)
    r4_nonnegative = sum(r4_model["events"][name] >= r4_base["events"][name] for name in r4_model["events"])
    r4_checks = {"hard_case_auroc_ge_0p70": router_auc >= .70,
                 "macro_f1_gain_ge_0p01": r4_model["macro_f1"] >= r4_base["macro_f1"] + .01,
                 "nonnegative_events_ge_10": r4_nonnegative >= 10,
                 "routed_fraction_nontrivial": .01 <= float(routed.float().mean()) <= .50}
    torch.save({"protocol": PROTOCOL, "stage": "R-M4", "model_state_dict": router.cpu().state_dict(),
                "input_dim": train_router_x.shape[1], "hidden": args.hidden}, output / "r_m4_hard_case_router.pth")
    save_json(result / "r_m4_hard_case_router.json", {"protocol": PROTOCOL, "stage": "R-M4", "router_auroc": router_auc,
              "routed_fraction": float(routed.float().mean()), "baseline": r4_base, "routed": r4_model,
              "checks": r4_checks, "decision": "ADMIT_HARD_CASE_ROUTER_MODEL_INTEGRATION" if all(r4_checks.values()) else "STOP_HARD_CASE_ROUTER_AFTER_CARD_SCREEN"})

    # R-M5: same inference model; only the train sampler changes.
    original_steps = args.metric_steps; args.metric_steps = args.sampler_steps
    random_model = train_metric("r_m5_random", full_train, soft, label, purity, event, args, device,
                                event_balance=False, hard_negative=False, soft_target=True)
    balanced_model = train_metric("r_m5_event_class_hard", full_train, soft, label, purity, event, args, device,
                                  event_balance=True, hard_negative=True, soft_target=True)
    args.metric_steps = original_steps
    random_logits, _ = encode(random_model, full_cal, device); balanced_logits, _ = encode(balanced_model, full_cal, device)
    r5_random = classification_report(cal_label, random_logits.softmax(1), cal_events)
    r5_balanced = classification_report(cal_label, balanced_logits.softmax(1), cal_events)
    r5_nonnegative = sum(r5_balanced["events"][name] >= r5_random["events"][name] for name in r5_random["events"])
    r5_checks = {"macro_f1_gain_ge_0p01": r5_balanced["macro_f1"] >= r5_random["macro_f1"] + .01,
                 "damaged_recall_not_below": r5_balanced["class_recall"][2] >= r5_random["class_recall"][2],
                 "destroyed_recall_not_below": r5_balanced["class_recall"][3] >= r5_random["class_recall"][3],
                 "nonnegative_events_ge_10": r5_nonnegative >= 10}
    torch.save({"protocol": PROTOCOL, "stage": "R-M5", "model_state_dict": balanced_model.state_dict(),
                "input_dim": full_train.shape[1], "hidden": args.hidden}, output / "r_m5_balanced_sampler.pth")
    save_json(result / "r_m5_sampler.json", {"protocol": PROTOCOL, "stage": "R-M5", "random": r5_random,
              "event_class_hard": r5_balanced, "checks": r5_checks,
              "decision": "ADMIT_MEMORY_CURRICULUM_FOR_P3_TRAINING" if all(r5_checks.values()) else "STOP_MEMORY_CURRICULUM_AFTER_CARD_SCREEN"})

    summary = {
        "protocol": PROTOCOL, "configuration": vars(args),
        "stages": {"r_m1": r1_decision,
                   "r_m2": "ADMIT" if all(r2_checks.values()) else "STOP",
                   "r_m3": "ADMIT" if all(r3_checks.values()) else "STOP",
                   "r_m4": "ADMIT" if all(r4_checks.values()) else "STOP",
                   "r_m5": "ADMIT" if all(r5_checks.values()) else "STOP"},
        "model_integration_allowed": {"prototype": all(r1_checks.values()) and all(r2_checks.values()),
                                      "role_support": all(r1_checks.values()) and all(r3_checks.values()),
                                      "hard_case_router": all(r1_checks.values()) and all(r4_checks.values()),
                                      "training_sampler": all(r5_checks.values())},
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    save_json(result / "memory_roles_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Memory Roles V1 queue complete: {result / 'memory_roles_summary.json'}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
