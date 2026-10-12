from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bda_benchmark.experiments.spi_complete_building_memory_v1.audit_semantic_controls import _key_pair
from bda_benchmark.experiments.spi_complete_building_memory_v1.audit_suitability import _events, _validate, load_cards
from bda_benchmark.experiments.spi_complete_building_memory_v1.core import balanced_memory_indices, binary_auroc
from bda_benchmark.experiments.spi_memory_roles_v1.core import (
    HardCaseRouter, MetricTeacher, RoleSupportAdapter, balanced_weights,
    classification_report, event_ids, retrieval_report, role_means,
    soft_cross_entropy,
)
from bda_benchmark.experiments.spi_memory_roles_v1.train_all import encode, limit_rows

from . import PROTOCOL
from .core import (
    event_nonnegative, gain_pp, p3_uncertainty_features,
    random_cross_event_support, route_probability, shuffle_within_event,
)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def load_teacher(path: str, device: torch.device) -> MetricTeacher:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "SPI_MEMORY_ROLES_V1" or payload.get("variant") != "b2_learned_full":
        raise RuntimeError("expected frozen R-M1 b2_learned_full checkpoint")
    model = MetricTeacher(int(payload["input_dim"]), int(payload["hidden"]))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def materialize_support(
    mode: str, query_roles: torch.Tensor, memory_roles: torch.Tensor,
    local_indices: torch.Tensor | None, topk: int,
) -> torch.Tensor:
    """Materialize only the current minibatch; never expand all card supports."""
    if mode == "memory":
        if local_indices is None:
            raise RuntimeError("memory support requires indices")
        support = memory_roles[local_indices.clamp_min(0)].clone()
        support *= (local_indices >= 0)[..., None, None]
        return support
    if mode == "self":
        return query_roles[:, None].expand(-1, topk, -1, -1)
    if mode == "zero":
        return torch.zeros(
            len(query_roles), topk, query_roles.shape[1], query_roles.shape[2],
            dtype=query_roles.dtype,
        )
    raise ValueError(f"unknown support mode: {mode}")


def train_role_arm(
    name: str, classifier: torch.nn.Linear,
    train_embedding: torch.Tensor, calibration_embedding: torch.Tensor,
    train_roles: torch.Tensor, calibration_roles: torch.Tensor,
    memory_roles: torch.Tensor, support_mode: str,
    train_support_indices: torch.Tensor | None,
    calibration_support_indices: torch.Tensor | None,
    train_soft: torch.Tensor, train_purity: torch.Tensor,
    train_label: torch.Tensor, train_event: torch.Tensor,
    args, device: torch.device,
) -> tuple[RoleSupportAdapter, torch.Tensor]:
    # Identical initialization and minibatches across every support condition.
    torch.manual_seed(args.seed + 310)
    model = RoleSupportAdapter(args.hidden, train_roles.shape[-1], classifier).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate * .5, weight_decay=1e-4)
    sampling = balanced_weights(train_label, train_event, train_purity)
    generator = torch.Generator().manual_seed(args.seed + 311)
    batch_size = min(args.role_batch_size, len(train_label))
    for step in range(1, args.role_steps + 1):
        ids = torch.multinomial(sampling, batch_size, replacement=True, generator=generator)
        batch_support = materialize_support(
            support_mode, train_roles[ids], memory_roles,
            train_support_indices[ids] if train_support_indices is not None else None,
            args.topk,
        )
        logits, _ = model(train_embedding[ids].to(device), train_roles[ids].to(device),
                          batch_support.to(device))
        loss = soft_cross_entropy(logits, train_soft[ids].to(device), train_purity[ids].to(device))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step == 1 or step % args.progress_every == 0:
            print(json.dumps({"stage": "R-M3 dependency", "arm": name, "step": step,
                              "loss": float(loss.detach()),
                              "gate": float(torch.tanh(model.gate).detach())}), flush=True)
    probability = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(calibration_embedding), args.role_batch_size):
            stop = min(start + args.role_batch_size, len(calibration_embedding))
            batch_indices = (calibration_support_indices[start:stop]
                             if calibration_support_indices is not None else None)
            batch_support = materialize_support(
                support_mode, calibration_roles[start:stop], memory_roles,
                batch_indices, args.topk,
            )
            logits, _ = model(
                calibration_embedding[start:stop].to(device),
                calibration_roles[start:stop].to(device),
                batch_support.to(device),
            )
            probability.append(logits.softmax(1).cpu())
    return model.cpu(), torch.cat(probability)


def train_router_arm(
    name: str, train_feature: torch.Tensor, calibration_feature: torch.Tensor,
    train_error: torch.Tensor, train_sampling: torch.Tensor,
    args, device: torch.device,
) -> tuple[HardCaseRouter, torch.Tensor, torch.Tensor, float]:
    torch.manual_seed(args.seed + 410)
    model = HardCaseRouter(train_feature.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed + 411)
    positive = int(train_error.sum()); negative = len(train_error) - positive
    pos_weight = torch.tensor(negative / max(positive, 1), device=device)
    batch_size = min(args.batch_size, len(train_error))
    for step in range(1, args.router_steps + 1):
        ids = torch.multinomial(train_sampling, batch_size, replacement=True, generator=generator)
        logits = model(train_feature[ids].to(device))
        target = train_error[ids].float().to(device)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step == 1 or step % args.progress_every == 0:
            print(json.dumps({"stage": "R-M4 dependency", "arm": name,
                              "step": step, "loss": float(loss.detach())}), flush=True)
    model.eval()
    with torch.no_grad():
        train_score = torch.sigmoid(model(train_feature.to(device))).cpu()
        calibration_score = torch.sigmoid(model(calibration_feature.to(device))).cpu()
    # The route budget is frozen using meta_train only, never calibration labels.
    threshold = float(torch.quantile(train_score, 1.0 - args.target_route_fraction))
    return model.cpu(), train_score, calibration_score, threshold


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("R-M3/R-M4 Memory dependency audit")
    parser.add_argument("--train-cards", required=True)
    parser.add_argument("--calibration-cards", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--role-steps", type=int, default=500)
    parser.add_argument("--router-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--role-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--maximum-per-event", type=int, default=2)
    parser.add_argument("--maximum-per-event-class", type=int, default=256)
    parser.add_argument("--target-route-fraction", type=float, default=.20)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-calibration-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    if args.topk != 8 or args.maximum_per_event != 2:
        raise RuntimeError("protocol freezes topk=8 and maximum_per_event=2")
    if not 0.05 <= args.target_route_fraction <= 0.30:
        raise RuntimeError("target route fraction must be within [0.05, 0.30]")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    result_root = Path(args.result_root); result_root.mkdir(parents=True, exist_ok=True)
    train = limit_rows(load_cards(args.train_cards), args.max_train_rows)
    calibration = limit_rows(load_cards(args.calibration_cards), args.max_calibration_rows)
    _validate(train, "meta_train"); _validate(calibration, "meta_calibration")
    full_train, full_calibration = _key_pair(train, calibration, "complete")
    p3_train, p3_calibration = _key_pair(train, calibration, "p3_global")
    train_label = train["labels"].long(); calibration_label = calibration["labels"].long()
    train_soft = train["class_fractions"].float(); train_purity = train["label_purity"].float()
    train_events = _events(train); calibration_events = _events(calibration)
    train_event_id = event_ids(train_events)

    teacher = load_teacher(args.teacher_checkpoint, device)
    train_logits, train_embedding = encode(teacher, full_train, device)
    calibration_logits, calibration_embedding = encode(teacher, full_calibration, device)
    expert_train_probability = train_logits.softmax(1)
    expert_calibration_probability = calibration_logits.softmax(1)
    p3_train_probability = train["p3_probabilities"].float()
    p3_calibration_probability = calibration["p3_probabilities"].float()

    memory_rows = balanced_memory_indices(
        train_label, train_events,
        maximum_per_event_class=args.maximum_per_event_class, seed=args.seed,
    )
    memory_events = [train_events[index] for index in memory_rows.tolist()]
    _, calibration_posterior, calibration_true_local = retrieval_report(
        calibration_embedding, train_embedding[memory_rows], calibration_label,
        train_label[memory_rows], calibration_events, memory_events,
        topk=args.topk, maximum_per_event=args.maximum_per_event, device=str(device),
    )
    _, train_posterior, train_true_local = retrieval_report(
        train_embedding, train_embedding[memory_rows], train_label,
        train_label[memory_rows], train_events, memory_events,
        topk=args.topk, maximum_per_event=args.maximum_per_event, device=str(device),
    )

    # ------------------------------------------------------------------ R-M3
    train_roles = role_means(train); calibration_roles = role_means(calibration)
    memory_roles = train_roles[memory_rows]
    train_random_local = random_cross_event_support(train_events, memory_events, args.topk, args.seed + 320)
    calibration_random_local = random_cross_event_support(calibration_events, memory_events, args.topk, args.seed + 321)
    role_arms = {
        "a1_true_cross_event": ("memory", train_true_local, calibration_true_local),
        "a2_random_cross_event": ("memory", train_random_local, calibration_random_local),
        "a3_shuffled_correspondence": (
            "memory",
            shuffle_within_event(train_true_local, train_events, args.seed + 322),
            shuffle_within_event(calibration_true_local, calibration_events, args.seed + 323),
        ),
        "a4_self_roles": ("self", None, None),
        "a5_zero_support": ("zero", None, None),
    }
    role_results = {
        "a0_b2_classifier": classification_report(
            calibration_label, expert_calibration_probability, calibration_events),
    }
    role_gates = {}
    for name, (support_mode, train_support_indices, calibration_support_indices) in role_arms.items():
        model, probability = train_role_arm(
            name, teacher.classifier, train_embedding, calibration_embedding,
            train_roles, calibration_roles, memory_roles, support_mode,
            train_support_indices, calibration_support_indices,
            train_soft, train_purity, train_label, train_event_id, args, device,
        )
        role_results[name] = classification_report(calibration_label, probability, calibration_events)
        role_gates[name] = float(torch.tanh(model.gate))
        torch.save({"protocol": PROTOCOL, "stage": "R-M3 dependency", "arm": name,
                    "model_state_dict": model.state_dict(), "configuration": vars(args)},
                   output_root / f"r_m3_{name}.pth")
    true_role = role_results["a1_true_cross_event"]
    role_nonnegative, role_event_total = event_nonnegative(true_role, role_results["a0_b2_classifier"])
    role_checks = {
        "true_beats_random_by_0p50pp": gain_pp(true_role, role_results["a2_random_cross_event"]) >= .50,
        "true_beats_shuffled_by_0p50pp": gain_pp(true_role, role_results["a3_shuffled_correspondence"]) >= .50,
        "true_beats_self_by_0p30pp": gain_pp(true_role, role_results["a4_self_roles"]) >= .30,
        "true_beats_zero_support": true_role["macro_f1"] > role_results["a5_zero_support"]["macro_f1"],
        "nonnegative_events_vs_b2_ge_10": role_nonnegative >= 10,
    }
    role_decision = "ADMIT_MEMORY_ROLE_SUPPORT_CAUSAL_SIGNAL" if all(role_checks.values()) else "STOP_MEMORY_ROLE_SUPPORT_DEPENDENCY"
    save_json(result_root / "r_m3_dependency_audit.json", {
        "protocol": PROTOCOL, "stage": "R-M3 dependency", "reports": role_results,
        "gains_vs_b2_pp": {name: gain_pp(row, role_results["a0_b2_classifier"])
                            for name, row in role_results.items()},
        "gates": role_gates, "nonnegative_events_vs_b2": f"{role_nonnegative}/{role_event_total}",
        "checks": role_checks, "decision": role_decision,
    })

    # ------------------------------------------------------------------ R-M4
    shuffled_train_posterior = shuffle_within_event(train_posterior, train_events, args.seed + 420)
    shuffled_calibration_posterior = shuffle_within_event(
        calibration_posterior, calibration_events, args.seed + 421)
    train_p3_uncertainty = p3_uncertainty_features(p3_train_probability)
    calibration_p3_uncertainty = p3_uncertainty_features(p3_calibration_probability)
    router_features = {
        "b2_p3_confidence": (train_p3_uncertainty, calibration_p3_uncertainty),
        "b3_p3_feature": (
            torch.cat((train_p3_uncertainty, p3_train), 1),
            torch.cat((calibration_p3_uncertainty, p3_calibration), 1),
        ),
        "b4_true_memory_disagreement": (
            torch.cat((train_p3_uncertainty, p3_train, train_posterior,
                       (train_posterior - p3_train_probability).abs()), 1),
            torch.cat((calibration_p3_uncertainty, p3_calibration, calibration_posterior,
                       (calibration_posterior - p3_calibration_probability).abs()), 1),
        ),
        "b5_shuffled_memory_disagreement": (
            torch.cat((train_p3_uncertainty, p3_train, shuffled_train_posterior,
                       (shuffled_train_posterior - p3_train_probability).abs()), 1),
            torch.cat((calibration_p3_uncertainty, p3_calibration, shuffled_calibration_posterior,
                       (shuffled_calibration_posterior - p3_calibration_probability).abs()), 1),
        ),
    }
    train_error = train["p3_predictions"].long() != train_label
    calibration_error = calibration["p3_predictions"].long() != calibration_label
    router_sampling = balanced_weights(train_label, train_event_id, train_purity)
    p3_report = classification_report(calibration_label, p3_calibration_probability, calibration_events)
    expert_report = classification_report(calibration_label, expert_calibration_probability, calibration_events)
    router_results = {"b0_p3": p3_report, "b1_always_expert": expert_report}
    router_diagnostics = {}
    for name, (train_feature, calibration_feature) in router_features.items():
        model, _, score, threshold = train_router_arm(
            name, train_feature, calibration_feature, train_error,
            router_sampling, args, device,
        )
        probability, routed = route_probability(
            p3_calibration_probability, expert_calibration_probability, score, threshold,
        )
        router_results[name] = classification_report(calibration_label, probability, calibration_events)
        router_diagnostics[name] = {
            "router_auroc": binary_auroc(calibration_error, score),
            "threshold_frozen_on_meta_train": threshold,
            "calibration_route_fraction": float(routed.float().mean()),
        }
        torch.save({"protocol": PROTOCOL, "stage": "R-M4 dependency", "arm": name,
                    "model_state_dict": model.state_dict(), "threshold": threshold,
                    "configuration": vars(args)}, output_root / f"r_m4_{name}.pth")
    true_router = router_results["b4_true_memory_disagreement"]
    router_nonnegative, router_event_total = event_nonnegative(true_router, expert_report)
    true_diagnostics = router_diagnostics["b4_true_memory_disagreement"]
    router_checks = {
        "true_memory_beats_always_expert": true_router["macro_f1"] > expert_report["macro_f1"],
        "true_memory_beats_p3_confidence": true_router["macro_f1"] > router_results["b2_p3_confidence"]["macro_f1"],
        "true_memory_beats_shuffled_memory": true_router["macro_f1"] > router_results["b5_shuffled_memory_disagreement"]["macro_f1"],
        "router_auroc_ge_0p75": true_diagnostics["router_auroc"] >= .75,
        "route_fraction_between_0p05_and_0p30": .05 <= true_diagnostics["calibration_route_fraction"] <= .30,
        "nonnegative_events_vs_always_expert_ge_10": router_nonnegative >= 10,
    }
    router_decision = "ADMIT_MEMORY_HARD_CASE_ROUTER_CAUSAL_SIGNAL" if all(router_checks.values()) else "STOP_MEMORY_HARD_CASE_ROUTER_DEPENDENCY"
    save_json(result_root / "r_m4_dependency_audit.json", {
        "protocol": PROTOCOL, "stage": "R-M4 dependency", "reports": router_results,
        "diagnostics": router_diagnostics,
        "gains_vs_p3_pp": {name: gain_pp(row, p3_report) for name, row in router_results.items()},
        "gains_vs_always_expert_pp": {name: gain_pp(row, expert_report) for name, row in router_results.items()},
        "nonnegative_events_vs_always_expert": f"{router_nonnegative}/{router_event_total}",
        "checks": router_checks, "decision": router_decision,
    })

    summary = {
        "protocol": PROTOCOL,
        "configuration": vars(args),
        "r_m3_decision": role_decision,
        "r_m4_decision": router_decision,
        "memory_model_integration_allowed": all(role_checks.values()) or all(router_checks.values()),
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    save_json(result_root / "memory_dependency_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Memory dependency audit complete. Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
