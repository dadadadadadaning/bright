# -*- coding: utf-8 -*-
"""Train RM5-A cross-event metric adapter on train-only P3 region caches."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch

from model.RM5_CrossEventMetric import (
    CrossEventResidualMetric,
    cross_event_supervised_contrastive_loss,
    embedding_preservation_loss,
)
from script.standard_ML.audit_RM_region_retrieval import load_cache
from script.standard_ML.rm_region_retrieval_runtime import (
    class_balanced_knn,
    complement_audit,
    metrics_from_predictions,
    selective_damaged_rescue_audit,
    validate_disjoint_ids,
)


def event_codes(events: Sequence[str]) -> torch.Tensor:
    names = sorted(set(map(str, events)))
    lookup = {name: index for index, name in enumerate(names)}
    return torch.tensor([lookup[str(item)] for item in events], dtype=torch.long)


def class_event_index(
    labels: torch.Tensor, codes: torch.Tensor
) -> Dict[int, Dict[int, torch.Tensor]]:
    result: Dict[int, Dict[int, torch.Tensor]] = {}
    for cls in range(3):
        result[cls] = {}
        for event in torch.unique(codes[labels == cls]).tolist():
            indices = torch.nonzero(
                (labels == cls) & (codes == int(event)), as_tuple=False
            ).flatten()
            if indices.numel():
                result[cls][int(event)] = indices
        if len(result[cls]) < 2:
            raise RuntimeError(
                f"class {cls} needs at least two events for cross-event positives"
            )
    return result


def sample_class_event_batch(
    pools: Mapping[int, Mapping[int, torch.Tensor]],
    *,
    events_per_class: int,
    samples_per_event: int,
    generator: torch.Generator,
) -> torch.Tensor:
    selected = []
    for cls in range(3):
        event_names = sorted(pools[cls])
        take_events = min(int(events_per_class), len(event_names))
        event_order = torch.randperm(len(event_names), generator=generator)[:take_events]
        for position in event_order.tolist():
            indices = pools[cls][event_names[int(position)]]
            if indices.numel() >= int(samples_per_event):
                order = torch.randperm(indices.numel(), generator=generator)[
                    : int(samples_per_event)
                ]
            else:
                order = torch.randint(
                    indices.numel(),
                    (int(samples_per_event),),
                    generator=generator,
                )
            selected.append(indices[order])
    if not selected:
        raise RuntimeError("class-event sampler produced an empty batch")
    output = torch.cat(selected)
    return output[torch.randperm(output.numel(), generator=generator)]


@torch.inference_mode()
def project_all(
    model: CrossEventResidualMetric,
    feature: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int = 1024,
) -> torch.Tensor:
    outputs = []
    for start in range(0, feature.shape[0], int(chunk_size)):
        outputs.append(model(feature[start : start + int(chunk_size)].to(device)).cpu())
    return torch.cat(outputs)


@torch.inference_mode()
def evaluate(
    model: CrossEventResidualMetric,
    memory: Mapping[str, object],
    query: Mapping[str, object],
    *,
    device: torch.device,
    topk: int,
    beta: float,
    max_neighbors_per_event: int,
) -> Dict[str, object]:
    memory_key = project_all(model, memory["keys"], device=device)
    query_key = project_all(model, query["keys"], device=device)
    retrieval = class_balanced_knn(
        query_key.to(device),
        memory_key.to(device),
        memory["labels"],
        topk_per_class=int(topk),
        beta=float(beta),
        query_events=query["events"],
        memory_events=memory["events"],
        exclude_same_event=True,
        max_neighbors_per_event=int(max_neighbors_per_event),
    )
    prediction = retrieval["prediction"]
    metrics = metrics_from_predictions(query["labels"], prediction, query["events"])
    base_metrics = metrics_from_predictions(
        query["labels"], query["base_predictions"], query["events"]
    )
    complement = complement_audit(
        query["labels"], query["base_predictions"], prediction
    )
    event_delta = {
        event: 100.0
        * (metrics["events"][event]["macro_f1"] - base_metrics["events"][event]["macro_f1"])
        for event in metrics["events"]
    }
    confidence = selective_damaged_rescue_audit(
        query["labels"],
        query["base_predictions"],
        query["base_probabilities"],
        prediction,
        retrieval["scores"],
        query["events"],
    )
    strict_points = [
        point for point in confidence["operating_points"]
        if point["net_correction"] > 0
        and point["precision"] >= 0.60
        and point["rescued"] >= 10
        and point["nonnegative_events"] >= 10
    ]
    best_auc = float(confidence["signal_auroc"][confidence["best_signal"]])
    checks = {
        "confidence_auroc_at_least_0_70": best_auc >= 0.70,
        "strict_selective_rescue_point_exists": bool(strict_points),
        "damaged_net_complement_positive": (
            complement["per_class"]["damaged"]["net_complement"] > 0
        ),
        "ten_events_nondecreasing_vs_p3": sum(
            value >= 0.0 for value in event_delta.values()
        ) >= 10,
    }
    return {
        "metrics": metrics,
        "base_metrics": base_metrics,
        "complement": complement,
        "event_macro_f1_delta_pp_vs_p3": event_delta,
        "nonnegative_events_vs_p3": sum(value >= 0.0 for value in event_delta.values()),
        "confidence_audit": confidence,
        "strict_best_point": (
            max(strict_points, key=lambda item: item["net_correction"])
            if strict_points else None
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    model: CrossEventResidualMetric,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    validation: Mapping[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "protocol": "RM5_cross_event_metric_v2",
        "variant": str(args.variant),
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "validation": dict(validation),
        "config": vars(args),
        "test_accessed": False,
    }


def apply_safety_gate(
    result: Dict[str, object], reference: Mapping[str, object]
) -> Dict[str, object]:
    """Protect the useful raw retrieval geometry while learning confidence."""

    macro = float(result["metrics"]["macro_f1"])
    reference_macro = float(reference["metrics"]["macro_f1"])
    damaged_net = int(result["complement"]["per_class"]["damaged"]["net_complement"])
    reference_damaged_net = int(
        reference["complement"]["per_class"]["damaged"]["net_complement"]
    )
    minimum_damaged_net = math.ceil(0.80 * reference_damaged_net)
    safety_checks = {
        "macro_f1_drop_at_most_1pp": macro >= reference_macro - 0.01,
        "retains_80pct_initial_damaged_net": damaged_net >= minimum_damaged_net,
    }
    result["safety"] = {
        "reference_macro_f1": reference_macro,
        "macro_f1_delta_pp": 100.0 * (macro - reference_macro),
        "reference_damaged_net": reference_damaged_net,
        "minimum_damaged_net": int(minimum_damaged_net),
        "observed_damaged_net": damaged_net,
        "checks": safety_checks,
        "passed": all(safety_checks.values()),
    }
    result["checks"].update(safety_checks)
    result["passed"] = all(result["checks"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser("RM5-A train-only cross-event metric learning")
    parser.add_argument("--variant", choices=("rm5a", "rm5b"), default="rm5a")
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--optimizer_steps", type=int, default=1500)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--events_per_class", type=int, default=4)
    parser.add_argument("--samples_per_event", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--preserve_weight", type=float, default=5.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--max_neighbors_per_event", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if args.optimizer_steps < 0 or args.eval_every <= 0:
        raise ValueError("optimizer_steps must be nonnegative and eval_every positive")
    if float(args.preserve_weight) < 0.0:
        raise ValueError("preserve_weight cannot be negative")
    effective_preserve_weight = (
        0.0 if args.variant == "rm5a" else float(args.preserve_weight)
    )
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    memory = load_cache(args.memory, "memory")
    query = load_cache(args.queries, "query")
    validate_disjoint_ids(memory["image_ids"], query["image_ids"])
    if memory["metadata"]["p3_checkpoint_sha256"] != query["metadata"]["p3_checkpoint_sha256"]:
        raise RuntimeError("memory/query P3 checkpoint mismatch")
    if "base_probabilities" not in query:
        raise KeyError("RM5 requires P3 base_probabilities in the query cache")

    feature = memory["keys"].float().cpu()
    labels = memory["labels"].long().cpu()
    codes = event_codes(memory["events"])
    pools = class_event_index(labels, codes)
    model = CrossEventResidualMetric(feature.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        identity_delta = float(
            (
                model(feature[: min(256, feature.shape[0])].to(device)).cpu()
                - torch.nn.functional.normalize(
                    feature[: min(256, feature.shape[0])], dim=1
                )
            ).abs().max()
        )
    if identity_delta > 1e-7:
        raise AssertionError(f"RM5 step-zero identity guard failed: {identity_delta}")

    history = []
    initial = evaluate(
        model, memory, query, device=device, topk=args.topk, beta=args.beta,
        max_neighbors_per_event=args.max_neighbors_per_event,
    )
    initial["step"] = 0
    initial["identity_max_abs_delta"] = identity_delta
    initial = apply_safety_gate(initial, initial)
    history.append(initial)
    best_safe_validation = initial
    best_safe_step = 0
    best_auc_validation = initial
    best_auc_step = 0

    def safe_selection_key(result: Mapping[str, object]):
        confidence = result["confidence_audit"]
        auc = confidence["signal_auroc"][confidence["best_signal"]]
        return (
            bool(result["passed"]),
            bool(result["safety"]["passed"]),
            float(auc),
            float(result["metrics"]["event_macro_f1"]),
            float(result["metrics"]["macro_f1"]),
        )

    def auc_selection_key(result: Mapping[str, object]):
        confidence = result["confidence_audit"]
        return float(confidence["signal_auroc"][confidence["best_signal"]])

    atomic_torch_save(
        checkpoint_payload(
            model, optimizer, step=0, validation=initial, args=args
        ),
        output_dir / "best_metric.pth",
    )
    atomic_torch_save(
        checkpoint_payload(
            model, optimizer, step=0, validation=initial, args=args
        ),
        output_dir / "best_auc_diagnostic.pth",
    )
    print(
        "Initial RM5 | "
        f"macro={100*initial['metrics']['macro_f1']:.4f} | "
        f"event={100*initial['metrics']['event_macro_f1']:.4f} | "
        f"D-net={initial['complement']['per_class']['damaged']['net_complement']} | "
        f"AUC={initial['confidence_audit']['signal_auroc'][initial['confidence_audit']['best_signal']]:.4f}",
        flush=True,
    )

    model.train()
    for step in range(1, int(args.optimizer_steps) + 1):
        batch = sample_class_event_batch(
            pools,
            events_per_class=args.events_per_class,
            samples_per_event=args.samples_per_event,
            generator=generator,
        )
        embedding = model(feature[batch].to(device))
        contrastive_loss = cross_event_supervised_contrastive_loss(
            embedding,
            labels[batch].to(device),
            codes[batch].to(device),
            temperature=args.temperature,
        )
        original_embedding = torch.nn.functional.normalize(
            feature[batch].to(device), dim=1
        )
        preserve_loss = embedding_preservation_loss(
            embedding, original_embedding
        )
        loss = contrastive_loss + effective_preserve_weight * preserve_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite RM5 loss at step {step}: {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % int(args.log_every) == 0:
            print(
                f"RM5 train | step={step} | loss={float(loss):.6f} | "
                f"contrastive={float(contrastive_loss):.6f} | "
                f"preserve={float(preserve_loss):.6f} | "
                f"preserve_weight={effective_preserve_weight:.4f}",
                flush=True,
            )

        if step % int(args.eval_every) == 0 or step == int(args.optimizer_steps):
            model.eval()
            validation = evaluate(
                model, memory, query, device=device, topk=args.topk, beta=args.beta,
                max_neighbors_per_event=args.max_neighbors_per_event,
            )
            validation["step"] = int(step)
            validation = apply_safety_gate(validation, initial)
            history.append(validation)
            auc = validation["confidence_audit"]["signal_auroc"][
                validation["confidence_audit"]["best_signal"]
            ]
            print(
                f"RM5 validation | step={step} | "
                f"macro={100*validation['metrics']['macro_f1']:.4f} | "
                f"event={100*validation['metrics']['event_macro_f1']:.4f} | "
                f"D-net={validation['complement']['per_class']['damaged']['net_complement']} | "
                f"AUC={auc:.4f} | safety={validation['safety']['passed']} | "
                f"passed={validation['passed']}",
                flush=True,
            )
            if safe_selection_key(validation) > safe_selection_key(best_safe_validation):
                best_safe_validation = validation
                best_safe_step = int(step)
                atomic_torch_save(
                    checkpoint_payload(
                        model, optimizer, step=step,
                        validation=validation, args=args,
                    ),
                    output_dir / "best_metric.pth",
                )
                print(f"New best safe RM5 checkpoint: step={step}", flush=True)
            if auc_selection_key(validation) > auc_selection_key(best_auc_validation):
                best_auc_validation = validation
                best_auc_step = int(step)
                atomic_torch_save(
                    checkpoint_payload(
                        model, optimizer, step=step,
                        validation=validation, args=args,
                    ),
                    output_dir / "best_auc_diagnostic.pth",
                )
                print(f"New best AUROC diagnostic checkpoint: step={step}", flush=True)
            model.train()

        if step % max(int(args.eval_every), 500) == 0 or step == int(args.optimizer_steps):
            atomic_torch_save(
                checkpoint_payload(
                    model, optimizer, step=step,
                    validation=history[-1], args=args,
                ),
                output_dir / "latest_resume.pth",
            )

    summary = {
        "protocol": "RM5_cross_event_metric_summary_v2",
        "variant": str(args.variant),
        "effective_preserve_weight": effective_preserve_weight,
        "completed_step": int(args.optimizer_steps),
        "selected_step": int(best_safe_step),
        "best_auc_diagnostic_step": int(best_auc_step),
        "selection_order": "strict gate -> safety gate -> confidence AUROC -> event macro F1 -> macro F1",
        "best_validation": best_safe_validation,
        "best_auc_diagnostic_validation": best_auc_validation,
        "history": history,
        "decision": (
            "PROCEED_TO_TRAIN_ONLY_RETRIEVAL_CALIBRATION_DO_NOT_TEST"
            if best_safe_validation["passed"]
            else "STOP_RM5_METRIC_NOT_RELIABLE_DO_NOT_TRAIN_FUSION"
        ),
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Selected safe RM5 step: {best_safe_step}")
    print(f"Best AUROC diagnostic step: {best_auc_step}")
    print(f"Decision: {summary['decision']}")
    print(f"Saved: {output_dir}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
