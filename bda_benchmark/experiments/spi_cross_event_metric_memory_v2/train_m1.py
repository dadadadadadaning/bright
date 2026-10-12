from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import PROTOCOL
from .core import (
    CrossEventMetricEncoder, balanced_memory_indices, class_similarity_scores,
    classification_metrics, cross_event_supervised_contrastive,
    event_balanced_weights, event_metrics, event_prototype_loss, prepare_cache,
    probability_from_scores, soft_cross_entropy,
)


VARIANTS = {
    "m1_ce": (0.0, 0.0),
    "m2_cross_event_contrast": (0.20, 0.0),
    "m3_contrast_event_prototype": (0.20, 0.10),
}


def train_one(name, train, calibration, args, device):
    contrast_weight, prototype_weight = VARIANTS[name]
    torch.manual_seed(args.seed + list(VARIANTS).index(name) * 101)
    model = CrossEventMetricEncoder(train.x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    sampling = event_balanced_weights(train)
    generator = torch.Generator().manual_seed(args.seed + 11)
    batch_size = min(args.batch_size, len(train.x))
    for step in range(1, args.steps + 1):
        ids = torch.multinomial(sampling, batch_size, replacement=True, generator=generator)
        x = train.x[ids].to(device); soft = train.soft_label[ids].to(device)
        hard = train.hard_label[ids].to(device); event = train.event[ids].to(device)
        purity = train.purity[ids].to(device)
        logits, embedding = model(x)
        semantic = soft_cross_entropy(logits, soft, purity)
        contrast = cross_event_supervised_contrastive(embedding, hard, event, args.temperature)
        prototype = event_prototype_loss(embedding, hard, event)
        loss = semantic + contrast_weight * contrast + prototype_weight * prototype
        optimizer.zero_grad(); loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % args.progress_every == 0:
            print(json.dumps({"variant": name, "step": step, "loss": float(loss.detach()),
                              "semantic": float(semantic.detach()), "contrast": float(contrast.detach()),
                              "prototype": float(prototype.detach()), "gradient_norm": float(gradient)}), flush=True)
    model.eval()
    with torch.no_grad():
        train_logits, train_embedding = model(train.x.to(device))
        cal_logits, cal_embedding = model(calibration.x.to(device))
    train_embedding = train_embedding.cpu(); cal_embedding = cal_embedding.cpu()
    parametric_probability = cal_logits.softmax(1).cpu()
    selected = balanced_memory_indices(train.hard_label, train.event,
                                       args.capacity_per_class, args.seed + 29)
    scores, availability = class_similarity_scores(
        cal_embedding, train_embedding[selected], train.hard_label[selected],
        calibration.event, train.event[selected], topk=args.topk,
        exclude_same_event=True, purity=train.purity[selected],
        minimum_purity=args.minimum_memory_purity, chunk_size=args.chunk_size,
    )
    retrieval_probability = probability_from_scores(scores, args.retrieval_temperature)
    parametric = classification_metrics(calibration.hard_label, parametric_probability)
    retrieval = classification_metrics(calibration.hard_label, retrieval_probability)
    parametric_event = event_metrics(calibration, parametric_probability)
    retrieval_event = event_metrics(calibration, retrieval_probability)
    nonnegative = sum(
        retrieval_event["events"][event]["present_class_macro_f1"] >=
        parametric_event["events"][event]["present_class_macro_f1"]
        for event in retrieval_event["events"]
    )
    return model.cpu(), {
        "variant": name, "contrast_weight": contrast_weight,
        "prototype_weight": prototype_weight,
        "parametric": parametric, "retrieval": retrieval,
        "parametric_event": parametric_event, "retrieval_event": retrieval_event,
        "event_nonnegative_vs_parametric": f"{nonnegative}/{len(retrieval_event['events'])}",
        "availability_lt_topk_fraction": (availability < args.topk).float().mean(0).tolist(),
        "memory_indices": selected,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("M1 cross-event metric encoder ablation")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--d0-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--retrieval-temperature", type=float, default=0.10)
    parser.add_argument("--capacity-per-class", type=int, default=2048)
    parser.add_argument("--minimum-memory-purity", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    d0 = json.loads(Path(args.d0_json).read_text(encoding="utf-8"))
    if d0.get("decision") != "ADMIT_CROSS_EVENT_METRIC_M1":
        raise RuntimeError("M1 refused: D0 found fatal label/coverage failure")
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration_cache = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    for cache, split in ((train_cache, "meta_train"), (calibration_cache, "meta_calibration")):
        if cache.get("split") != split or cache.get("meta_confirm_accessed", False) or cache.get("val450_accessed", False) or cache.get("test_accessed", False):
            raise RuntimeError("M1 received unexpected or forbidden cache")
    train, mean, std = prepare_cache(train_cache)
    calibration, _, _ = prepare_cache(calibration_cache, mean=mean, std=std)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    results, models = {}, {}
    for name in VARIANTS:
        print(f"M1 starting {name}", flush=True)
        model, row = train_one(name, train, calibration, args, device)
        models[name] = model; results[name] = row
        checkpoint = {
            "protocol": PROTOCOL, "stage": "M1", "variant": name,
            "model_state_dict": model.state_dict(), "input_dim": train.x.shape[1],
            "hidden": args.hidden, "feature_mean": mean, "feature_std": std,
            "memory_indices": row["memory_indices"], "configuration": vars(args),
            "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
        }
        torch.save(checkpoint, output / f"{name}.pth")
        del row["memory_indices"]
    baseline = results["m1_ce"]["retrieval"]
    selected_name = max(
        results,
        key=lambda name: (results[name]["retrieval_event"]["event_macro_present_f1"],
                          results[name]["retrieval"]["macro_f1"]),
    )
    selected = results[selected_name]
    metric_gain = selected["retrieval"]["macro_f1"] - baseline["macro_f1"]
    parametric_gain = selected["retrieval"]["macro_f1"] - selected["parametric"]["macro_f1"]
    rare_ok = all(selected["retrieval"]["class_recall"][index] >=
                  selected["parametric"]["class_recall"][index] for index in (1, 2))
    nonnegative = int(selected["event_nonnegative_vs_parametric"].split("/")[0])
    checks = {
        "selected_uses_cross_event_objective": selected_name != "m1_ce",
        "metric_gain_vs_ce_retrieval_ge_0p01": metric_gain >= .01,
        "retrieval_gain_vs_own_parametric_ge_0p01": parametric_gain >= .01,
        "damaged_destroyed_recall_not_below_parametric": rare_ok,
        "event_nonnegative_ge_10": nonnegative >= min(10, len(selected["retrieval_event"]["events"])),
    }
    decision = "ADMIT_HIERARCHICAL_MEMORY_M2" if all(checks.values()) else "STOP_CROSS_EVENT_METRIC_M1"
    report = {
        "protocol": PROTOCOL, "stage": "M1", "configuration": vars(args),
        "variants": results, "selected_variant": selected_name,
        "selected_checkpoint": str(output / f"{selected_name}.pth"),
        "metric_gain_vs_ce_retrieval": metric_gain,
        "retrieval_gain_vs_own_parametric": parametric_gain,
        "checks": checks, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    report_path = output / "m1_screen.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_variant": selected_name,
                      "metric_gain_vs_ce_retrieval": metric_gain,
                      "retrieval_gain_vs_own_parametric": parametric_gain,
                      "checks": checks, "decision": decision}, indent=2), flush=True)
    print(f"Saved: {report_path}", flush=True)
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
