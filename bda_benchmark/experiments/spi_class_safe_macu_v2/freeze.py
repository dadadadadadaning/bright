from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import (
    per_image_selective_histograms,
)
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier import (
    matrix, sha256,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from . import CHECKPOINT_PROTOCOL, LOEO_PROTOCOL
from .core import (
    apply_thresholds, choose_thresholds, dual_targets, frozen_gradients,
    predict, train_dual_linear,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Freeze final CS-MACU V2 after successful LOEO")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--loeo-json", required=True)
    parser.add_argument("--d0-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--maximum-reject-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "verifier_frozen.pth"
    screen_path = output_dir / "development_screen.json"
    if checkpoint_path.exists() or screen_path.exists():
        raise RuntimeError("refusing to overwrite a frozen CS-MACU V2 model")
    loeo = json.loads(Path(args.loeo_json).read_text(encoding="utf-8"))
    if loeo.get("protocol") != LOEO_PROTOCOL or loeo.get("decision") != "FREEZE_CS_MACU_V2_FOR_VAL450":
        raise RuntimeError("final freeze refused: LOEO did not pass")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    groups = tuple(loeo["groups"])
    raw_train = matrix(train, groups, "meta_train")
    raw_calibration = matrix(calibration, groups, "meta_calibration")
    train_images = torch.ones(len(train["image_names"]), dtype=torch.bool)
    train_clusters = torch.ones(len(train["image_index"]), dtype=torch.bool)
    cal_images = torch.ones(len(calibration["image_names"]), dtype=torch.bool)
    cal_clusters = torch.ones(len(calibration["image_index"]), dtype=torch.bool)
    metric_gradient, damaged_gradient = frozen_gradients(train, train_images)
    metric_label, harm_label, metric_utility, damaged_utility = dual_targets(
        train, metric_gradient, damaged_gradient
    )
    mean = raw_train.mean(0)
    std = raw_train.std(0, unbiased=False).clamp_min(1e-4)
    train_features = ((raw_train - mean) / std).clamp(-8, 8)
    cal_features = ((raw_calibration - mean) / std).clamp(-8, 8)
    model = train_dual_linear(
        train_features, train, train_clusters,
        metric_label, harm_label, metric_utility, damaged_utility,
        steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed,
    )
    cal_metric_score, cal_harm_score = predict(model, cal_features)
    policy = choose_thresholds(
        calibration, cal_metric_score, cal_harm_score,
        cal_clusters, cal_images,
        maximum_reject_fraction=args.maximum_reject_fraction,
    )
    accept = apply_thresholds(cal_metric_score, cal_harm_score, policy)
    selected_by_image = per_image_selective_histograms(calibration, accept)
    fixed = metrics_from_histogram(calibration["fixed_s2_histograms"].sum(0))
    selected = metrics_from_histogram(selected_by_image.sum(0))
    development_gain_pp = 100.0 * (
        selected["valid_miou"] - fixed["valid_miou"]
    )
    damaged_change_pp = 100.0 * (
        selected["damaged_f1"] - fixed["damaged_f1"]
    )
    if development_gain_pp <= 0 or damaged_change_pp < 0 or bool(accept.all()):
        raise RuntimeError(
            "refusing final freeze: full-development policy is ineffective, "
            "Damaged-unsafe, or identical to fixed S2"
        )
    screen = {
        "protocol": "SPI_CLASS_SAFE_MACU_V2_DEVELOPMENT_SCREEN_v1",
        "configuration": vars(args),
        "policy": policy,
        "fixed_s2": fixed,
        "selected": selected,
        "gain_vs_fixed_s2_miou_pp": development_gain_pp,
        "damaged_f1_change_pp": damaged_change_pp,
        "loeo_sha256": sha256(args.loeo_json),
        "meta_confirm_used": False,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    checkpoint = {
        "protocol": CHECKPOINT_PROTOCOL,
        "model_state_dict": model.state_dict(),
        "input_dim": raw_train.shape[1],
        "groups": groups,
        "feature_mean": mean,
        "feature_std": std,
        "metric_threshold": policy["metric_threshold"],
        "harm_threshold": policy["harm_threshold"],
        "train_frozen_miou_gradient": metric_gradient,
        "train_frozen_damaged_f1_gradient": damaged_gradient,
        "p3_sha256": train["p3_sha256"],
        "train_cache_sha256": sha256(args.train_cache),
        "calibration_cache_sha256": sha256(args.calibration_cache),
        "loeo_sha256": sha256(args.loeo_json),
        "d0_sha256": sha256(args.d0_json),
        "seed": args.seed,
        "meta_confirm_used": False,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    screen_path.write_text(json.dumps(screen, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "policy": policy,
        "gain_vs_fixed_s2_miou_pp": screen["gain_vs_fixed_s2_miou_pp"],
        "damaged_f1_change_pp": screen["damaged_f1_change_pp"],
        "decision": "FROZEN_CS_MACU_V2_AWAITING_VAL450",
    }, indent=2), flush=True)
    print(f"Saved: {checkpoint_path}")
    print("Meta-confirm/Val450/official instances/test used: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
