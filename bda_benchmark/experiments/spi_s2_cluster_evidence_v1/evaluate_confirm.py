from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from . import PROTOCOL
from .train_verifier import ClusterUtilityVerifier, _matrix


def _per_image_histograms(
    cache: dict, scores: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    p3 = cache["p3_histograms"].double()
    fixed = cache["fixed_s2_histograms"].double()
    selective = p3.clone()
    accepted = scores >= threshold
    for image_index in torch.unique(cache["image_index"].long()).tolist():
        mask = accepted & (cache["image_index"].long() == image_index)
        if bool(mask.any()):
            selective[image_index] += cache["delta_hist"][mask].sum(0).double()
    return fixed, selective


def _event_rows(cache: dict, fixed: torch.Tensor, selective: torch.Tensor) -> dict:
    rows = {}
    for event in sorted(set(cache["image_events"])):
        ids = torch.tensor(
            [index for index, value in enumerate(cache["image_events"]) if value == event]
        )
        fixed_report = metrics_from_histogram(fixed[ids].sum(0))
        selective_report = metrics_from_histogram(selective[ids].sum(0))
        rows[event] = {
            "fixed_s2_miou": fixed_report["valid_miou"],
            "selective_miou": selective_report["valid_miou"],
            "gain_miou_pp": 100.0
            * (selective_report["valid_miou"] - fixed_report["valid_miou"]),
        }
    return rows


def _bootstrap(
    fixed: torch.Tensor,
    selective: torch.Tensor,
    iterations: int,
    seed: int,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    count = fixed.shape[0]
    gains = []
    for _ in range(iterations):
        indices = torch.randint(count, (count,), generator=generator)
        fixed_report = metrics_from_histogram(fixed[indices].sum(0))
        selective_report = metrics_from_histogram(selective[indices].sum(0))
        gains.append(
            100.0
            * (selective_report["valid_miou"] - fixed_report["valid_miou"])
        )
    values = np.asarray(gains, dtype=np.float64)
    return {
        "mean_pp": float(values.mean()),
        "ci95_low_pp": float(np.quantile(values, 0.025)),
        "ci95_high_pp": float(np.quantile(values, 0.975)),
        "iterations": int(iterations),
    }


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("One-time frozen Cluster-E1 confirmation")
    parser.add_argument("--confirm-cache", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--calibration-screen", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.30)
    parser.add_argument("--minimum-nonnegative-events", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    screen = json.loads(Path(args.calibration_screen).read_text(encoding="utf-8"))
    if screen.get("decision") != "FREEZE_CLUSTER_E1_FOR_META_CONFIRM":
        raise RuntimeError("confirm refused: Cluster-E1 did not pass calibration screening")
    cache = torch.load(args.confirm_cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.verifier_checkpoint, map_location="cpu", weights_only=False)
    if cache.get("protocol") != PROTOCOL or cache.get("split") != "meta_confirm":
        raise RuntimeError("unexpected meta_confirm cache")
    if not cache.get("meta_confirm_accessed", False):
        raise RuntimeError("meta_confirm cache provenance is incomplete")
    if cache.get("val450_accessed", True) or cache.get("test_accessed", True):
        raise RuntimeError("forbidden split was accessed")
    if checkpoint.get("protocol") != "SPI_S2_CLUSTER_UTILITY_VERIFIER_E1_v1":
        raise RuntimeError("unexpected verifier protocol")
    if checkpoint["p3_sha256"] != cache["p3_sha256"]:
        raise RuntimeError("P3 fingerprint differs between training and confirm")

    groups = tuple(checkpoint["groups"])
    matrix = _matrix(cache, groups)
    features = (
        (matrix - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    ).clamp(-8, 8)
    model = ClusterUtilityVerifier(checkpoint["input_dim"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scores = model(features).sigmoid()
    threshold = float(checkpoint["threshold"])
    fixed_by_image, selective_by_image = _per_image_histograms(cache, scores, threshold)
    p3 = metrics_from_histogram(cache["p3_histograms"].sum(0))
    fixed = metrics_from_histogram(fixed_by_image.sum(0))
    selective = metrics_from_histogram(selective_by_image.sum(0))
    gain_pp = 100.0 * (selective["valid_miou"] - fixed["valid_miou"])
    events = _event_rows(cache, fixed_by_image, selective_by_image)
    nonnegative = sum(row["gain_miou_pp"] >= 0 for row in events.values())
    bootstrap = _bootstrap(
        fixed_by_image, selective_by_image, args.bootstrap_iterations, args.seed
    )
    checks = {
        "gain_vs_fixed_s2_ge_threshold": gain_pp >= args.minimum_gain_pp,
        "bootstrap_ci_low_gt_zero": bootstrap["ci95_low_pp"] > 0,
        "nonnegative_events_ge_threshold": nonnegative
        >= min(args.minimum_nonnegative_events, len(events)),
        "damaged_f1_not_below_fixed_s2": selective["damaged_f1"]
        >= fixed["damaged_f1"],
    }
    decision = (
        "ADMIT_CLUSTER_E1_TO_VAL450"
        if all(checks.values())
        else "STOP_CLUSTER_E1_AFTER_META_CONFIRM"
    )
    payload = {
        "protocol": "SPI_S2_CLUSTER_UTILITY_VERIFIER_E1_CONFIRM_v1",
        "configuration": vars(args),
        "groups": groups,
        "threshold": threshold,
        "accepted_fraction": float((scores >= threshold).float().mean()),
        "p3": p3,
        "fixed_s2": fixed,
        "selective": selective,
        "gain_vs_fixed_s2_miou_pp": gain_pp,
        "events": events,
        "nonnegative_events": nonnegative,
        "event_total": len(events),
        "bootstrap": bootstrap,
        "checks": checks,
        "decision": decision,
        "meta_confirm_accessed": True,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing repeated meta_confirm evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "gain_miou_pp": gain_pp,
        "event_nonnegative": f"{nonnegative}/{len(events)}",
        "bootstrap": bootstrap,
        "checks": checks,
        "decision": decision,
    }, indent=2), flush=True)
    print(f"Saved: {output}")
    print("Val450/official instances/test accessed: NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
