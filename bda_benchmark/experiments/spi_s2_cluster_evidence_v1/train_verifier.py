from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_s2_evidence_v2.audit_evidence import average_precision
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from . import PROTOCOL
from .audit_e0 import utility_labels


class ClusterUtilityVerifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(cache: dict, groups: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([cache["features"][name].float() for name in groups], dim=1)


def _event_rows(cache: dict) -> list[str]:
    return [cache["image_events"][int(index)] for index in cache["image_index"].long()]


def _weights(cache: dict, valid: torch.Tensor) -> torch.Tensor:
    events = [event for event, keep in zip(_event_rows(cache), valid.tolist()) if keep]
    counts = Counter(events)
    event_weight = torch.tensor([1.0 / counts[event] for event in events])
    changed = cache["utility"]["changed"].float()[valid].sqrt().clamp(1.0, 8.0)
    weights = event_weight * changed
    return weights / weights.mean().clamp_min(1e-8)


def _selective_histogram(cache: dict, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    accepted = scores >= threshold
    base = cache["p3_histograms"].sum(0).double()
    delta = cache["delta_hist"][accepted].sum(0).double() if bool(accepted.any()) else torch.zeros_like(base)
    return base + delta


def _event_nonnegative(cache: dict, scores: torch.Tensor, threshold: float) -> tuple[int, int, dict]:
    rows = {}
    for event in sorted(set(cache["image_events"])):
        image_ids = torch.tensor([i for i, value in enumerate(cache["image_events"]) if value == event])
        cluster_mask = torch.isin(cache["image_index"].long(), image_ids)
        base = cache["p3_histograms"][image_ids].sum(0).double()
        accepted = cluster_mask & (scores >= threshold)
        selective = base + (
            cache["delta_hist"][accepted].sum(0).double()
            if bool(accepted.any()) else torch.zeros_like(base)
        )
        fixed = cache["fixed_s2_histograms"][image_ids].sum(0).double()
        selective_report = metrics_from_histogram(selective)
        fixed_report = metrics_from_histogram(fixed)
        gain = 100.0 * (selective_report["valid_miou"] - fixed_report["valid_miou"])
        rows[event] = {"gain_vs_fixed_s2_miou_pp": gain}
    return sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in rows.values()), len(rows), rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train one low-capacity cluster utility verifier")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--admission-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.10)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--maximum-reject-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    admission = json.loads(Path(args.admission_json).read_text(encoding="utf-8"))
    if admission.get("decision") != "ADMIT_CLUSTER_E1_SINGLE_VERIFIER":
        raise RuntimeError("Cluster-E1 refused: parameter-free Cluster-E0 did not pass")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    if train.get("protocol") != PROTOCOL or calibration.get("protocol") != PROTOCOL:
        raise RuntimeError("unexpected cluster evidence cache protocol")
    for name, cache in (("train", train), ("calibration", calibration)):
        if cache.get("val450_accessed", True) or cache.get("test_accessed", True):
            raise RuntimeError(f"{name} cache accessed a forbidden split")
        if cache.get("meta_confirm_accessed", True):
            raise RuntimeError(f"{name} cache accessed meta_confirm")
    if train["p3_sha256"] != calibration["p3_sha256"]:
        raise RuntimeError("train/calibration P3 fingerprints differ")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("train/calibration images overlap")
    if admission.get("train_cache_sha256") != _sha256(args.train_cache):
        raise RuntimeError("Cluster-E0/train cache fingerprint mismatch")
    if admission.get("calibration_cache_sha256") != _sha256(args.calibration_cache):
        raise RuntimeError("Cluster-E0/calibration cache fingerprint mismatch")
    admitted = tuple(admission.get("admitted_groups", ()))
    groups = ("spatial", *admitted)
    train_labels_all = utility_labels(train, args.utility_margin)
    calibration_labels_all = utility_labels(calibration, args.utility_margin)
    train_valid = train_labels_all >= 0
    calibration_valid = calibration_labels_all >= 0
    train_matrix = _matrix(train, groups)
    calibration_matrix = _matrix(calibration, groups)
    mean = train_matrix[train_valid].mean(0)
    std = train_matrix[train_valid].std(0, unbiased=False).clamp_min(1e-4)
    train_features = ((train_matrix[train_valid] - mean) / std).clamp(-8, 8)
    calibration_features = ((calibration_matrix - mean) / std).clamp(-8, 8)
    train_labels = train_labels_all[train_valid].float()
    weights = _weights(train, train_valid)

    torch.manual_seed(args.seed)
    model = ClusterUtilityVerifier(train_features.shape[1], args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    positives = train_labels.sum().clamp_min(1.0)
    negatives = (train_labels == 0).sum().float()
    pos_weight = (negatives / positives).detach()
    generator = torch.Generator().manual_seed(args.seed + 1)
    batch_size = min(args.batch_size, len(train_features))
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randint(
            len(train_features), (batch_size,), generator=generator
        )
        logits = model(train_features[indices])
        raw = F.binary_cross_entropy_with_logits(
            logits, train_labels[indices], reduction="none", pos_weight=pos_weight
        )
        loss = (raw * weights[indices]).mean()
        optimizer.zero_grad()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"optimizer_step": step, "loss": float(loss), "gradient_norm": float(gradient)}), flush=True)

    model.eval()
    with torch.no_grad():
        calibration_scores = model(calibration_features).sigmoid()
    candidates = []
    for threshold in torch.linspace(0.05, 0.50, 46):
        reject_fraction = float((calibration_scores < threshold).float().mean())
        if reject_fraction > args.maximum_reject_fraction:
            continue
        report = metrics_from_histogram(
            _selective_histogram(calibration, calibration_scores, float(threshold))
        )
        candidates.append(
            {
                "threshold": float(threshold), "reject_fraction": reject_fraction,
                "miou": report["valid_miou"], "damaged_f1": report["damaged_f1"],
            }
        )
    if not candidates:
        raise RuntimeError("no safe rejection threshold satisfied the reject-fraction cap")
    selected = max(candidates, key=lambda row: (row["miou"], row["damaged_f1"], -row["reject_fraction"]))
    fixed = metrics_from_histogram(calibration["fixed_s2_histograms"].sum(0))
    p3 = metrics_from_histogram(calibration["p3_histograms"].sum(0))
    gain_pp = 100.0 * (selected["miou"] - fixed["valid_miou"])
    nonnegative, event_total, event_rows = _event_nonnegative(
        calibration, calibration_scores, selected["threshold"]
    )
    labels = calibration_labels_all[calibration_valid]
    scores = calibration_scores[calibration_valid]
    checks = {
        "gain_vs_fixed_s2_ge_0_30pp": gain_pp >= 0.30,
        "damaged_f1_not_below_fixed_s2": selected["damaged_f1"] >= fixed["damaged_f1"],
        "nonnegative_events_ge_10": nonnegative >= min(10, event_total),
        "reject_fraction_within_cap": selected["reject_fraction"] <= args.maximum_reject_fraction,
    }
    decision = (
        "FREEZE_CLUSTER_E1_FOR_META_CONFIRM"
        if all(checks.values()) else "STOP_CLUSTER_E1_AFTER_CALIBRATION_SCREEN"
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "protocol": "SPI_S2_CLUSTER_UTILITY_VERIFIER_E1_v1",
        "model_state_dict": model.state_dict(),
        "input_dim": train_features.shape[1], "hidden": args.hidden,
        "groups": groups, "feature_mean": mean, "feature_std": std,
        "threshold": selected["threshold"], "maximum_reject_fraction": args.maximum_reject_fraction,
        "p3_sha256": train["p3_sha256"], "seed": args.seed,
        "admission_sha256": _sha256(args.admission_json),
        "train_cache_sha256": _sha256(args.train_cache),
        "calibration_cache_sha256": _sha256(args.calibration_cache),
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    torch.save(checkpoint, output / "verifier_frozen.pth")
    report = {
        "protocol": "SPI_S2_CLUSTER_UTILITY_VERIFIER_E1_CALIBRATION_v1",
        "configuration": vars(args), "groups": groups,
        "calibration_auroc": binary_auroc(labels, scores),
        "calibration_average_precision": average_precision(labels, scores),
        "p3": p3, "fixed_s2": fixed, "selected": selected,
        "gain_vs_fixed_s2_miou_pp": gain_pp,
        "event_nonnegative": nonnegative, "event_total": event_total,
        "event_rows": event_rows, "checks": checks, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    (output / "calibration_screen.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "groups": groups, "selected": selected, "gain_miou_pp": gain_pp,
        "event_nonnegative": f"{nonnegative}/{event_total}", "checks": checks,
        "decision": decision,
    }, indent=2), flush=True)
    print(f"Saved: {output / 'verifier_frozen.pth'}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
