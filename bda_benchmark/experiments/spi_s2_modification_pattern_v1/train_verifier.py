from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from .audit_e0 import utility_labels


class ModificationPatternVerifier(nn.Module):
    def __init__(self, inputs: int, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(cache: dict, groups: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([cache["features"][name].float() for name in groups], dim=1)


def _holdout(cache: dict, seed: int) -> torch.Tensor:
    output = []
    for image_index in cache["image_index"].long():
        name = cache["image_names"][int(image_index)]
        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
        output.append(int.from_bytes(digest[:4], "little") % 5 == 0)
    return torch.tensor(output, dtype=torch.bool)


def _weights(cache: dict, valid: torch.Tensor) -> torch.Tensor:
    image_indices = cache["image_index"].long()[valid]
    events = [cache["image_events"][int(index)] for index in image_indices]
    counts = Counter(events)
    utility = cache["utility"]["utility_ratio"].float()[valid].abs().clamp_min(0.10).sqrt()
    weights = torch.tensor([1.0 / counts[event] for event in events]) * utility
    return weights / weights.mean()


def _holdout_hist(cache: dict, holdout: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    holdout_rows = torch.nonzero(holdout, as_tuple=False).flatten()
    image_ids = torch.unique(cache["image_index"].long()[holdout_rows])
    base = cache["p3_histograms"][image_ids].sum(0).double()
    accepted_rows = holdout_rows[scores[holdout] >= threshold]
    return base + cache["delta_hist"][accepted_rows].sum(0).double()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train one low-capacity S2 modification-pattern verifier")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--admission-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.10)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    admission = json.loads(Path(args.admission_json).read_text(encoding="utf-8"))
    if admission.get("decision") != "ADMIT_SMP_E1_SINGLE_VERIFIER":
        raise RuntimeError("SMP-E1 refused: parameter-free E0 did not pass")
    cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    if admission.get("train_cache_sha256") != _sha256(args.train_cache):
        raise RuntimeError("E0/train cache fingerprint mismatch")
    admitted = tuple(admission.get("admitted_groups", ()))
    if not admitted:
        raise RuntimeError("admission contains no spatial group")
    groups = ("baseline", *admitted)
    labels_all = utility_labels(cache, args.utility_margin)
    valid = labels_all >= 0
    feature_matrix = _matrix(cache, groups)
    holdout_all = _holdout(cache, args.seed)
    fit_all = valid & ~holdout_all
    if min(int((valid & holdout_all).sum()), int(fit_all.sum())) == 0:
        raise RuntimeError("deterministic meta_train holdout is empty")
    mean = feature_matrix[fit_all].mean(0)
    std = feature_matrix[fit_all].std(0, unbiased=False).clamp_min(1e-4)
    features_full = ((feature_matrix - mean) / std).clamp(-8, 8)
    features = features_full[valid]
    labels = labels_all[valid].float()
    holdout = holdout_all[valid]
    fit = ~holdout
    weights = _weights(cache, valid)
    torch.manual_seed(args.seed)
    model = ModificationPatternVerifier(features.shape[1], args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    positives = labels[fit].sum().clamp_min(1)
    negatives = (labels[fit] == 0).sum()
    pos_weight = (negatives / positives).detach()
    for step in range(1, args.steps + 1):
        model.train()
        logits = model(features[fit])
        raw = F.binary_cross_entropy_with_logits(logits, labels[fit], reduction="none", pos_weight=pos_weight)
        loss = (raw * weights[fit]).mean()
        optimizer.zero_grad(); loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"optimizer_step": step, "loss": float(loss), "gradient_norm": float(gradient)}), flush=True)
    model.eval()
    with torch.no_grad():
        all_scores = model(features_full).sigmoid()
    thresholds = torch.linspace(0.05, 0.95, 91)
    rows = []
    for threshold in thresholds:
        report = metrics_from_histogram(_holdout_hist(cache, holdout_all, all_scores, float(threshold)))
        rows.append({"threshold": float(threshold), "miou": report["valid_miou"], "damaged_f1": report["damaged_f1"],
                     "accepted_fraction": float((all_scores[holdout_all] >= threshold).float().mean())})
    selected = max(rows, key=lambda row: (row["miou"], row["damaged_f1"], -abs(row["threshold"] - 0.5)))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "protocol": "SPI_S2_MODIFICATION_PATTERN_E1_v1", "model_state_dict": model.state_dict(),
        "input_dim": features.shape[1], "hidden": args.hidden, "groups": groups,
        "mean": mean, "std": std, "threshold": selected["threshold"],
        "threshold_source": "deterministic_20pct_meta_train_image_holdout",
        "train_cache_sha256": _sha256(args.train_cache), "seed": args.seed,
        "meta_calibration_used_for_training": False, "meta_confirm_accessed": False,
        "val450_accessed": False, "official_instances_accessed": False, "test_accessed": False,
    }
    torch.save(checkpoint, output / "verifier_frozen.pth")
    result = {"protocol": checkpoint["protocol"], "groups": groups, "selected": selected,
              "threshold_table": rows, "decision": "FREEZE_SMP_E1_AWAIT_META_CONFIRM",
              "meta_confirm_accessed": False, "val450_accessed": False,
              "official_instances_accessed": False, "test_accessed": False}
    (output / "train_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": groups, "selected": selected, "decision": result["decision"]}, indent=2))
    print(f"Saved: {output / 'verifier_frozen.pth'}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
