#!/usr/bin/env python3
"""Frozen P5 audit: accuracy, event errors, perturbations, and internal evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BDA_ROOT = REPO_ROOT / "bda_benchmark"
for value in (str(REPO_ROOT), str(BDA_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from bda_benchmark.dataset.make_data_loader_p4_native_sar import build_p4_dataset
from bda_benchmark.model.DamageFormer_S8_P5_DCMF import (
    DamageFormer_S8_P5_DCMF,
    FrozenSARFilterBank,
    get_p5_variant_spec,
)
from bda_benchmark.model.DamageFormer_S8_P5_Hierarchical import (
    DamageFormer_S8_P5_Hierarchical,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.experiments.p5_ablation_validation_v1.metrics import (
    aggregate_by_event,
    event_from_name,
    metrics_from_confusion,
)


MODEL_KWARGS = dict(
    roi_alpha=0.1,
    detach_building_prior=True,
    token_dim=128,
    mil_topk=8,
    max_tokens_per_bag=256,
    mil_residual_scale=0.0,
    use_asymmetric_regions=False,
)


def fast_hist(prediction: torch.Tensor, target: torch.Tensor, classes: int = 4,
              ignore_index: int = 255) -> torch.Tensor:
    valid = (target != ignore_index) & (target >= 0) & (target < classes)
    encoded = classes * target[valid].long() + prediction[valid].long()
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def shift_zero(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate SAR without circular wrap-around."""
    output = torch.zeros_like(x)
    height, width = x.shape[-2:]
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
    dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        output[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    return output


def perturb_sar(sar: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return sar
    if mode.startswith("shift_x"):
        return shift_zero(sar, int(mode.removeprefix("shift_x")), 0)
    if mode.startswith("shift_y"):
        return shift_zero(sar, 0, int(mode.removeprefix("shift_y")))
    if mode == "pair_shuffle":
        # A wrong RGB-SAR pair is the causal control. Batch size one gets a
        # deterministic spatial reversal and is marked in the output contract.
        return sar.roll(1, 0) if sar.shape[0] > 1 else sar.flip((-2, -1))
    raise ValueError(f"unknown perturbation {mode!r}")


def _tensor_stats(value: torch.Tensor) -> Dict[str, float]:
    data = value.detach().float()
    return {
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "abs_mean": float(data.abs().mean().item()),
    }


def collect_aux(aux: Mapping[str, object], target: torch.Tensor) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key in (
        "p_building", "roi_weight", "rgb_feature", "sar_feature", "fused_roi",
        "joint_rgb", "joint_sar", "occurrence_logits", "severity_logits",
        "occurrence_delta", "severity_delta", "occurrence_transition",
        "severity_transition",
    ):
        value = aux.get(key)
        if isinstance(value, torch.Tensor):
            for stat_name, stat_value in _tensor_stats(value).items():
                output[f"{key}.{stat_name}"] = stat_value
    rgb, sar = aux.get("rgb_feature"), aux.get("sar_feature")
    if isinstance(rgb, torch.Tensor) and isinstance(sar, torch.Tensor):
        output["rgb_sar_cosine"] = float(F.cosine_similarity(
            rgb.float().flatten(2), sar.float().flatten(2), dim=1
        ).mean().item())
        output["rgb_sar_abs_difference"] = float((rgb.float() - sar.float()).abs().mean().item())
    p_building = aux.get("p_building")
    if isinstance(p_building, torch.Tensor):
        small_target = F.interpolate((target > 0).float().unsqueeze(1),
                                     p_building.shape[-2:], mode="nearest").bool()
        if small_target.any():
            output["p_building.on_building"] = float(p_building[small_target].mean().item())
        if (~small_target).any():
            output["p_building.on_background"] = float(p_building[~small_target].mean().item())
    return output


def merge_means(records):
    values: Dict[str, list] = {}
    for record in records:
        for key, value in record.items():
            values.setdefault(key, []).append(float(value))
    return {key: float(np.mean(items)) for key, items in sorted(values.items())}


def save_frequency_audit(sar: torch.Tensor, output: Path, filtered: bool) -> None:
    """Save the four views actually consumed by the C0/C1+ SAR stem."""
    source = sar[:1].detach().cpu()
    views = FrozenSARFilterBank()(source)[0] if filtered else source.repeat(1, 4, 1, 1)[0]
    spectra = []
    for view in views:
        spectrum = torch.log1p(torch.fft.fftshift(torch.fft.fft2(view.float())).abs())
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min()).clamp_min(1e-6)
        spectra.append((255.0 * spectrum).byte().numpy())
    canvas = np.concatenate(spectra, axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(canvas, mode="L").save(output)
    except ImportError:
        np.save(output.with_suffix(".npy"), canvas)


def sar_view_diagnostics(sar: torch.Tensor, filtered: bool) -> Dict[str, float]:
    source = sar.detach().float()
    views = FrozenSARFilterBank().to(source.device)(source) if filtered else source.repeat(1, 4, 1, 1)
    return {
        "view_diversity_abs": float((views[:, 1:] - views[:, :1]).abs().mean().item()),
        "high_frequency_energy": float(views[:, 2].square().mean().sqrt().item()),
        "gradient_energy": float(views[:, 3].square().mean().sqrt().item()),
    }


def gradient_probe(model, pre, sar, target, ignore_index: int) -> Dict[str, float]:
    model.zero_grad(set_to_none=True)
    _, damage = model(pre, sar)
    F.cross_entropy(damage.float(), target, ignore_index=ignore_index).backward()
    groups: Dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        prefix = name.split(".", 1)[0]
        groups[prefix] = groups.get(prefix, 0.0) + float(parameter.grad.float().norm().item() ** 2)
    model.zero_grad(set_to_none=True)
    return {key: value ** 0.5 for key, value in sorted(groups.items())}


def build_model(checkpoint: Mapping[str, object], architecture: str):
    saved_args = checkpoint.get("args", {})
    variant = str(checkpoint.get("variant", saved_args.get("variant", "p3")))
    if architecture == "p3":
        model = DamageFormer_S8_MICB_DSMIL(**MODEL_KWARGS)
        variant = "p3"
    else:
        model_type = (DamageFormer_S8_P5_Hierarchical
                      if architecture == "hierarchical" else DamageFormer_S8_P5_DCMF)
        model = model_type(variant=variant, **MODEL_KWARGS)
    state = checkpoint.get(
        "model", checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    )
    if state is None and checkpoint and all(isinstance(value, torch.Tensor)
                                            for value in checkpoint.values()):
        state = checkpoint
    if not isinstance(state, Mapping):
        raise RuntimeError("checkpoint must contain model/state_dict")
    cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return model, variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Frozen P5 mechanism audit")
    parser.add_argument("--architecture", choices=("p3", "flat", "hierarchical"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--perturbation", choices=(
        "none", "shift_x4", "shift_x8", "shift_y4", "shift_y8", "pair_shuffle"
    ), default="none")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--frequency_png", default="")
    parser.add_argument("--gradient_probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test":
        raise RuntimeError("This audit is validation-only; official test remains locked.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("checkpoint is not a mapping")
    model, variant = build_model(checkpoint, args.architecture)
    filtered_sar = (
        args.architecture != "p3"
        and get_p5_variant_spec(variant).filter_augmentation
    )
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = build_p4_dataset(args.dataset_path, args.data_list_path, 1024, "val", "a2",
                               args.sar_stats_json)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    rows, internal_flat_rows, aux_records, sar_view_records = [], [], [], []
    total_hist = torch.zeros(4, 4, dtype=torch.int64)
    gradient_norms = None
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            pre = batch["pre_img"].to(device).float()
            sar_key = "teacher_post_img" if args.architecture == "p3" else "post_img"
            sar = batch[sar_key].to(device).float()
            target = batch["label_dam"].to(device).long()
            names = batch["name"]
            if batch_index == 0 and args.frequency_png:
                save_frequency_audit(
                    sar, Path(args.frequency_png), filtered=filtered_sar
                )
            if args.architecture != "p3":
                sar_view_records.append(
                    sar_view_diagnostics(sar, filtered=filtered_sar)
                )
            tested_sar = perturb_sar(sar, args.perturbation)
            _, logits, aux = model(pre, tested_sar, return_aux=True)
            prediction = logits.argmax(1)
            flat_logits = aux.get("flat_damage_logits")
            flat_prediction = None
            if isinstance(flat_logits, torch.Tensor):
                flat_prediction = F.interpolate(
                    flat_logits, target.shape[-2:], mode="bilinear", align_corners=False
                ).argmax(1)
            aux_records.append(collect_aux(aux, target))
            for sample_index, name in enumerate(names):
                hist = fast_hist(prediction[sample_index], target[sample_index],
                                 ignore_index=args.ignore_index).cpu()
                total_hist += hist
                rows.append({
                    "name": str(name),
                    "event": event_from_name(str(name)),
                    "confusion": hist.tolist(),
                })
                if flat_prediction is not None:
                    flat_hist = fast_hist(flat_prediction[sample_index], target[sample_index],
                                          ignore_index=args.ignore_index).cpu()
                    internal_flat_rows.append({
                        "name": str(name),
                        "event": event_from_name(str(name)),
                        "confusion": flat_hist.tolist(),
                    })
    if args.gradient_probe:
        batch = next(iter(loader))
        pre = batch["pre_img"].to(device).float()
        sar_key = "teacher_post_img" if args.architecture == "p3" else "post_img"
        sar = perturb_sar(batch[sar_key].to(device).float(), args.perturbation)
        target = batch["label_dam"].to(device).long()
        gradient_norms = gradient_probe(model, pre, sar, target, args.ignore_index)
    payload = {
        "protocol": "P5_ABLATION_MECHANISM_AUDIT_v1",
        "architecture": args.architecture,
        "variant": variant,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split": args.split,
        "test_accessed": False,
        "perturbation": args.perturbation,
        "pair_shuffle_batch1_fallback": args.perturbation == "pair_shuffle" and args.batch_size == 1,
        "images": len(rows),
        "confusion": total_hist.tolist(),
        "metrics": metrics_from_confusion(total_hist.numpy()),
        "events": aggregate_by_event(rows),
        "image_rows": rows,
        "feature_statistics": merge_means(aux_records),
        "gradient_norms": gradient_norms,
        "sar_view_diagnostics": merge_means(sar_view_records),
        "internal_flat": None if not internal_flat_rows else {
            "metrics": metrics_from_confusion(np.sum(
                [row["confusion"] for row in internal_flat_rows], axis=0
            )),
            "events": aggregate_by_event(internal_flat_rows),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("events", "image_rows")}, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
