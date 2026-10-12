# -*- coding: utf-8 -*-
"""Damage-conditioned feature-predictive pretraining for the A2 SAR pathway.

This is an intermediate train-split-only stage.  Its single objective predicts
masked stage-3/4 SAR features.  RGB features condition the target only on
background/intact pixels; damaged/destroyed pixels retain the EMA SAR target.
The predictor and EMA target are discarded after pretraining.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
    read_data_list,
)
from bda_benchmark.model.DamageFormer_RGBSAR_A2_DCSM import (  # noqa: E402
    A2_DCSM_VARIANTS,
    DCSMFeaturePredictor,
    DamageFormerRGBSARA2DCSM,
    StructuralSARAdapter,
    class_balanced_region_weights,
    damage_conditioned_target,
    masked_cosine_prediction_loss,
    unchanged_mask_from_damage,
)
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import seed_worker  # noqa: E402


DEEP_STAGE_INDICES = (2, 3)
DEEP_LATENT_CHANNELS = (128, 160)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(payload: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def checkpoint_state(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("A2 checkpoint does not contain a model state")
    return state  # type: ignore[return-value]


def load_locked_a2(
    model: DamageFormerRGBSARA2DCSM, path: str
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError(
            "DCSM pretraining requires the locked H6-A2 checkpoint; "
            f"received variant={checkpoint.get('variant')!r}"
        )
    missing, unexpected = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    allowed_missing = {
        key for key in missing if key.startswith("structural_adapter.")
    }
    illegal_missing = sorted(set(missing) - allowed_missing)
    if unexpected or illegal_missing:
        raise RuntimeError(
            "H6-A2 to DCSM migration is not strict: "
            f"missing={illegal_missing}, unexpected={unexpected}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "variant": checkpoint.get("variant"),
        "step": int(checkpoint.get("step", 0)),
        "metrics": dict(
            checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
        ),
        "allowed_new_keys": sorted(allowed_missing),
    }


def event_name(sample_name: str) -> str:
    return sample_name.rsplit("_", 1)[0]


def event_balanced_sampler(names: Sequence[str], num_samples: int) -> WeightedRandomSampler:
    counts = Counter(event_name(name) for name in names)
    weights = [1.0 / counts[event_name(name)] for name in names]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
    )


def random_patch_mask(
    batch_size: int,
    height: int,
    width: int,
    patch_size: int,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    rows = (height + patch_size - 1) // patch_size
    columns = (width + patch_size - 1) // patch_size
    count = rows * columns
    masked = max(1, min(count - 1, int(round(count * ratio))))
    scores = torch.rand(batch_size, count, device=device)
    indices = scores.topk(masked, dim=1).indices
    coarse = torch.zeros(batch_size, count, device=device)
    coarse.scatter_(1, indices, 1.0)
    coarse = coarse.reshape(batch_size, 1, rows, columns)
    return torch.nn.functional.interpolate(
        coarse, size=(height, width), mode="nearest"
    )


class TargetSARPathway(nn.Module):
    """EMA target containing only modules that define the SAR representation."""

    def __init__(self, model: DamageFormerRGBSARA2DCSM) -> None:
        super().__init__()
        self.structural_adapter = copy.deepcopy(model.structural_adapter)
        self.sar_encoder = copy.deepcopy(model.sar_encoder)
        self.deep_projections = nn.ModuleList()
        for index in DEEP_STAGE_INDICES:
            stage = model.joint_stages[index]
            self.deep_projections.append(
                nn.ModuleDict(
                    {
                        "sar_common": copy.deepcopy(stage.sar_common),
                        "sar_unique": copy.deepcopy(stage.sar_unique),
                    }
                )
            )
        self.requires_grad_(False)

    def forward(self, sar: torch.Tensor):
        sar = self.structural_adapter(sar)
        stages = self.sar_encoder(sar)
        outputs = []
        for projection, index in zip(self.deep_projections, DEEP_STAGE_INDICES):
            outputs.append(
                (
                    projection["sar_common"](stages[index]),
                    projection["sar_unique"](stages[index]),
                )
            )
        return outputs


@torch.no_grad()
def ema_update(target: nn.Module, model: DamageFormerRGBSARA2DCSM, momentum: float) -> None:
    sources: Iterable[nn.Module] = [model.structural_adapter, model.sar_encoder]
    targets: Iterable[nn.Module] = [target.structural_adapter, target.sar_encoder]
    source_modules = list(sources)
    target_modules = list(targets)
    for projection, index in zip(target.deep_projections, DEEP_STAGE_INDICES):
        stage = model.joint_stages[index]
        source_modules.extend([stage.sar_common, stage.sar_unique])
        target_modules.extend([projection["sar_common"], projection["sar_unique"]])
    for target_module, source_module in zip(target_modules, source_modules):
        for target_parameter, source_parameter in zip(
            target_module.parameters(), source_module.parameters()
        ):
            target_parameter.mul_(momentum).add_(
                source_parameter.detach(), alpha=1.0 - momentum
            )
        for target_buffer, source_buffer in zip(
            target_module.buffers(), source_module.buffers()
        ):
            target_buffer.copy_(source_buffer)


def configure_pretraining_parameters(
    model: DamageFormerRGBSARA2DCSM, variant: str
) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_prefixes = ["sar_encoder."]
    if variant in ("m2", "m3"):
        trainable_prefixes.append("structural_adapter.")
    if variant == "m3":
        trainable_prefixes.extend(
            [
                "joint_stages.2.sar_common.",
                "joint_stages.2.sar_unique.",
                "joint_stages.3.sar_common.",
                "joint_stages.3.sar_unique.",
            ]
        )
    selected = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            parameter.requires_grad_(True)
            selected.append(parameter)
    if not selected:
        raise RuntimeError("DCSM pretraining selected no pathway parameters")
    return selected


def pathway_state(model: DamageFormerRGBSARA2DCSM, variant: str) -> Dict[str, torch.Tensor]:
    prefixes = ["sar_encoder."]
    if variant in ("m2", "m3"):
        prefixes.append("structural_adapter.")
    if variant == "m3":
        prefixes.extend(
            [
                "joint_stages.2.sar_common.",
                "joint_stages.2.sar_unique.",
                "joint_stages.3.sar_common.",
                "joint_stages.3.sar_unique.",
            ]
        )
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if any(name.startswith(prefix) for prefix in prefixes)
    }


def main() -> None:
    parser = argparse.ArgumentParser("Pretrain A2-DCSM native-SAR pathway")
    parser.add_argument("--variant", choices=("m1", "m2", "m3"), required=True)
    parser.add_argument("--source_a2_checkpoint", required=True)
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.60)
    parser.add_argument("--mask_patch_size", type=int, default=16)
    parser.add_argument("--ema_momentum", type=float, default=0.996)
    parser.add_argument("--rgb_anchor_strength", type=float, default=0.25)
    parser.add_argument("--focus_probability", type=float, default=0.75)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if not 0.0 < args.mask_ratio < 1.0:
        parser.error("--mask_ratio must lie in (0, 1)")
    if not 0.0 <= args.rgb_anchor_strength <= 1.0:
        parser.error("--rgb_anchor_strength must lie in [0, 1]")
    if not 0.0 <= args.ema_momentum < 1.0:
        parser.error("--ema_momentum must lie in [0, 1)")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DamageFormerRGBSARA2DCSM(
        variant=args.variant, pretrained_rgb=False
    ).to(device)
    source = load_locked_a2(model, args.source_a2_checkpoint)
    pathway_parameters = configure_pretraining_parameters(model, args.variant)
    predictors = nn.ModuleList(
        [DCSMFeaturePredictor(channels) for channels in DEEP_LATENT_CHANNELS]
    ).to(device)
    target = TargetSARPathway(model).to(device).eval()
    optimizer = torch.optim.AdamW(
        [
            {"params": pathway_parameters, "lr": args.learning_rate},
            {"params": predictors.parameters(), "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )

    amp_enabled = args.amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and args.amp == "fp16"
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and args.amp == "fp16"
        )

    step = 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("protocol") != "A2_DCSM_pretraining_resume_v1":
            raise RuntimeError("unexpected DCSM resume protocol")
        if resume.get("variant") != args.variant:
            raise RuntimeError("DCSM resume variant mismatch")
        model.load_state_dict(resume["model"], strict=True)
        target.load_state_dict(resume["target"], strict=True)
        predictors.load_state_dict(resume["predictors"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        if resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        step = int(resume.get("step", 0))

    names = read_data_list(args.train_data_list_path)
    micro_batches = (
        args.optimizer_steps * args.accumulation_steps * args.train_batch_size
    )
    dataset = HeterogeneousRGBSARDataset(
        args.train_dataset_path,
        names,
        args.crop_size,
        split="train",
        variant="h3",
        sar_stats_path=args.sar_stats_path,
        focus_probability=args.focus_probability,
        focus_class_weights=(0.0, 1.0, 2.0, 2.0),
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = event_balanced_sampler(names, micro_batches)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )

    event_counts = Counter(event_name(name) for name in names)
    contract = {
        "protocol": "A2_DCSM_pretraining_contract_v1",
        "variant": args.variant,
        "name": A2_DCSM_VARIANTS[args.variant].name,
        "source_a2": source,
        "train_list": str(Path(args.train_data_list_path).resolve()),
        "train_list_sha256": file_sha256(args.train_data_list_path),
        "event_counts": dict(sorted(event_counts.items())),
        "event_balanced_sampling": True,
        "class_balanced_region_weights": [0.25, 1.0, 2.0, 2.0],
        "building_focused_crop_probability": args.focus_probability,
        "pretraining_objectives": ["damage_conditioned_masked_feature_prediction"],
        "rgb_anchor_scope": "background_and_intact_train_labels_only",
        "changed_alignment": False,
        "structural_view": "scharr_display_domain" if args.variant in ("m2", "m3") else "none",
        "migrated_scope": (
            "adapter+encoder+stage34_sar_projection_neck"
            if args.variant == "m3"
            else "adapter+encoder"
            if args.variant == "m2"
            else "encoder"
        ),
        "test_accessed": False,
        "args": vars(args),
    }
    (output_dir / "pretraining_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )

    print("=" * 80, flush=True)
    print(f"Experiment       : {A2_DCSM_VARIANTS[args.variant].name}", flush=True)
    print(f"Source H6-A2     : {args.source_a2_checkpoint}", flush=True)
    print("Objective        : one damage-conditioned masked feature loss", flush=True)
    print("RGB anchor       : background/intact only; changed pixels never aligned", flush=True)
    print(f"Event balancing  : enabled ({len(event_counts)} events)", flush=True)
    print(f"Output           : {output_dir}", flush=True)
    print("=" * 80, flush=True)

    model.train()
    model.rgb_encoder.eval()
    for name, module in model.named_modules():
        if name.startswith("rgb_encoder"):
            module.eval()
    predictors.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    running: Dict[str, float] = {}
    window_start = time.time()
    micro_step = 0

    while step < args.optimizer_steps:
        try:
            rgb, sar, _, damage, sample_names = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            rgb, sar, _, damage, sample_names = next(iterator)
        rgb = rgb.to(device, non_blocking=True)
        sar = sar[:, :1].to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        patch_mask = random_patch_mask(
            sar.shape[0], sar.shape[2], sar.shape[3],
            args.mask_patch_size, args.mask_ratio, device,
        )
        masked_sar = sar * (1.0 - patch_mask.to(sar.dtype))

        with torch.no_grad():
            rgb_stages = model.rgb_encoder(rgb)
            target_features = target(sar)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            online_sar = model.adapt_native_sar(masked_sar)
            online_stages = model.sar_encoder(online_sar)
            unchanged = unchanged_mask_from_damage(damage, args.ignore_index)
            changed = ((damage == 2) | (damage == 3))[:, None].float()
            stage_losses = []
            common_cosines = []
            changed_cosines = []
            target_stds = []
            prediction_stds = []
            for predictor, index, target_pair in zip(
                predictors, DEEP_STAGE_INDICES, target_features
            ):
                stage = model.joint_stages[index]
                online_common = stage.sar_common(online_stages[index])
                online_unique = stage.sar_unique(online_stages[index])
                target_common, target_unique = target_pair
                with torch.no_grad():
                    rgb_common = stage.rgb_common(rgb_stages[index])
                    conditioned_common = damage_conditioned_target(
                        target_common,
                        rgb_common,
                        unchanged,
                        args.rgb_anchor_strength,
                    )
                prediction = predictor(
                    torch.cat([online_common, online_unique], dim=1)
                )
                target_value = torch.cat(
                    [conditioned_common, target_unique], dim=1
                )
                # Collapse diagnostics only.  These values never participate
                # in the objective and therefore do not add another loss.
                target_stds.append(
                    target_value.float().std((0, 2, 3), unbiased=False).mean()
                )
                prediction_stds.append(
                    prediction.float().std((0, 2, 3), unbiased=False).mean()
                )
                weights = class_balanced_region_weights(
                    damage,
                    prediction.shape[-2:],
                    ignore_index=args.ignore_index,
                )
                stage_losses.append(
                    masked_cosine_prediction_loss(
                        prediction, target_value, patch_mask, weights
                    )
                )
                with torch.no_grad():
                    similarity = torch.nn.functional.cosine_similarity(
                        target_common.float(), rgb_common.float(), dim=1
                    )
                    unchanged_low = torch.nn.functional.interpolate(
                        unchanged, size=similarity.shape[-2:], mode="nearest"
                    )[:, 0].bool()
                    # Use only explicit damaged/destroyed labels.  Taking the
                    # inverse of ``unchanged_low`` would incorrectly include
                    # ignore/padding pixels in this diagnostic.
                    changed_low = torch.nn.functional.interpolate(
                        changed, size=similarity.shape[-2:], mode="nearest"
                    )[:, 0].bool()
                    if unchanged_low.any():
                        common_cosines.append(similarity[unchanged_low].mean())
                    if changed_low.any():
                        changed_cosines.append(similarity[changed_low].mean())
            loss = torch.stack(stage_losses).mean()
            scaled_loss = loss / args.accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite DCSM loss before step {step + 1}")
        scaler.scale(scaled_loss).backward()
        micro_step += 1
        running["dcsm_loss"] = running.get("dcsm_loss", 0.0) + float(loss.item())
        if common_cosines:
            running["unchanged_rgb_sar_cosine"] = running.get(
                "unchanged_rgb_sar_cosine", 0.0
            ) + float(torch.stack(common_cosines).mean().item())
        if changed_cosines:
            running["changed_rgb_sar_cosine"] = running.get(
                "changed_rgb_sar_cosine", 0.0
            ) + float(torch.stack(changed_cosines).mean().item())
        if target_stds:
            running["target_feature_std"] = running.get(
                "target_feature_std", 0.0
            ) + float(torch.stack(target_stds).mean().item())
        if prediction_stds:
            running["prediction_feature_std"] = running.get(
                "prediction_feature_std", 0.0
            ) + float(torch.stack(prediction_stds).mean().item())
        if micro_step % args.accumulation_steps:
            continue

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.grad is not None]
            + [parameter for parameter in predictors.parameters() if parameter.grad is not None],
            5.0,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        ema_update(target, model, args.ema_momentum)

        if step % args.log_every == 0:
            denominator = args.log_every * args.accumulation_steps
            elapsed = max(time.time() - window_start, 1e-6)
            payload = {
                "step": step,
                "variant": args.variant,
                **{key: value / denominator for key, value in running.items()},
                "structural_gain": (
                    float(torch.tanh(model.structural_adapter.transition).item())
                    if isinstance(model.structural_adapter, StructuralSARAdapter)
                    else 0.0
                ),
                "optimizer_steps_per_second": args.log_every / elapsed,
                "peak_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda" else 0.0
                ),
            }
            print(json.dumps(payload, allow_nan=False), flush=True)
            running = {}
            window_start = time.time()

        if step % args.save_every == 0 or step == args.optimizer_steps:
            resume_payload = {
                "protocol": "A2_DCSM_pretraining_resume_v1",
                "variant": args.variant,
                "step": step,
                "model": model.state_dict(),
                "target": target.state_dict(),
                "predictors": predictors.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "source_a2": source,
                "test_accessed": False,
            }
            atomic_save(resume_payload, output_dir / "latest_resume.pth")

    final = {
        "protocol": "A2_DCSM_pathway_v1",
        "variant": args.variant,
        "step": step,
        "pathway": pathway_state(model, args.variant),
        "source_a2": source,
        "train_list_sha256": file_sha256(args.train_data_list_path),
        "objective": "damage_conditioned_masked_feature_prediction",
        "changed_alignment": False,
        "test_accessed": False,
    }
    atomic_save(final, output_dir / "pathway_final.pth")
    print(f"A2-DCSM pretraining complete at step {step}", flush=True)


if __name__ == "__main__":
    main()
