#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import parse_batch, read_list, sha256_file
from bda_benchmark.experiments.spi_selective_memory_oracle_v1.oracle_core import confusion, force_occurrence_predictions, valid_miou
from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.core import (
    MetricKeyEncoder, extract_region_records, local_tolerant_alignment, relation_map,
    retrieve_global_topk, rgb_sobel_edge,
)
from bda_benchmark.experiments.spi_superpixel_relation_occurrence_v1.core import make_region_ids
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import fast_hist, metrics_from_hist

from . import PROTOCOL
from .core import entropy_from_counts


def event_name(name: str) -> str:
    return re.sub(r"_\d+$", "", str(name))


def stats(value: torch.Tensor) -> tuple[float, float]:
    return float(value.mean().item()), float(value.float().std(unbiased=False).item())


def proposal(
    *, source: str, index: int, region_key: str, image: str, event: str,
    logits: torch.Tensor, target: torch.Tensor, target_affected: bool,
    features: dict[str, float],
) -> dict | None:
    base = logits.argmax(1)
    proposed = force_occurrence_predictions(logits, target_affected)
    changed = proposed != base
    if not bool(changed.any()):
        return None
    base_correct, proposed_correct = base == target, proposed == target
    return {
        "index": int(index), "region_key": region_key, "image": image, "event": event,
        "source": source, "target_affected": bool(target_affected), "features": features,
        "delta": confusion(proposed, target) - confusion(base, target),
        "changed": int(changed.sum().item()),
        "rescued": int((~base_correct & proposed_correct).sum().item()),
        "harmed": int((base_correct & ~proposed_correct).sum().item()),
        "affected_recovered": int(((target >= 2) & (base <= 1) & (proposed >= 2)).sum().item()),
        "false_affected": int(((target <= 1) & (base <= 1) & (proposed >= 2)).sum().item()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Build train/validation proposal utility cache")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--parent_ckpt", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--alignment_radius", type=int, default=1)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--expected_images", type=int, default=0)
    parser.add_argument("--expected_parent_miou", type=float, default=-1.0)
    parser.add_argument("--parent_tolerance_pp", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if "test" in Path(args.data_list).name.lower():
        raise RuntimeError("Test access is forbidden for target construction")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    names = read_list(args.data_list)
    if args.expected_images and len(names) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, found {len(names)}")
    model = DamageFormer_S8_MICB_DSMIL().to(device).eval()
    parent_report = load_parent(model, args.parent_ckpt)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    saved = torch.load(args.metric, map_location="cpu", weights_only=False)
    bank = torch.load(args.memory, map_location="cpu", weights_only=False)
    metric = MetricKeyEncoder(saved["input_dim"], saved["key_dim"]).to(device).eval()
    metric.load_state_dict(saved["model"], strict=True)
    for parameter in metric.parameters(): parameter.requires_grad_(False)
    memory_keys, memory_labels = bank["keys"].to(device), bank["labels"].to(device)
    memory_events = list(map(str, bank["events"]))

    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, names, crop_size=1024, type="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    base_hist = torch.zeros(4, 4, dtype=torch.float64)
    loc_hist = torch.zeros(2, 2, dtype=torch.float64)
    event_base = defaultdict(lambda: torch.zeros(4, 4, dtype=torch.float64))
    image_base, image_loc, proposals = {}, {}, []
    proposal_index = 0

    for image_number, batch in enumerate(loader, 1):
        pre, post, location, damage, sample_names = parse_batch(batch, device)
        loc_logits, damage_logits, aux = model(pre, post, return_aux=True)
        damage_logits = F.interpolate(damage_logits.float(), damage.shape[-2:], mode="bilinear", align_corners=False)
        loc_logits = F.interpolate(loc_logits.float(), location.shape[-2:], mode="bilinear", align_corners=False)
        base_prediction, loc_prediction = damage_logits.argmax(1), loc_logits.argmax(1)
        image = str(sample_names[0]); event = event_name(image)
        local_base_hist = fast_hist(base_prediction, damage, 4, 255).cpu().double()
        local_loc_hist = fast_hist(loc_prediction, location, 2, 255).cpu().double()
        base_hist += local_base_hist; loc_hist += local_loc_hist; event_base[event] += local_base_hist
        image_base[image], image_loc[image] = local_base_hist, local_loc_hist

        head = model.micb_dsmil
        rgb, sar = head.rgb_proj(aux["rgb_feature"]), head.sar_proj(aux["sar_feature"])
        fused = head.fused_proj(aux["fused_roi"] / aux["roi_weight"].clamp_min(1e-6))
        aligned, align_entropy, offset = local_tolerant_alignment(rgb, sar, args.alignment_radius)
        relation = relation_map(rgb, aligned, fused, head.cross_token)
        edge = rgb_sobel_edge(pre, relation.shape[-2:])
        ids = make_region_ids(pre, relation.shape[-2:], mode="slic", n_segments=args.segments, compactness=args.compactness)
        rows = extract_region_records(
            relation, fused, post[:, :1], ids, aux["p_building"], align_entropy, offset, edge,
            None, None, sample_names, [event], require_reliable_label=False,
        )
        if rows:
            descriptors = torch.stack([row.descriptor for row in rows]).to(device)
            keys, direct_logits = metric(descriptors)
            direct_prob = direct_logits.softmax(1)
            direct_pred = direct_prob.argmax(1) + 1
            retrieval = retrieve_global_topk(
                keys, memory_keys, memory_labels, topk=args.topk,
                query_images=[row.image_id for row in rows], memory_images=bank["images"],
                query_events=[row.event_id for row in rows], memory_events=bank["events"],
                exclude_same_image=True, exclude_same_event=True,
            )
            ids_full = F.interpolate(ids[:, None].float(), damage.shape[-2:], mode="nearest")[:, 0].long()
            p_building = F.interpolate(aux["p_building"].float(), damage.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            for row_number, row in enumerate(rows):
                mask = (ids_full[row.batch_index] == row.region_id) & (p_building[row.batch_index] >= 0.5) & (damage[row.batch_index] != 255)
                if not bool(mask.any()): continue
                local_logits = damage_logits[row.batch_index, :, mask].transpose(0, 1)
                local_target = damage[row.batch_index, mask]
                probability = local_logits.softmax(1)
                entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1)
                affected = probability[:, 2:4].sum(1)
                occ_margin = probability[:, 2:4].max(1).values - probability[:, :2].max(1).values
                base_fraction = torch.bincount(local_logits.argmax(1), minlength=4).float() / mask.sum()
                direct = direct_prob[row_number]; memory = retrieval["posterior"][row_number]
                neighbour_indices = retrieval["indices"][row_number].detach().cpu().tolist()
                memory_event_count, memory_event_entropy = entropy_from_counts([memory_events[i] for i in neighbour_indices])
                geometry = row.descriptor[-9:].float()
                features: dict[str, float] = {}
                for cls in range(4):
                    features[f"p3_p{cls}_mean"], features[f"p3_p{cls}_std"] = stats(probability[:, cls])
                    features[f"base_frac{cls}"] = float(base_fraction[cls].item())
                features["p3_affected_mean"], features["p3_affected_std"] = stats(affected)
                features["p3_entropy_mean"], features["p3_entropy_std"] = stats(entropy)
                features["p3_occ_margin_mean"], features["p3_occ_margin_std"] = stats(occ_margin)
                features.update({
                    "log_pixel_count": float(torch.tensor(mask.sum().item() + 1.0).log().item()),
                    "region_area_fraction": float(geometry[0].item()), "building_probability": float(geometry[4].item()),
                    "alignment_entropy": float(geometry[5].item()), "expected_offset": float(geometry[6].item()),
                    "rgb_edge_mean": float(geometry[7].item()), "rgb_boundary_edge": float(geometry[8].item()),
                    "direct_p0": float(direct[0]), "direct_p1": float(direct[1]), "direct_p2": float(direct[2]),
                    "direct_margin": float(direct.topk(2).values.diff().abs().item()),
                    "direct_affected": float(direct[1:].sum().item()),
                    "memory_p0": float(memory[0]), "memory_p1": float(memory[1]), "memory_p2": float(memory[2]),
                    "memory_margin": float(retrieval["margin"][row_number]),
                    "memory_agreement": float(retrieval["agreement"][row_number]),
                    "memory_max_similarity": float(retrieval["max_similarity"][row_number]),
                    "memory_affected": float(memory[1:].sum().item()),
                    "memory_event_count": memory_event_count, "memory_event_entropy": memory_event_entropy,
                    "direct_memory_l1": float((direct - memory).abs().sum().item()),
                    "direct_memory_affected_delta": float(direct[1:].sum().item() - memory[1:].sum().item()),
                    "direct_memory_same_prediction": float(int(direct_pred[row_number]) == int(retrieval["prediction"][row_number])),
                    "direct_memory_same_occurrence": float((int(direct_pred[row_number]) >= 2) == (int(retrieval["prediction"][row_number]) >= 2)),
                })
                region_key = f"{image}:{row.region_id}"
                for source, target_affected in (
                    ("direct", bool(direct_pred[row_number] >= 2)),
                    ("memory", bool(retrieval["prediction"][row_number] >= 2)),
                ):
                    local_features = dict(features)
                    local_features.update({
                        "source_direct": float(source == "direct"), "source_memory": float(source == "memory"),
                        "proposal_affected": float(target_affected),
                    })
                    item = proposal(
                        source=source, index=proposal_index, region_key=region_key, image=image, event=event,
                        logits=local_logits, target=local_target, target_affected=target_affected, features=local_features,
                    )
                    proposal_index += 1
                    if item is not None: proposals.append(item)
        if image_number % 25 == 0 or image_number == len(loader):
            print(f"targets {image_number}/{len(loader)} proposals={len(proposals)}", flush=True)

    baseline = valid_miou(base_hist)
    if args.expected_parent_miou >= 0:
        error_pp = 100.0 * (baseline - args.expected_parent_miou)
        if abs(error_pp) > args.parent_tolerance_pp:
            raise RuntimeError(
                f"P3 reproduction failed: expected={args.expected_parent_miou:.8f}, "
                f"observed={baseline:.8f}, error={error_pp:+.4f}pp"
            )
    for row in proposals:
        row["utility_miou_pp"] = 100.0 * (valid_miou(base_hist + row["delta"]) - baseline)
        row["accept_label"] = bool(row["utility_miou_pp"] > 0.0)
    payload = {
        "protocol": PROTOCOL + "_METRIC_TARGETS", "split": args.split,
        "allowed_for_verifier_training": args.split == "train", "test_accessed": False,
        "rows": proposals, "base_hist": base_hist, "loc_hist": loc_hist,
        "event_base_hist": dict(event_base), "image_base_hist": image_base, "image_loc_hist": image_loc,
        "parent_metrics": metrics_from_hist(loc_hist, base_hist), "parent_load": parent_report,
        "metadata": {
            "data_list": str(Path(args.data_list).resolve()), "data_list_sha256": sha256_file(args.data_list),
            "image_count": len(names), "exclude_same_image": True, "exclude_same_event": True,
            "gt_is_feature": False, "gt_role": "metric-aligned utility target only",
            "severity_policy": "occurrence-only; preserve Damaged-vs-Destroyed ordering",
        },
    }
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved: {destination} rows={len(proposals)} split={args.split} parent_mIoU={baseline:.8f}")


if __name__ == "__main__": main()
