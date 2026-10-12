#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import parse_batch, read_list, sha256_file
from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.core import (
    MetricKeyEncoder,
    extract_region_records,
    local_tolerant_alignment,
    relation_map,
    retrieve_global_topk,
    rgb_sobel_edge,
)
from bda_benchmark.experiments.spi_superpixel_relation_occurrence_v1.core import make_region_ids
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import fast_hist, metrics_from_hist

from . import PROTOCOL
from .oracle_core import (
    OracleCandidate, confusion, force_occurrence_predictions,
    greedy_grouped_metric_oracle, greedy_metric_oracle, valid_miou,
)


def event_name(name: str) -> str:
    return re.sub(r"_\d+$", "", str(name))


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist() if value.numel() != 1 else value.item()
    return value


def make_candidate(
    *, index: int, event: str, logits: torch.Tensor, target: torch.Tensor,
    target_affected: bool, source: str,
) -> OracleCandidate | None:
    base = logits.argmax(1)
    proposal = force_occurrence_predictions(logits, target_affected)
    changed_mask = proposal != base
    if not bool(changed_mask.any()):
        return None
    base_hist = confusion(base, target)
    proposal_hist = confusion(proposal, target)
    base_correct = base == target
    proposal_correct = proposal == target
    return OracleCandidate(
        index=index,
        event=event,
        delta=proposal_hist - base_hist,
        changed=int(changed_mask.sum().item()),
        rescued=int((~base_correct & proposal_correct).sum().item()),
        harmed=int((base_correct & ~proposal_correct).sum().item()),
        affected_recovered=int(((target >= 2) & (base <= 1) & (proposal >= 2)).sum().item()),
        false_affected=int(((target <= 1) & (base <= 1) & (proposal >= 2)).sum().item()),
        target_affected=bool(target_affected),
        source=str(source),
    )


def summarize_oracle(
    name: str,
    base_hist: torch.Tensor,
    loc_hist: torch.Tensor,
    event_base: dict[str, torch.Tensor],
    candidates: list[OracleCandidate],
    base_affected_misses: int,
    grouped: bool = False,
) -> tuple[dict[str, Any], set[Any], list[OracleCandidate]]:
    oracle = greedy_grouped_metric_oracle if grouped else greedy_metric_oracle
    final_hist, accepted_ids, ordered = oracle(base_hist, candidates)
    accepted = set(accepted_ids)
    selected = [
        row for row in ordered
        if ((row.index, row.source) in accepted if grouped else row.index in accepted)
    ]
    event_final = {event: hist.clone() for event, hist in event_base.items()}
    for row in selected:
        event_final[row.event] += row.delta
    event_gain = {
        event: 100.0 * (valid_miou(event_final[event]) - valid_miou(event_base[event]))
        for event in sorted(event_base)
    }
    rescued = sum(row.rescued for row in selected)
    harmed = sum(row.harmed for row in selected)
    affected_recovered = sum(row.affected_recovered for row in selected)
    changed = sum(row.changed for row in selected)
    output = {
        "name": name,
        "proposal_count": len(candidates),
        "accepted_count": len(selected),
        "acceptance_rate": len(selected) / max(len(candidates), 1),
        "metrics": metrics_from_hist(loc_hist, final_hist),
        "gain_pp": 100.0 * (valid_miou(final_hist) - valid_miou(base_hist)),
        "rescued_exact_pixels": rescued,
        "harmed_exact_pixels": harmed,
        "net_exact_pixels": rescued - harmed,
        "changed_pixels": changed,
        "correction_precision": rescued / max(rescued + harmed, 1),
        "affected_miss_recovered": affected_recovered,
        "affected_oracle_recovery": affected_recovered / max(base_affected_misses, 1),
        "event_gain_pp": event_gain,
        "nonnegative_events": sum(value >= 0.0 for value in event_gain.values()),
        "improved_events": sum(value > 0.0 for value in event_gain.values()),
        "event_count": len(event_gain),
    }
    return output, accepted, ordered


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Proposal-conditioned selective Memory Oracle on validation only")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--parent_ckpt", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--target_cache", required=True)
    parser.add_argument("--segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--alignment_radius", type=int, default=1)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--exclude_same_event", action="store_true")
    parser.add_argument("--expected_images", type=int, default=450)
    parser.add_argument("--expected_parent_miou", type=float, default=-1.0)
    parser.add_argument("--parent_tolerance_pp", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if "test" in Path(args.val_list).name.lower():
        raise RuntimeError("Oracle is validation-only; test-list access is forbidden")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    names = read_list(args.val_list)
    if len(names) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} validation images, found {len(names)}")

    # The locked P3 was evaluated with the model default roi_alpha=0.1.  This
    # scalar is not part of state_dict, so constructing with roi_alpha=1.0
    # silently creates a different pixel model even though all weights load.
    model = DamageFormer_S8_MICB_DSMIL().to(device).eval()
    parent_report = load_parent(model, args.parent_ckpt)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metric_saved = torch.load(args.metric, map_location="cpu", weights_only=False)
    memory = torch.load(args.memory, map_location="cpu", weights_only=False)
    if memory.get("protocol") != "SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2_BANK":
        raise RuntimeError(f"unexpected Memory protocol: {memory.get('protocol')!r}")
    if int(memory["keys"].shape[1]) != int(metric_saved["key_dim"]):
        raise RuntimeError("Metric key dimension does not match the Memory bank")
    metric = MetricKeyEncoder(metric_saved["input_dim"], metric_saved["key_dim"]).to(device).eval()
    metric.load_state_dict(metric_saved["model"], strict=True)
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    memory_keys = memory["keys"].to(device)
    memory_labels = memory["labels"].to(device)

    # Full-image validation is used here, not the 640-pixel record-training crop.
    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, names, crop_size=1024, type="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    loc_hist = torch.zeros(2, 2, dtype=torch.float64)
    base_hist = torch.zeros(4, 4, dtype=torch.float64)
    event_base: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(4, 4, dtype=torch.float64))
    memory_candidates: list[OracleCandidate] = []
    direct_candidates: list[OracleCandidate] = []
    cache_rows: list[dict[str, Any]] = []
    base_affected_misses = 0
    candidate_index = 0
    funnel = {
        "base_affected_misses": 0,
        "inside_predicted_building": 0,
        "inside_candidate_slic": 0,
        "memory_affected_advice": 0,
        "memory_changed_to_affected": 0,
        "direct_affected_advice": 0,
        "direct_changed_to_affected": 0,
    }

    for image_number, batch in enumerate(loader, 1):
        pre, post, location, damage, sample_names = parse_batch(batch, device)
        loc_logits, damage_logits, aux = model(pre, post, return_aux=True)
        damage_logits = F.interpolate(damage_logits.float(), size=damage.shape[-2:], mode="bilinear", align_corners=False)
        loc_logits = F.interpolate(loc_logits.float(), size=location.shape[-2:], mode="bilinear", align_corners=False)
        base_prediction = damage_logits.argmax(1)
        loc_prediction = loc_logits.argmax(1)
        image_hist = fast_hist(base_prediction, damage, 4, 255).cpu()
        base_hist += image_hist
        loc_hist += fast_hist(loc_prediction, location, 2, 255).cpu()
        event = event_name(str(sample_names[0]))
        event_base[event] += image_hist
        affected_miss = (damage >= 2) & (damage <= 3) & (base_prediction <= 1)
        image_misses = int(affected_miss.sum().item())
        base_affected_misses += image_misses
        funnel["base_affected_misses"] += image_misses

        head = model.micb_dsmil
        rgb = head.rgb_proj(aux["rgb_feature"])
        sar = head.sar_proj(aux["sar_feature"])
        # Existing V2 records/Metric/Memory were deliberately (but previously
        # undocumented) built from roi_alpha=1.0, i.e. the unweighted fused
        # feature.  Keep that descriptor contract while evaluating proposals
        # against the real roi_alpha=0.1 P3 pixel logits.
        fused_unweighted = aux["fused_roi"] / aux["roi_weight"].clamp_min(1e-6)
        fused = head.fused_proj(fused_unweighted)
        aligned, entropy, offset = local_tolerant_alignment(rgb, sar, args.alignment_radius)
        relation = relation_map(rgb, aligned, fused, head.cross_token)
        edge = rgb_sobel_edge(pre, relation.shape[-2:])
        ids = make_region_ids(
            pre, relation.shape[-2:], mode="slic",
            n_segments=args.segments, compactness=args.compactness,
        )
        rows = extract_region_records(
            relation, fused, post[:, :1], ids, aux["p_building"], entropy, offset, edge,
            None, None, sample_names, [event], require_reliable_label=False,
        )
        if not rows:
            continue
        descriptors = torch.stack([row.descriptor for row in rows]).to(device)
        keys, direct_logits = metric(descriptors)
        direct_posterior = direct_logits.softmax(1)
        direct_prediction = direct_posterior.argmax(1) + 1
        retrieval = retrieve_global_topk(
            keys, memory_keys, memory_labels, topk=args.topk,
            query_images=[row.image_id for row in rows], memory_images=memory["images"],
            query_events=[row.event_id for row in rows], memory_events=memory["events"],
            exclude_same_image=True, exclude_same_event=args.exclude_same_event,
        )
        ids_full = F.interpolate(ids[:, None].float(), size=damage.shape[-2:], mode="nearest")[:, 0].long()
        p_building = F.interpolate(aux["p_building"].float(), size=damage.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        funnel["inside_predicted_building"] += int((affected_miss & (p_building >= 0.5)).sum().item())
        candidate_union = torch.zeros_like(affected_miss, dtype=torch.bool)
        memory_advice_union = torch.zeros_like(affected_miss, dtype=torch.bool)
        memory_changed_union = torch.zeros_like(affected_miss, dtype=torch.bool)
        direct_advice_union = torch.zeros_like(affected_miss, dtype=torch.bool)
        direct_changed_union = torch.zeros_like(affected_miss, dtype=torch.bool)

        for row_number, row in enumerate(rows):
            batch_index = row.batch_index
            mask = (
                (ids_full[batch_index] == row.region_id)
                & (p_building[batch_index] >= 0.5)
                & (damage[batch_index] != 255)
            )
            if not bool(mask.any()):
                continue
            candidate_union[batch_index] |= mask
            local_logits = damage_logits[batch_index, :, mask].transpose(0, 1)
            local_target = damage[batch_index, mask]
            memory_target_affected = bool(retrieval["prediction"][row_number].item() >= 2)
            direct_target_affected = bool(direct_prediction[row_number].item() >= 2)
            memory_candidate = make_candidate(
                index=candidate_index, event=event, logits=local_logits,
                target=local_target, target_affected=memory_target_affected, source="memory",
            )
            direct_candidate = make_candidate(
                index=candidate_index, event=event, logits=local_logits,
                target=local_target, target_affected=direct_target_affected, source="direct",
            )
            base_local = local_logits.argmax(1)
            if memory_target_affected:
                memory_advice_union[batch_index] |= mask
                changed = force_occurrence_predictions(local_logits, True) != base_local
                local_changed = torch.zeros_like(mask); local_changed[mask] = changed
                memory_changed_union[batch_index] |= local_changed
            if direct_target_affected:
                direct_advice_union[batch_index] |= mask
                changed = force_occurrence_predictions(local_logits, True) != base_local
                local_changed = torch.zeros_like(mask); local_changed[mask] = changed
                direct_changed_union[batch_index] |= local_changed
            if memory_candidate is not None:
                memory_candidates.append(memory_candidate)
            if direct_candidate is not None:
                direct_candidates.append(direct_candidate)
            cache_rows.append({
                "index": candidate_index,
                "image": row.image_id,
                "event": event,
                "region_id": row.region_id,
                "descriptor": row.descriptor.detach().cpu().half(),
                "base_class_counts": torch.bincount(local_logits.argmax(1).cpu(), minlength=4),
                "memory_posterior": retrieval["posterior"][row_number].detach().cpu(),
                "memory_prediction": int(retrieval["prediction"][row_number].item()),
                "memory_margin": float(retrieval["margin"][row_number].item()),
                "memory_agreement": float(retrieval["agreement"][row_number].item()),
                "memory_max_similarity": float(retrieval["max_similarity"][row_number].item()),
                "direct_posterior": direct_posterior[row_number].detach().cpu(),
                "direct_prediction": int(direct_prediction[row_number].item()),
                "pixel_count": int(mask.sum().item()),
            })
            candidate_index += 1
        funnel["inside_candidate_slic"] += int((affected_miss & candidate_union).sum().item())
        funnel["memory_affected_advice"] += int((affected_miss & memory_advice_union).sum().item())
        funnel["memory_changed_to_affected"] += int((affected_miss & memory_changed_union).sum().item())
        funnel["direct_affected_advice"] += int((affected_miss & direct_advice_union).sum().item())
        funnel["direct_changed_to_affected"] += int((affected_miss & direct_changed_union).sum().item())
        if image_number % 25 == 0 or image_number == len(loader):
            print(
                f"oracle {image_number}/{len(loader)} regions={candidate_index} "
                f"memory_proposals={len(memory_candidates)} direct_proposals={len(direct_candidates)}",
                flush=True,
            )

    parent_metrics = metrics_from_hist(loc_hist, base_hist)
    if args.expected_parent_miou >= 0:
        error_pp = 100.0 * (parent_metrics["valid_miou"] - args.expected_parent_miou)
        if abs(error_pp) > args.parent_tolerance_pp:
            raise RuntimeError(
                f"P3 reproduction failed: expected={args.expected_parent_miou:.8f}, "
                f"observed={parent_metrics['valid_miou']:.8f}, error={error_pp:+.4f}pp"
            )
    memory_result, memory_accepted, memory_ordered = summarize_oracle(
        "memory_occurrence_oracle", base_hist, loc_hist, event_base,
        memory_candidates, base_affected_misses,
    )
    direct_result, direct_accepted, _ = summarize_oracle(
        "direct_head_occurrence_oracle", base_hist, loc_hist, event_base,
        direct_candidates, base_affected_misses,
    )
    memory_by_region = {row.index: row for row in memory_candidates}
    direct_by_region = {row.index: row for row in direct_candidates}
    union_candidates: list[OracleCandidate] = []
    overlap = {"same_proposal": 0, "memory_only": 0, "direct_only": 0, "conflicting_proposals": 0}
    memory_unique: list[OracleCandidate] = []
    direct_unique: list[OracleCandidate] = []
    for index in sorted(set(memory_by_region) | set(direct_by_region)):
        memory_row = memory_by_region.get(index)
        direct_row = direct_by_region.get(index)
        if memory_row is not None and direct_row is not None:
            if memory_row.target_affected == direct_row.target_affected:
                overlap["same_proposal"] += 1
                union_candidates.append(memory_row)
            else:
                overlap["conflicting_proposals"] += 1
                union_candidates.extend((memory_row, direct_row))
                memory_unique.append(memory_row); direct_unique.append(direct_row)
        elif memory_row is not None:
            overlap["memory_only"] += 1
            union_candidates.append(memory_row); memory_unique.append(memory_row)
        elif direct_row is not None:
            overlap["direct_only"] += 1
            union_candidates.append(direct_row); direct_unique.append(direct_row)
    union_result, union_accepted, _ = summarize_oracle(
        "direct_union_memory_occurrence_oracle", base_hist, loc_hist, event_base,
        union_candidates, base_affected_misses, grouped=True,
    )
    memory_unique_result, _, _ = summarize_oracle(
        "memory_unique_proposal_oracle", base_hist, loc_hist, event_base,
        memory_unique, base_affected_misses, grouped=True,
    )
    direct_unique_result, _, _ = summarize_oracle(
        "direct_unique_proposal_oracle", base_hist, loc_hist, event_base,
        direct_unique, base_affected_misses, grouped=True,
    )
    total_misses = max(funnel["base_affected_misses"], 1)
    funnel["memory_oracle_recovered"] = memory_result["affected_miss_recovered"]
    funnel["direct_oracle_recovered"] = direct_result["affected_miss_recovered"]
    funnel["union_oracle_recovered"] = union_result["affected_miss_recovered"]
    funnel_rates = {key + "_rate": value / total_misses for key, value in funnel.items() if key != "base_affected_misses"}
    utility = {row.index: row.individual_gain_pp for row in memory_ordered}
    memory_lookup = {row.index: row for row in memory_candidates}
    for row in cache_rows:
        index = row["index"]
        candidate = memory_lookup.get(index)
        row["has_memory_proposal"] = candidate is not None
        row["utility_miou_pp"] = float(utility.get(index, 0.0))
        row["oracle_accept"] = index in memory_accepted
        row["rescued_exact_pixels"] = candidate.rescued if candidate else 0
        row["harmed_exact_pixels"] = candidate.harmed if candidate else 0
        row["affected_recovered"] = candidate.affected_recovered if candidate else 0
        row["false_affected"] = candidate.false_affected if candidate else 0

    result = {
        "protocol": PROTOCOL,
        "scope": "validation-only proposal-conditioned occurrence upper bound",
        "parent": {"metrics": parent_metrics, "checkpoint_load": parent_report},
        "memory_oracle": memory_result,
        "direct_head_oracle": direct_result,
        "union_oracle": union_result,
        "memory_unique_oracle": memory_unique_result,
        "direct_unique_oracle": direct_unique_result,
        "proposal_overlap": overlap,
        "recovery_funnel": funnel | funnel_rates,
        "memory_increment_over_direct_oracle_pp": memory_result["gain_pp"] - direct_result["gain_pp"],
        "union_increment_over_direct_oracle_pp": union_result["gain_pp"] - direct_result["gain_pp"],
        "memory_complementarity_pass": union_result["gain_pp"] - direct_result["gain_pp"] >= 0.20,
        "base_affected_miss_pixels": base_affected_misses,
        "proposal_definition": {
            "region": "SLIC intersected with predicted-building probability >= 0.5",
            "memory_read": f"global cosine top-{args.topk}",
            "decision": "Intact versus Affected occurrence only",
            "severity": "Damaged/Destroyed logit ordering is preserved",
            "gt_role": "accept/reject proposals only; never used to generate retrieval advice",
            "oracle": "greedy positive global-mIoU utility; not an exact combinatorial maximum",
            "p3_pixel_roi_alpha": 0.1,
            "memory_descriptor_fused_view": "unweighted fused feature, compatible with existing V2 roi_alpha=1.0 records",
        },
        "data": {
            "val_list": str(Path(args.val_list).resolve()),
            "val_list_sha256": sha256_file(args.val_list),
            "image_count": len(names),
            "exclude_same_event": args.exclude_same_event,
            "parent_checkpoint_sha256": sha256_file(args.parent_ckpt),
            "metric_checkpoint_sha256": sha256_file(args.metric),
            "memory_bank_sha256": sha256_file(args.memory),
        },
        "test_accessed": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(native(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    target = Path(args.target_cache)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": PROTOCOL + "_TARGETS",
        "rows": cache_rows,
        "metadata": result["proposal_definition"] | result["data"] | {
            "split": "validation",
            "allowed_use": "oracle analysis only; forbidden for Verifier fitting",
            "forbid_verifier_training": True,
            "test_accessed": False,
        },
    }, target)
    print(json.dumps(native(result), indent=2, ensure_ascii=False))
    print(f"Saved: {output}")
    print(f"Saved verifier targets: {target}")


if __name__ == "__main__":
    main()
