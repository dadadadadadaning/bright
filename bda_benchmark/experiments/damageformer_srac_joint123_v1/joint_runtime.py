"""Runtime for the isolated J1/J2/J3 joint-context comparison.

Q, the region embedding, the A1 front, the source Memory bank and the learned
retrieval metric are read-only.  J2 and J3 deliberately enter the same decoder
branch; their only experimental difference is the selected Memory row IDs.
"""
from __future__ import annotations

import hashlib
from typing import Dict

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
    transfer_checked_batch as parse_batch,
)
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import (
    JointContextDecoder,
)
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import (
    gather_support,
    select_ids,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    confusion_update,
    event,
    region_targets,
)


JOINT_FORMAT = "SRAC_JOINT123_V1"
ARMS = ("j1", "j2", "j3")
ARM_MEANING = {
    "j1": "A1 pixel evidence plus the current region read through learned control slots",
    "j2": "A1 pixel evidence plus current region read from cosine-retrieved Train Memory",
    "j3": "A1 pixel evidence plus current region read from metric-retrieved Train Memory",
}


def module_fingerprint(module: torch.nn.Module) -> str:
    checksum = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        checksum.update(name.encode())
        checksum.update(str(tensor.dtype).encode())
        checksum.update(str(tuple(tensor.shape)).encode())
        checksum.update(tensor.numpy().tobytes())
    return checksum.hexdigest()


def bank_fingerprint(bank: Dict[str, object]) -> str:
    checksum = hashlib.sha256()
    for name in sorted(bank):
        value = bank[name]
        checksum.update(name.encode())
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            checksum.update(str(tensor.dtype).encode())
            checksum.update(str(tuple(tensor.shape)).encode())
            checksum.update(tensor.numpy().tobytes())
        else:
            checksum.update(repr(value).encode())
    return checksum.hexdigest()


def upsample_probability(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(logits.float(), size=size, mode="bilinear", align_corners=False).softmax(1)


@torch.no_grad()
def retrieve_pair(
    query: torch.Tensor,
    bank: Dict[str, torch.Tensor],
    event_id: int,
    image_id: int,
    topk: int,
    metric_projector: torch.nn.Module,
):
    """Select row IDs first, then gather the unchanged source-bank payload."""
    cosine_ids = select_ids(query, bank, event_id, image_id, topk, projector=None)
    metric_ids = select_ids(query, bank, event_id, image_id, topk, projector=metric_projector)
    return cosine_ids, gather_support(cosine_ids, bank), metric_ids, gather_support(metric_ids, bank)


def decoder_forward(
    decoder: JointContextDecoder,
    arm: str,
    pixel_feature: torch.Tensor,
    result: Dict[str, torch.Tensor],
    cosine_support,
    metric_support,
):
    if arm not in ARMS:
        raise ValueError(arm)
    kwargs = {}
    decoder_variant = arm
    if arm == "j2":
        support, distribution, valid, metadata = cosine_support
        kwargs = dict(support=support, distribution=distribution, valid=valid, metadata=metadata)
    elif arm == "j3":
        support, distribution, valid, metadata = metric_support
        kwargs = dict(support=support, distribution=distribution, valid=valid, metadata=metadata)
        # J3 intentionally uses the exact J2 decoder computation.  Only the
        # retrieved row IDs (and therefore gathered support) may differ.
        decoder_variant = "j2"
    return decoder(
        pixel_feature,
        result["tokens"],
        result["q"],
        result["ids"],
        result["shape"],
        decoder_variant,
        **kwargs,
    )


def _new_retrieval_diagnostic() -> dict[str, torch.Tensor]:
    return {
        "query_count": torch.zeros(4, dtype=torch.long),
        "supported_query_count": torch.zeros(4, dtype=torch.long),
        "hit_at_k": torch.zeros(4, dtype=torch.long),
        "same_class_slots": torch.zeros(4, dtype=torch.long),
        "valid_slots": torch.zeros(4, dtype=torch.long),
        "neighbor_class_matrix": torch.zeros(4, 4, dtype=torch.long),
    }


def _update_retrieval_diagnostic(
    diagnostic,
    selected: torch.Tensor,
    bank_class: torch.Tensor,
    query_class: torch.Tensor,
    usable_query: torch.Tensor,
) -> None:
    selected_cpu = selected.detach().cpu()
    valid_slot = selected_cpu >= 0
    safe = selected_cpu.clamp_min(0).long()
    neighbor_class = bank_class[safe]
    for class_id in range(4):
        rows = usable_query & (query_class == class_id)
        diagnostic["query_count"][class_id] += int(rows.sum())
        if not bool(rows.any()):
            continue
        row_valid = valid_slot[rows]
        row_labels = neighbor_class[rows]
        same = row_valid & (row_labels == class_id)
        diagnostic["supported_query_count"][class_id] += int(row_valid.any(1).sum())
        diagnostic["hit_at_k"][class_id] += int(same.any(1).sum())
        diagnostic["same_class_slots"][class_id] += int(same.sum())
        diagnostic["valid_slots"][class_id] += int(row_valid.sum())
        for neighbor_id in range(4):
            diagnostic["neighbor_class_matrix"][class_id, neighbor_id] += int(
                (row_valid & (row_labels == neighbor_id)).sum()
            )


def _finalize_retrieval_diagnostic(diagnostic) -> dict:
    queries = diagnostic["query_count"]
    supported = diagnostic["supported_query_count"]
    slots = diagnostic["valid_slots"]
    return {
        "query_count": queries.tolist(),
        "supported_query_count": supported.tolist(),
        "supported_query_coverage": [
            None if int(queries[index]) == 0 else float(supported[index]) / int(queries[index])
            for index in range(4)
        ],
        "recall_at_k": [
            None
            if int(queries[index]) == 0
            else float(diagnostic["hit_at_k"][index]) / int(queries[index])
            for index in range(4)
        ],
        "same_class_slot_fraction": [
            None
            if int(slots[index]) == 0
            else float(diagnostic["same_class_slots"][index]) / int(slots[index])
            for index in range(4)
        ],
        "valid_slots": slots.tolist(),
        "neighbor_class_matrix": diagnostic["neighbor_class_matrix"].tolist(),
    }


@torch.no_grad()
def evaluate_joint(
    capture,
    decoders: Dict[str, JointContextDecoder],
    data,
    device: torch.device,
    bank: Dict[str, torch.Tensor],
    metric_projector: torch.nn.Module,
    topk: int,
) -> dict:
    if len(decoders) != len(ARMS) or set(decoders) != set(ARMS):
        raise ValueError(f"Expected exactly {ARMS}, received {tuple(decoders)}")
    for decoder in decoders.values():
        decoder.eval()
    metric_projector.eval()
    names = ("a1_pixel", "direct_region", *ARMS)
    confusions = {name: torch.zeros(4, 4, dtype=torch.long) for name in names}
    per_event = {name: {} for name in names}
    harmed = {name: 0 for name in ARMS}
    rescued = {name: 0 for name in ARMS}
    damaged_j2_to_j3 = torch.zeros(4, 4, dtype=torch.long)
    max_abs_to_a1 = {name: 0.0 for name in ARMS}
    retrieval = {"cosine": _new_retrieval_diagnostic(), "metric": _new_retrieval_diagnostic()}
    same_rank = torch.zeros(4, dtype=torch.long)
    comparable_rank = torch.zeros(4, dtype=torch.long)
    bank_class_cpu = bank["distribution"].detach().cpu().argmax(-1)
    image_count = 0

    for batch in data:
        pre, post, _loc, target, image_names = parse_batch(batch, device)
        if len(pre) != 1:
            raise ValueError("Joint Memory evaluation requires image batch size one")
        result, pixel_feature = capture(pre, post)
        event_name = event(image_names[0])
        event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
        cosine_ids, cosine_support, metric_ids, metric_support = retrieve_pair(
            result["tokens"][0], bank, event_id, -1, topk, metric_projector
        )
        probabilities = {
            "a1_pixel": result["old_pixel"].softmax(1),
            "direct_region": result["prob"],
        }
        for arm, decoder in decoders.items():
            decoded = decoder_forward(
                decoder, arm, pixel_feature, result, cosine_support, metric_support
            )
            probabilities[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
            max_abs_to_a1[arm] = max(
                max_abs_to_a1[arm],
                float((probabilities[arm] - probabilities["a1_pixel"]).abs().max()),
            )

        predictions = {name: probability.argmax(1) for name, probability in probabilities.items()}
        for name, prediction in predictions.items():
            confusion_update(confusions[name], target, prediction)
            per_event[name].setdefault(event_name, torch.zeros(4, 4, dtype=torch.long))
            confusion_update(per_event[name][event_name], target, prediction)
        damaged = target == 2
        for arm in ARMS:
            harmed[arm] += int(
                (damaged & (predictions["a1_pixel"] == 2) & (predictions[arm] == 1)).sum()
            )
            rescued[arm] += int(
                (damaged & (predictions["a1_pixel"] == 1) & (predictions[arm] == 2)).sum()
            )
        if bool(damaged.any()):
            transition = predictions["j2"][damaged] * 4 + predictions["j3"][damaged]
            damaged_j2_to_j3 += torch.bincount(
                transition.long(), minlength=16
            ).reshape(4, 4).cpu()

        # GT is introduced only here, after both neighbor selections, and is
        # used solely to diagnose retrieval—not to select or rank supports.
        query_distribution, query_weight = region_targets(target, result)
        query_distribution, query_weight = query_distribution[0], query_weight[0]
        query_valid = result["valid_regions"][0]
        query_class_cpu = query_distribution.detach().cpu().argmax(-1)
        usable_query_cpu = (query_weight.detach().cpu() > 0) & query_valid.detach().cpu()
        _update_retrieval_diagnostic(
            retrieval["cosine"], cosine_ids, bank_class_cpu, query_class_cpu, usable_query_cpu
        )
        _update_retrieval_diagnostic(
            retrieval["metric"], metric_ids, bank_class_cpu, query_class_cpu, usable_query_cpu
        )
        cosine_cpu, metric_cpu = cosine_ids.cpu(), metric_ids.cpu()
        comparable = (cosine_cpu >= 0) & (metric_cpu >= 0)
        for class_id in range(4):
            rows = usable_query_cpu & (query_class_cpu == class_id)
            same_rank[class_id] += int(((cosine_cpu == metric_cpu) & comparable)[rows].sum())
            comparable_rank[class_id] += int(comparable[rows].sum())

        image_count += 1
        if image_count % 25 == 0:
            print(f"evaluate joint123 {image_count}", flush=True)

    damaged_false_negative_by_prediction = damaged_j2_to_j3[:, [0, 1, 3]].sum(0)
    return {
        "metrics": {name: metrics(value) for name, value in confusions.items()},
        "per_event": {
            name: {event_name: metrics(value) for event_name, value in values.items()}
            for name, values in per_event.items()
        },
        "damaged_to_intact_vs_a1": {
            name: {"harm": harmed[name], "rescue": rescued[name]} for name in ARMS
        },
        "damaged_j3_vs_j2": {
            "harm_j2_correct_to_j3_intact": int(damaged_j2_to_j3[2, 1]),
            "rescue_j2_intact_to_j3_correct": int(damaged_j2_to_j3[1, 2]),
            "retained_correct": int(damaged_j2_to_j3[2, 2]),
            "j3_false_negative_total": int(damaged_false_negative_by_prediction.sum()),
            "j3_false_negative_by_prediction_background_intact_destroyed": (
                damaged_false_negative_by_prediction.tolist()
            ),
            "prediction_transition_matrix_j2_rows_j3_columns": damaged_j2_to_j3.tolist(),
        },
        "retrieval": {
            name: _finalize_retrieval_diagnostic(value) for name, value in retrieval.items()
        },
        "cosine_metric_same_neighbor_at_rank_fraction": [
            None
            if int(comparable_rank[index]) == 0
            else float(same_rank[index]) / int(comparable_rank[index])
            for index in range(4)
        ],
        "cosine_metric_comparable_slots": comparable_rank.tolist(),
        "max_abs_probability_to_a1": max_abs_to_a1,
        "image_count": image_count,
        "test_accessed": False,
    }


__all__ = [
    "ARMS",
    "ARM_MEANING",
    "JOINT_FORMAT",
    "bank_fingerprint",
    "decoder_forward",
    "evaluate_joint",
    "module_fingerprint",
    "retrieve_pair",
    "upsample_probability",
]
