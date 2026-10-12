"""Shared provenance, data, checkpoint and evaluation utilities."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
    transfer_checked_batch as parse_batch,
)
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import event_name, seed_everything


FRONT_FORMAT = "SRAC_TRANSFORMER_FRONT_V1"
READER_FORMAT = "SRAC_TRANSFORMER_READER_V1"


def digest(path):
    checksum = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def source_hash():
    checksum = hashlib.sha256()
    repository = Path(__file__).resolve().parents[3]
    external = [
        "bda_benchmark/experiments/damageformer_joint_region_v1/model.py",
        "bda_benchmark/experiments/damageformer_iterative_partition_v1/model.py",
        "bda_benchmark/experiments/damageformer_a1_cross_modal_superpixel_v1/association.py",
        "bda_benchmark/experiments/damageformer_srac_fourstage_v2/a1_init.py",
        "bda_benchmark/experiments/damageformer_srac_fourstage_v2/common.py",
        "bda_benchmark/experiments/damageformer_srac_fourstage_v2/label_audit.py",
        "bda_benchmark/experiments/damageformer_srac_fourstage_v2/losses.py",
        "bda_benchmark/experiments/damageformer_srac_fourstage_v2/metadata.py",
        "bda_benchmark/experiments/damageformer_srac_predsplit_v1/audit.py",
        "bda_benchmark/experiments/p3_damageformer_ablation_v1/core.py",
        "bda_benchmark/model/DamageFormer_S8_MICB_DSMIL.py",
        "bda_benchmark/model/DamageFormer_S3_BGMP_K3.py",
        "bda_benchmark/dataset/make_data_loader.py",
        "bda_benchmark/dataset/imutils.py",
        "bda_benchmark/script/standard_ML/train_S5_S8_GPU.py",
    ]
    files = sorted(Path(__file__).parent.glob("*.py")) + [repository / name for name in external]
    for path in files:
        checksum.update(path.relative_to(repository).as_posix().encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def save_pt(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def worker_seed(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def event(name):
    return event_name(Path(str(name)).stem)


def lists(train_list, val_list):
    train_path, val_path = Path(train_list), Path(val_list)
    if train_path.name != "train_set.txt" or val_path.name != "val_set.txt":
        raise ValueError("Only standard Train2918/Val450 lists are allowed; Test is forbidden")
    train, val = (
        [line.strip() for line in path.read_text().splitlines() if line.strip()]
        for path in (train_path, val_path)
    )
    if len(train) != 2918 or len(val) != 450 or set(train) & set(val):
        raise ValueError("Expected disjoint Train2918 and Val450")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Duplicate sample names in split")
    return train, val


def loader(dataset, names, train=False, batch=1, workers=4, crop=512, seed=3407):
    from torch.utils.data import DataLoader
    from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset

    data = MultimodalDamageAssessmentDatset(
        dataset,
        names,
        crop if train else 1024,
        type="train" if train else "test",
    )
    return DataLoader(
        data,
        batch_size=batch,
        shuffle=train,
        num_workers=workers,
        drop_last=train,
        worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed),
    )


def load_a1_strict(model, path):
    from bda_benchmark.experiments.damageformer_srac_fourstage_v2.a1_init import load_a1

    report = load_a1(model, path)
    model.base.requires_grad_(False).eval()
    return report


def front_payload(model, optimizer, step, valid_miou, manifest):
    return dict(
        adapter=model.adapter_state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        valid_miou=valid_miou,
        **manifest,
    )


def load_front(path, device, a1_override=None):
    from .model import SRACTransformerFront

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FRONT_FORMAT:
        raise ValueError("Wrong front checkpoint format")
    if not payload.get("run_complete") or payload.get("completed_steps") != payload["arguments"]["steps"]:
        raise ValueError("Front training did not complete its declared budget")
    if payload.get("source_hash") != source_hash():
        raise ValueError("Source changed since front training; use its archived code")
    a1_path = Path(a1_override or payload["a1_checkpoint"])
    if not a1_path.is_file() or digest(a1_path) != payload["a1_sha256"]:
        raise ValueError("The exact audited A1 file required by this front is unavailable")
    model = SRACTransformerFront(**payload["config"]).to(device)
    load_a1_strict(model, a1_path)
    model.load_adapter_state_dict(payload["adapter"])
    model.base.requires_grad_(False).eval()
    return model, payload


def region_targets(gt, result):
    """Soft Train labels for regions; GT labels never alter Q."""
    valid = (gt >= 0) & (gt < 4)
    one_hot = F.one_hot(gt.clamp(0, 3), 4).permute(0, 3, 1, 2).float() * valid[:, None]
    fractions = F.interpolate(one_hot, size=result["shape"], mode="area")
    from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import soft_pool_unpool

    pooled = soft_pool_unpool(fractions, result["q"], result["ids"], result["tokens"].shape[1])
    counts = pooled["tokens"] * pooled["mass"][..., None]
    weight = counts.sum(-1)
    distribution = counts / weight.clamp_min(1e-8)[..., None]
    return distribution, weight


def read_cache(path, checkpoint):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("split") != "train" or manifest.get("test_accessed"):
        raise ValueError("A complete Train-only cache is required")
    if manifest["checkpoint_sha256"] != digest(checkpoint) or manifest["source_hash"] != source_hash():
        raise ValueError("Stale or incompatible cache")
    bank = torch.load(root / "bank.pt", map_location="cpu", weights_only=False)
    if digest(root / "bank.pt") != manifest["bank_sha256"]:
        raise ValueError("Memory bank hash mismatch")
    if bank["tokens"].ndim != 2 or bank["distribution"].shape != (len(bank["tokens"]), 4):
        raise ValueError("Invalid memory shapes")
    if not torch.isfinite(bank["tokens"]).all() or not torch.isfinite(bank["distribution"]).all():
        raise ValueError("Non-finite memory")
    return root, manifest, bank


def device_bank(bank, device):
    result = {}
    for key, value in bank.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if key in ("tokens", "distribution", "geometry"):
                value = value.float()
        result[key] = value
    return result


def confusion_update(confusion, target, prediction):
    valid = (target >= 0) & (target < 4)
    confusion += torch.bincount(
        (target[valid] * 4 + prediction[valid]).long(), minlength=16
    ).reshape(4, 4).cpu()


@torch.inference_mode()
def evaluate_front(
    model,
    data,
    device,
    reader=None,
    bank=None,
    mode="memory",
    topk=8,
    intervention="normal",
    seed=3407,
    retrieval_policy="global",
    exclusion="cross_event",
):
    from .model import project_region_probabilities
    from .retrieval import retrieve

    model.eval()
    if reader is not None:
        reader.eval()
    names = ("final", "front_region", "a1_pixel")
    confusions = {name: torch.zeros(4, 4, dtype=torch.long) for name in names}
    event_confusions = {}
    image_count = supported = query_count = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in data:
        pre, post, _loc, gt, image_names = parse_batch(batch, device)
        result = model(pre, post)
        final_probability = result["prob"]
        if reader is not None:
            if len(pre) != 1:
                raise ValueError("Memory evaluation requires batch size one")
            query = result["tokens"][0]
            outputs = []
            event_id = bank["event_names"].index(event(image_names[0])) if event(image_names[0]) in bank["event_names"] else -1
            for start in range(0, len(query), 256):
                chunk = query[start : start + 256]
                if mode == "nomemory":
                    logits = reader(chunk, mode="nomemory")
                else:
                    support, distribution, valid, metadata = retrieve(
                        chunk,
                        bank,
                        event_id,
                        -1,
                        topk,
                        intervention,
                        generator,
                        return_metadata=True,
                        policy=retrieval_policy,
                        exclusion=exclusion,
                    )
                    supported += int(valid.any(1).sum())
                    query_count += len(chunk)
                    logits = reader(
                        chunk,
                        support,
                        distribution,
                        valid,
                        mode="memory",
                        metadata=None if intervention == "no_metadata" else metadata,
                    )
                outputs.append(logits)
            final_probability = project_region_probabilities(
                torch.cat(outputs)[None],
                result["q"],
                result["ids"],
                result["shape"],
                gt.shape[-2:],
            )
        predictions = {
            "final": final_probability.argmax(1),
            "front_region": result["prob"].argmax(1),
            "a1_pixel": result["old_pixel"].argmax(1),
        }
        for name, prediction in predictions.items():
            confusion_update(confusions[name], gt, prediction)
        event_id = event(image_names[0])
        event_confusions.setdefault(event_id, torch.zeros(4, 4, dtype=torch.long))
        confusion_update(event_confusions[event_id], gt, predictions["final"])
        image_count += len(pre)
        if image_count % 25 == 0:
            print(f"evaluate {image_count}", flush=True)
    return dict(
        metrics={name: metrics(confusion) for name, confusion in confusions.items()},
        per_event={name: metrics(confusion) for name, confusion in event_confusions.items()},
        image_count=image_count,
        support_coverage=supported / max(1, query_count),
        test_accessed=False,
    )

