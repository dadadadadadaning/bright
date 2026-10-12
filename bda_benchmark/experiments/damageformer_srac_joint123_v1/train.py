"""Train only J1/J2/J3 on one frozen, completed formal front and Memory bank."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
    transfer_checked_batch,
    validate_batch_labels,
)
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import (
    FrozenFrontCapture,
)
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import (
    JointContextDecoder,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    FRONT_FORMAT as ORIGINAL_FRONT_FORMAT,
)

from .common import (
    device_bank,
    digest,
    lists,
    load_front,
    load_metric,
    read_cache,
    save_json,
    save_pt,
    seed_everything,
    source_hash,
)
from .joint_runtime import (
    ARMS,
    ARM_MEANING,
    JOINT_FORMAT,
    bank_fingerprint,
    decoder_forward,
    evaluate_joint,
    module_fingerprint,
    retrieve_pair,
    upsample_probability,
)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(__doc__)
    for name in (
        "checkpoint",
        "cache",
        "a1_checkpoint",
        "metric_checkpoint",
        "dataset",
        "train_list",
        "val_list",
        "output",
    ):
        argument_parser.add_argument("--" + name, required=True)
    argument_parser.add_argument("--steps", type=int, default=5000)
    argument_parser.add_argument("--eval_every", type=int, default=500)
    argument_parser.add_argument("--workers", type=int, default=4)
    argument_parser.add_argument("--seed", type=int, default=3407)
    argument_parser.add_argument("--lr", type=float, default=1e-4)
    argument_parser.add_argument("--topk", type=int, default=8)
    argument_parser.add_argument("--device", default="cuda:0")
    argument_parser.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    argument_parser.add_argument(
        "--probe",
        action="store_true",
        help="20-update diagnostic using formal artifacts, four Train images and two Val images",
    )
    return argument_parser


def full_image_loader(dataset, names, train, workers, seed):
    from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset

    data = MultimodalDamageAssessmentDatset(dataset, names, 1024, type="test")
    return DataLoader(
        data,
        batch_size=1,
        shuffle=train,
        num_workers=workers,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


def cached_shard(cache_root, cache, index, verify_hash=False):
    record = cache["shards"][index]
    path = cache_root / record["path"]
    if verify_hash and digest(path) != record["sha256"]:
        raise ValueError("Cache shard hash mismatch: " + record["path"])
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_formal_cache(cache_root, cache, train_names):
    """Validate every source shard against the full ordered Train split."""
    if bool(cache.get("smoke")):
        raise ValueError("J1/J2/J3 requires a formal full-Train Memory cache")
    if len(cache["shards"]) != len(train_names):
        raise ValueError("Formal Train/cache image count mismatch")
    identity = []
    for index, expected_name in enumerate(train_names):
        shard = cached_shard(cache_root, cache, index, verify_hash=True)
        if str(shard["name"]) != str(expected_name):
            raise ValueError(f"Cache order mismatch at image {index}")
        identity.append(
            {
                "name": str(shard["name"]),
                "event": int(shard["event"]),
                "image": int(shard["image"]),
            }
        )
    return identity


def validate_formal_front_payload(payload, train_sha256, val_sha256):
    """Reject Q experiments and incomplete/smoke fronts even on direct CLI use."""
    if payload.get("format") != ORIGINAL_FRONT_FORMAT:
        raise ValueError("J1/J2/J3 isolates context heads and rejects Q-annealed fronts")
    arguments = payload.get("arguments", {})
    if (
        not payload.get("run_complete")
        or payload.get("completed_steps") != arguments.get("steps")
        or bool(payload.get("smoke"))
        or payload.get("test_accessed") is not False
    ):
        raise ValueError("J1/J2/J3 requires a completed formal Train/Val-only front")
    if payload.get("train_sha256") != train_sha256 or payload.get("val_sha256") != val_sha256:
        raise ValueError("Front Train/Val split hashes do not match this joint run")


def probe_train_names(cache_identity, limit=4):
    """Select the first four different-event images from validated cache metadata."""
    chosen, seen = [], set()
    for record in cache_identity:
        event_id = int(record["event"])
        if event_id not in seen:
            chosen.append(record["name"])
            seen.add(event_id)
        if len(chosen) == limit:
            break
    if len(chosen) != limit:
        raise ValueError(f"Probe needs {limit} distinct Train events; found {len(chosen)}")
    return chosen


def checkpoint_state(decoder, optimizer, step, score, manifest, arm):
    return {
        "format": JOINT_FORMAT,
        "arm": arm,
        "model": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "valid_miou": score,
        "run_complete": False,
        "reusable_as_parent": False,
        **manifest,
    }


def main() -> None:
    args = parser().parse_args()
    if min(args.steps, args.eval_every, args.workers + 1, args.topk) < 1 or args.lr <= 0:
        raise SystemExit("Invalid training arguments")
    if args.probe:
        if args.steps != 20 or args.eval_every != 20:
            raise SystemExit("Probe requires exactly --steps 20 --eval_every 20")
    elif args.steps != 5000 or args.eval_every != 500:
        raise SystemExit("Formal J1/J2/J3 comparison requires exactly 5000 updates and eval_every=500")
    if Path(args.output).exists():
        raise SystemExit("Output exists; preserve it and choose a new directory")
    device = torch.device(args.device)
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise SystemExit("BF16 unavailable")

    train_names, val_names = lists(args.train_list, args.val_list)
    seed_everything(args.seed)
    front, front_payload = load_front(args.checkpoint, device, args.a1_checkpoint)
    front.eval().requires_grad_(False)
    validate_formal_front_payload(
        front_payload, digest(args.train_list), digest(args.val_list)
    )

    cache_root, cache, bank_cpu = read_cache(args.cache, args.checkpoint)
    cache_identity = validate_formal_cache(cache_root, cache, train_names)
    bank = device_bank(bank_cpu, device)
    metric_projector, metric_payload = load_metric(
        args.metric_checkpoint, cache, args.checkpoint, device
    )
    metric_projector.eval().requires_grad_(False)
    if bool(metric_payload.get("smoke")):
        raise ValueError("J1/J2/J3 requires a completed formal retrieval metric")

    used_train_names = probe_train_names(cache_identity) if args.probe else train_names
    used_val_names = val_names[:2] if args.probe else val_names
    # This is deliberately indexed by the full cache, even in probe mode.
    name_to_index = {record["name"]: index for index, record in enumerate(cache_identity)}

    capture = FrozenFrontCapture(front)
    template = JointContextDecoder(
        front.base.clf_2,
        dim=cache["config"]["dim"],
        num_events=len(cache["event_names"]),
        slots=args.topk,
    ).to(device)
    decoders = {arm: copy.deepcopy(template).to(device) for arm in ARMS}
    reference_state = template.state_dict()
    for decoder in decoders.values():
        for name, value in decoder.state_dict().items():
            torch.testing.assert_close(value, reference_state[name])
    optimizers = {
        arm: torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=5e-3)
        for arm, decoder in decoders.items()
    }

    frozen_before = {
        "front": module_fingerprint(front),
        "metric": module_fingerprint(metric_projector),
        "bank": bank_fingerprint(bank),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "arguments": vars(args),
        "format": JOINT_FORMAT,
        "source_hash": source_hash(),
        "front_sha256": digest(args.checkpoint),
        "a1_sha256": digest(args.a1_checkpoint),
        "cache_manifest_sha256": digest(Path(args.cache) / "manifest.json"),
        "bank_sha256": cache["bank_sha256"],
        "metric_sha256": digest(args.metric_checkpoint),
        "metric_step": metric_payload.get("step"),
        "arms": ARM_MEANING,
        "fixed": (
            "complete formal front, A1, Q, embedding, building prior, source Memory bank "
            "and retrieval projector"
        ),
        "trainable": "three independent, identically initialized J1/J2/J3 joint decoders",
        "identity_initialization": (
            "A1 clf_2 pixel block copied exactly; context output block initialized to zero"
        ),
        "retrieval_difference": (
            "J2 and J3 differ only in cosine versus frozen learned-metric neighbor row IDs"
        ),
        "query_gt_role": (
            "Val GT is used only after neighbor selection for diagnostic Recall@K and class matrices"
        ),
        "training_data": (
            "deterministic full 1024 Train images, batch one, no augmentation"
            if not args.probe
            else "first four distinct-event Train images selected from validated formal cache metadata"
        ),
        "loss": "pixel NLL + Lovasz-Softmax; no class weighting, Q loss or boundary loss",
        "test_accessed": False,
        "probe": args.probe,
        "smoke": args.probe,
        "source_artifacts_smoke": False,
    }
    save_json(output / "manifest.json", manifest)

    train_data = full_image_loader(
        args.dataset, used_train_names, True, args.workers, args.seed + 1
    )
    val_data = full_image_loader(
        args.dataset, used_val_names, False, args.workers, args.seed + 2
    )
    initial = evaluate_joint(
        capture, decoders, val_data, device, bank, metric_projector, args.topk
    )
    identity = {
        arm: initial["metrics"][arm]["confusion"]
        == initial["metrics"]["a1_pixel"]["confusion"]
        for arm in ARMS
    }
    tolerance = 1e-6
    if not all(identity.values()) or max(initial["max_abs_probability_to_a1"].values()) > tolerance:
        raise RuntimeError("Step-zero A1 identity failed; no optimizer update performed")
    save_json(
        output / "initial_identity.json",
        {
            **initial,
            "exact_confusion": identity,
            "probability_tolerance": tolerance,
        },
    )

    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax

    iterator = iter(train_data)
    best = {arm: -1.0 for arm in ARMS}
    train_image_counts = {arm: 0 for arm in ARMS}
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_data)
            batch = next(iterator)
        label_report = validate_batch_labels(batch, step, 0)
        save_json(output / "last_batch_labels.json", label_report)
        if not label_report["valid"]:
            save_json(output / "invalid_labels.json", label_report)
            raise ValueError("Invalid Train labels")
        pre, post, _loc, target, image_names = transfer_checked_batch(batch, device)
        image_name = str(image_names[0])
        if image_name not in name_to_index:
            raise RuntimeError("Unexpected Train image")
        image_index = name_to_index[image_name]
        shard = cache_identity[image_index]
        if str(shard["name"]) != image_name:
            raise RuntimeError("Sample/cache identity mismatch")
        result, pixel_feature = capture(pre, post)
        event_id = int(shard["event"])
        stored_image_id = int(shard["image"])
        cosine_ids, cosine_support, metric_ids, metric_support = retrieve_pair(
            result["tokens"][0],
            bank,
            event_id,
            stored_image_id,
            args.topk,
            metric_projector,
        )

        warmup = min(500, max(1, args.steps // 10))
        factor = step / warmup if step <= warmup else 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))
        )
        rows = {}
        for arm, decoder in decoders.items():
            decoder.train()
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                decoded = decoder_forward(
                    decoder,
                    arm,
                    pixel_feature,
                    result,
                    cosine_support,
                    metric_support,
                )
                probability = upsample_probability(decoded["logits"], target.shape[-2:])
                if (target != 255).any():
                    loss = region_nll(probability, target)
                    loss = loss + lovasz_softmax(
                        probability, target, classes="present", ignore=255
                    )
                else:
                    loss = probability.sum() * 0
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            train_image_counts[arm] += 1
            rows[arm] = {
                "loss": float(loss.detach()),
                "grad_norm": float(gradient_norm),
            }

        if step == 1 or step % 20 == 0:
            row = {
                "step": step,
                "image": image_name,
                "lr": args.lr * factor,
                "cosine_valid_support_fraction": float((cosine_ids >= 0).float().mean()),
                "metric_valid_support_fraction": float((metric_ids >= 0).float().mean()),
                "arms": rows,
            }
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            evaluation = evaluate_joint(
                capture, decoders, val_data, device, bank, metric_projector, args.topk
            )
            with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **evaluation}) + "\n")
            scores = {arm: evaluation["metrics"][arm]["valid_miou"] for arm in ARMS}
            for arm in ARMS:
                state = checkpoint_state(
                    decoders[arm], optimizers[arm], step, scores[arm], manifest, arm
                )
                if scores[arm] > best[arm]:
                    best[arm] = scores[arm]
                    save_pt(output / f"{arm}_best.pth", state)
                if step == args.steps:
                    save_pt(output / f"{arm}_last.pth", state)
            print(
                f"VAL step={step} A1={evaluation['metrics']['a1_pixel']['valid_miou']:.6f} "
                + " ".join(f"{arm}={scores[arm]:.6f}" for arm in ARMS),
                flush=True,
            )

    frozen_after = {
        "front": module_fingerprint(front),
        "metric": module_fingerprint(metric_projector),
        "bank": bank_fingerprint(bank),
    }
    if frozen_after != frozen_before:
        raise RuntimeError("A frozen front, metric projector or Memory bank changed")
    for arm in ARMS:
        for suffix in ("best", "last"):
            path = output / f"{arm}_{suffix}.pth"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint.update(
                run_complete=not args.probe,
                reusable_as_parent=not args.probe,
                completed_steps=args.steps,
                probe=args.probe,
                smoke=args.probe,
            )
            save_pt(path, checkpoint)
    save_json(
        output / "summary.json",
        {
            "complete": True,
            "run_complete": not args.probe,
            "reusable_as_parent": not args.probe,
            "best": best,
            "a1_initial": initial["metrics"]["a1_pixel"]["valid_miou"],
            "train_image_counts": train_image_counts,
            "frozen_fingerprints_unchanged": True,
            "checkpoint_files": {
                arm: {suffix: f"{arm}_{suffix}.pth" for suffix in ("best", "last")}
                for arm in ARMS
            },
            "test_accessed": False,
            "probe": args.probe,
            "smoke": args.probe,
            "source_artifacts_smoke": False,
        },
    )
    capture.close()


if __name__ == "__main__":
    main()
