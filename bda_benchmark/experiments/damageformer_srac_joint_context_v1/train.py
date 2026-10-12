"""Train matched J0/J1/J2 joint decoders on one frozen front and one data stream.

The immutable neighbor table was built for deterministic full Train images.
Accordingly this causal comparison deliberately uses full Train images without
random crop/flip/rotation.  Adding augmentation would require a new online
retrieval experiment and is not mixed into this test.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
    transfer_checked_batch,
    validate_batch_labels,
)
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll

from .common import (
    FORMAT,
    FrozenFrontCapture,
    digest,
    evaluate_all,
    lists,
    load_artifacts,
    read_cached_name,
    save_json,
    save_pt,
    seed_everything,
    smoke_train_names,
    source_hash,
    supports_from_table,
    upsample_probability,
)
from .model import JointContextDecoder, VARIANTS, VARIANT_MEANING


def parser():
    p = argparse.ArgumentParser(__doc__)
    for name in ("checkpoint", "cache", "neighbors", "a1_checkpoint", "dataset", "train_list", "val_list", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--retrieval_policy", choices=("global", "per_class"), default="global")
    p.add_argument("--exclusion", choices=("same_image", "cross_event"), default="cross_event")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    p.add_argument("--smoke", action="store_true")
    return p


def full_image_train_loader(dataset, names, workers, seed):
    data = MultimodalDamageAssessmentDatset(dataset, names, 1024, type="test")
    return DataLoader(
        data,
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


def validation_loader(dataset, names, workers):
    data = MultimodalDamageAssessmentDatset(dataset, names, 1024, type="test")
    return DataLoader(data, batch_size=1, shuffle=False, num_workers=workers)


def state(decoder, optimizer, step, score, manifest, variant):
    return dict(
        format=FORMAT,
        variant=variant,
        model=decoder.state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        valid_miou=score,
        **manifest,
    )


def main():
    args = parser().parse_args()
    if min(args.steps, args.eval_every, args.workers + 1, args.topk) < 1 or args.lr <= 0:
        raise SystemExit("Invalid training arguments")
    if not args.smoke and args.steps != 5000:
        raise SystemExit("Formal matched comparison requires exactly 5000 optimizer updates")
    if args.retrieval_policy == "per_class" and args.topk % 4:
        raise SystemExit("per_class topk must be divisible by four")
    if Path(args.output).exists():
        raise SystemExit("Output exists; preserve it and choose a new directory")
    device = torch.device(args.device)
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise SystemExit("BF16 unavailable")

    train_names, val_names = lists(args.train_list, args.val_list)
    seed_everything(args.seed)
    front, front_payload, cache_root, cache, bank, table, table_manifest = load_artifacts(args, device)
    used_train_names = smoke_train_names(train_names) if args.smoke else train_names
    used_val_names = val_names[:2] if args.smoke else val_names
    if len(used_train_names) != len(cache["shards"]):
        raise RuntimeError("Train/cache image count mismatch")
    # Check endpoints immediately; every sampled image is checked again below.
    for index in sorted({0, len(used_train_names) - 1}):
        if read_cached_name(cache_root, cache, index) != used_train_names[index]:
            raise RuntimeError("Cache order differs from Train list")

    capture = FrozenFrontCapture(front)
    template = JointContextDecoder(
        front.base.clf_2,
        dim=cache["config"]["dim"],
        num_events=len(cache["event_names"]),
        slots=args.topk,
    ).to(device)
    decoders = {name: copy.deepcopy(template).to(device) for name in VARIANTS}
    reference_state = template.state_dict()
    for decoder in decoders.values():
        for key, value in decoder.state_dict().items():
            torch.testing.assert_close(value, reference_state[key])
    optimizers = {
        name: torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=5e-3)
        for name, decoder in decoders.items()
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        arguments=vars(args),
        source_hash=source_hash(),
        front_sha256=digest(args.checkpoint),
        cache_manifest_sha256=digest(Path(args.cache) / "manifest.json"),
        bank_sha256=cache["bank_sha256"],
        neighbor_table_sha256=table_manifest["table_sha256"],
        a1_sha256=digest(args.a1_checkpoint),
        front_step=front_payload["step"],
        variants=VARIANT_MEANING,
        fixed="A1, all front modules, Q, region tokens, Memory bank and neighbor IDs",
        trainable="three independent but identically initialized joint decoders",
        identity_initialization="first 256 classifier columns and bias copy A1 clf_2; 128 context columns are zero",
        memory_reliability_gate="intentionally absent in V1; first isolate real Memory versus matched learned slots",
        training_data="deterministic full Train images without augmentation, required by immutable neighbor IDs",
        loss="projected pixel NLL + Lovasz-Softmax; no class reweighting/new Q loss/boundary loss",
        test_accessed=False,
        smoke=args.smoke,
    )
    save_json(output / "manifest.json", manifest)

    train_data = full_image_train_loader(args.dataset, used_train_names, args.workers, args.seed + 1)
    val_data = validation_loader(args.dataset, used_val_names, args.workers)
    initial = evaluate_all(
        capture, decoders, val_data, device, bank, args.topk, args.retrieval_policy, args.exclusion
    )
    a1_score = initial["metrics"]["a1_pixel"]["valid_miou"]
    identity = {
        name: initial["metrics"][name]["confusion"] == initial["metrics"]["a1_pixel"]["confusion"]
        for name in VARIANTS
    }
    probability_error = {
        name: initial["max_abs_probability_to_a1"][name] for name in VARIANTS
    }
    identity_tolerance = 1e-6
    if not all(identity.values()) or max(probability_error.values()) > identity_tolerance:
        raise RuntimeError("Step-zero A1 identity failed; no optimizer update performed")
    save_json(
        output / "initial_identity.json",
        dict(
            a1_valid_miou=a1_score,
            exact_confusion=identity,
            probability_max_abs=probability_error,
            probability_tolerance=identity_tolerance,
            **initial,
        ),
    )

    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax

    iterator = iter(train_data)
    name_to_index = {name: index for index, name in enumerate(used_train_names)}
    best = {name: -1.0 for name in VARIANTS}
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_data)
            batch = next(iterator)
        report = validate_batch_labels(batch, step, 0)
        save_json(output / "last_batch_labels.json", report)
        if not report["valid"]:
            raise ValueError("Invalid Train labels")
        pre, post, _loc, target, image_names = transfer_checked_batch(batch, device)
        image_name = str(image_names[0])
        if image_name not in name_to_index:
            raise RuntimeError("Unexpected Train image")
        image_index = name_to_index[image_name]
        if read_cached_name(cache_root, cache, image_index) != image_name:
            raise RuntimeError("Sample/cache identity mismatch")
        result, pixel_feature = capture(pre, post)
        support, distribution, valid, metadata = supports_from_table(
            table, table_manifest, image_index, result["tokens"].shape[1], bank, device
        )

        factor = min(1.0, step / min(500, max(1, args.steps // 10)))
        if factor >= 1.0:
            warmup = min(500, max(1, args.steps // 10))
            factor = 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup)))
        rows = {}
        for variant, decoder in decoders.items():
            decoder.train()
            optimizer = optimizers[variant]
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            kwargs = {}
            if variant == "j2":
                kwargs = dict(support=support, distribution=distribution, valid=valid, metadata=metadata)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                output_variant = decoder(
                    pixel_feature,
                    result["tokens"],
                    result["q"],
                    result["ids"],
                    result["shape"],
                    variant,
                    **kwargs,
                )
                probability = upsample_probability(output_variant["logits"], target.shape[-2:])
                if (target != 255).any():
                    loss = region_nll(probability, target)
                    loss = loss + lovasz_softmax(probability, target, classes="present", ignore=255)
                else:
                    loss = probability.sum() * 0
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            rows[variant] = dict(loss=float(loss.detach()), grad_norm=float(grad_norm))

        if step == 1 or step % 20 == 0:
            row = dict(step=step, image=image_name, lr=args.lr * factor, variants=rows)
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            evaluation = evaluate_all(
                capture, decoders, val_data, device, bank, args.topk, args.retrieval_policy, args.exclusion
            )
            with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(step=step, **evaluation)) + "\n")
            scores = {name: evaluation["metrics"][name]["valid_miou"] for name in VARIANTS}
            for variant in VARIANTS:
                payload = state(decoders[variant], optimizers[variant], step, scores[variant], manifest, variant)
                if scores[variant] > best[variant]:
                    best[variant] = scores[variant]
                    save_pt(output / f"{variant}_best.pth", payload)
                if step == args.steps:
                    save_pt(output / f"{variant}_last.pth", payload)
            print(f"VAL step={step} A1={evaluation['metrics']['a1_pixel']['valid_miou']:.6f} " +
                  " ".join(f"{name}={scores[name]:.6f}" for name in VARIANTS), flush=True)

    for variant in VARIANTS:
        for suffix in ("best", "last"):
            path = output / f"{variant}_{suffix}.pth"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload.update(run_complete=True, completed_steps=args.steps)
            save_pt(path, payload)
    save_json(
        output / "summary.json",
        dict(complete=True, best=best, a1_initial=a1_score, test_accessed=False, smoke=args.smoke),
    )
    capture.close()


if __name__ == "__main__":
    main()

