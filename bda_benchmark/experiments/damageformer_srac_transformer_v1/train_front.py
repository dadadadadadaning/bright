"""Stage A: freeze exact A1; train Transformer-aware Q and direct region path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
    transfer_checked_batch,
    validate_batch_labels,
)
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll

from .common import (
    FRONT_FORMAT,
    digest,
    evaluate_front,
    front_payload,
    lists,
    load_a1_strict,
    loader,
    new_output,
    save_json,
    save_pt,
    seed_everything,
    source_hash,
)
from .model import SRACTransformerFront


def module_fingerprint(module):
    checksum = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        checksum.update(name.encode())
        checksum.update(str(tensor.dtype).encode())
        checksum.update(str(tuple(tensor.shape)).encode())
        checksum.update(tensor.numpy().tobytes())
    return checksum.hexdigest()


def parser():
    argument_parser = argparse.ArgumentParser(__doc__)
    for name in ("a1_checkpoint", "dataset", "train_list", "val_list", "output"):
        argument_parser.add_argument("--" + name, required=True)
    for name, default in dict(
        steps=50000,
        eval_every=1000,
        batch=2,
        accum=8,
        workers=4,
        seed=3407,
        dim=128,
        decoder_width=64,
        stride=8,
        iterations=3,
    ).items():
        argument_parser.add_argument("--" + name, type=int, default=default)
    argument_parser.add_argument("--temperature", type=float, default=0.2)
    argument_parser.add_argument("--spatial_weight", type=float, default=0.1)
    argument_parser.add_argument("--building_weight", type=float, default=0.5)
    argument_parser.add_argument("--lr", type=float, default=1e-4)
    argument_parser.add_argument("--device", default="cuda:0")
    argument_parser.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    argument_parser.add_argument("--region_context", action="store_true", help="Separate ablation; off in primary run")
    argument_parser.add_argument("--smoke", action="store_true")
    return argument_parser


def main():
    args = parser().parse_args()
    if not Path(args.a1_checkpoint).is_file():
        raise SystemExit("A1 checkpoint not found")
    if min(args.steps, args.eval_every, args.batch, args.accum, args.dim, args.stride, args.iterations) < 1:
        raise SystemExit("Positive sizes required")
    if args.batch * args.accum != 16:
        raise SystemExit("Effective batch must be 16")
    if not args.smoke and args.steps != 50000:
        raise SystemExit("Formal Stage A requires exactly 50000 optimizer updates")
    if Path(args.output).exists():
        raise SystemExit("Output exists; preserve it and choose a new directory")
    train_names, val_names = lists(args.train_list, args.val_list)
    device = torch.device(args.device)
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise SystemExit("BF16 is unavailable on the selected device")

    seed_everything(args.seed)
    model = SRACTransformerFront(
        dim=args.dim,
        decoder_width=args.decoder_width,
        stride=args.stride,
        iterations=args.iterations,
        temperature=args.temperature,
        spatial_weight=args.spatial_weight,
        building_weight=args.building_weight,
        region_context=args.region_context,
    ).to(device)
    load_report = load_a1_strict(model, args.a1_checkpoint)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in model.base.parameters()):
        raise RuntimeError("Backbone freeze contract failed")

    seed_everything(args.seed + 1)
    train_data = loader(
        args.dataset,
        train_names,
        train=True,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed + 2,
    )
    validation_data = loader(
        args.dataset,
        val_names[:2] if args.smoke else val_names,
        workers=args.workers,
        seed=args.seed + 3,
    )
    output = new_output(args.output)
    manifest = dict(
        arguments=vars(args),
        format=FRONT_FORMAT,
        config=model.config,
        source_hash=source_hash(),
        a1_checkpoint=str(Path(args.a1_checkpoint).resolve()),
        a1_sha256=digest(args.a1_checkpoint),
        load_report=load_report,
        train_sha256=digest(args.train_list),
        val_sha256=digest(args.val_list),
        test_accessed=False,
        backbone="unchanged frozen DamageFormer/A1 including buffers",
        partition="RGB Swin-grid local cross-attention, 3x3 candidate seeds, iterative soft updates, predicted-building soft prior",
        structure="feature-level LayerScale adapter using predicted building probability, soft Q boundary and Q entropy; no raw-image dimming/re-encoding",
        output_policy="direct region logits projected by Q; A1 logits are diagnostic only and never added",
        memory="disabled in Stage A; trained as a matched frozen-front experiment later",
        loss="region probability NLL + Lovasz-Softmax; no class reweighting/boundary loss/Memory",
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        checkpoint_storage="adapter only; frozen 761MiB A1 is referenced by SHA256, not duplicated",
        smoke=args.smoke,
    )
    save_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)

    initial = evaluate_front(model, validation_data, device)
    observed = initial["metrics"]["a1_pixel"]["valid_miou"]
    reproduction_pass = None if args.smoke else abs(observed - load_report["expected_valid_miou"]) <= 1e-5
    save_json(
        output / "initial_a1_audit.json",
        dict(**initial, expected=load_report["expected_valid_miou"], reproduction_pass=reproduction_pass),
    )
    if reproduction_pass is False:
        raise RuntimeError(f"A1 baseline mismatch ({observed}); no optimizer update performed")

    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=5e-3)
    iterator = iter(train_data)
    best = -1.0
    frozen_before = module_fingerprint(model.base)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        warmup = min(1000, max(1, args.steps // 10))
        factor = step / warmup if step <= warmup else 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        logged_loss = logged_entropy = logged_scale = 0.0
        for microstep in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_data)
                batch = next(iterator)
            label_report = validate_batch_labels(batch, step, microstep)
            save_json(output / "last_batch_labels.json", label_report)
            if not label_report["valid"]:
                save_json(output / "invalid_labels.json", label_report)
                raise ValueError("Invalid CPU labels; no remapping was performed")
            pre, post, _loc, target, _names = transfer_checked_batch(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                result = model(pre, post)
                if (target != 255).any():
                    loss = region_nll(result["prob"], target)
                    loss = loss + lovasz_softmax(result["prob"], target, classes="present", ignore=255)
                else:
                    loss = result["prob"].sum() * 0
                scaled_loss = loss / args.accum
            if not torch.isfinite(scaled_loss):
                raise RuntimeError("Non-finite loss")
            scaled_loss.backward()
            logged_loss += float(loss.detach()) / args.accum
            logged_entropy += float(result["assignment_entropy"].detach()) / args.accum
            logged_scale += float(model.adapter.layer_scale.detach().abs().mean()) / args.accum
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 20 == 0:
            row = dict(
                step=step,
                crops=step * 16,
                region_loss=logged_loss,
                assignment_entropy=logged_entropy,
                adapter_scale=logged_scale,
                lr=optimizer.param_groups[0]["lr"],
                grad_norm=float(gradient_norm),
            )
            with (output / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate_front(model, validation_data, device)
            score = result["metrics"]["front_region"]["valid_miou"]
            with (output / "validation.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(step=step, **result)) + "\n")
            state = front_payload(model, optimizer, step, score, manifest)
            if score > best:
                best = score
                save_pt(output / "best.pth", state)
            if step == args.steps:
                save_pt(output / "last.pth", state)
            print(
                f"VAL step={step} direct_region={score:.6f} "
                f"A1={result['metrics']['a1_pixel']['valid_miou']:.6f}",
                flush=True,
            )

    if module_fingerprint(model.base) != frozen_before:
        raise RuntimeError("A frozen A1 parameter or buffer changed")
    for checkpoint_name in ("best.pth", "last.pth"):
        checkpoint = torch.load(output / checkpoint_name, map_location="cpu", weights_only=False)
        checkpoint.update(run_complete=True, completed_steps=args.steps)
        save_pt(output / checkpoint_name, checkpoint)
    save_json(
        output / "summary.json",
        dict(complete=True, best_valid_miou=best, a1_unchanged=True, test_accessed=False, smoke=args.smoke),
    )


if __name__ == "__main__":
    main()

