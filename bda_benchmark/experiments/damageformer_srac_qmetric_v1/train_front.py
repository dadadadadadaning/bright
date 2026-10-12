"""Train the isolated beta-one control or Q-annealing front from the same A1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_transformer_v1 import common as parent_common

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
    transfer_checked_batch,
    validate_batch_labels,
)
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll

from .common import (
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
from .model import AnnealedFront, FRONT_FORMAT, VARIANTS


Q_LOG_FIELDS = (
    "assignment_entropy",
    "q_logit_scale_exp",
    "q_similarity_coefficient",
    "q_top1_mean",
    "q_entropy_valid_mean",
    "q_valid_candidates_mean",
    "q_soft_occupied_mean",
    "q_hard_occupied_mean",
    "q_region_slots",
    "adapter_delta_norm",
    "adapter_base_norm",
    "adapter_delta_base_ratio",
)


def module_fingerprint(module) -> str:
    checksum = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        checksum.update(name.encode())
        checksum.update(str(tensor.dtype).encode())
        checksum.update(str(tuple(tensor.shape)).encode())
        checksum.update(tensor.numpy().tobytes())
    return checksum.hexdigest()


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(__doc__)
    for name in ("a1_checkpoint", "dataset", "train_list", "val_list", "output"):
        argument_parser.add_argument("--" + name, required=True)
    argument_parser.add_argument("--variant", choices=VARIANTS, required=True)
    argument_parser.add_argument(
        "--reference_run",
        help="Optional existing parent front directory whose manifest must match this protocol",
    )
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
        q_beta_hold=1000,
        q_beta_ramp_end=10000,
    ).items():
        argument_parser.add_argument("--" + name, type=int, default=default)
    argument_parser.add_argument("--temperature", type=float, default=0.2)
    argument_parser.add_argument("--spatial_weight", type=float, default=0.1)
    argument_parser.add_argument("--building_weight", type=float, default=0.5)
    argument_parser.add_argument("--q_beta_start", type=float, default=1.0)
    argument_parser.add_argument("--q_beta_end", type=float, default=2.0)
    argument_parser.add_argument("--lr", type=float, default=1e-4)
    argument_parser.add_argument("--device", default="cuda:0")
    argument_parser.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    argument_parser.add_argument("--region_context", action="store_true", help="Separate ablation; off in primary run")
    argument_parser.add_argument("--smoke", action="store_true")
    return argument_parser


def checkpoint_payload(model, optimizer, step, score, manifest):
    payload = front_payload(model, optimizer, step, score, manifest)
    payload.update(
        q_variant=model.variant,
        q_beta_current=model.current_beta,
        q_beta_schedule={
            "start": model.q_beta_start,
            "end": model.q_beta_end,
            "hold": model.q_beta_hold,
            "ramp_end": model.q_beta_ramp_end,
        },
    )
    return payload


def audit_parent_reference(path, args, model) -> dict:
    """Verify that an existing/running parent run is a legitimate control.

    Completion is intentionally not required: the manifest is written before
    parent optimization starts and is enough to freeze its protocol.  No parent
    adapter weights are loaded or continued.
    """
    reference = Path(path)
    manifest_path = reference / "manifest.json" if reference.is_dir() else reference
    if not manifest_path.is_file():
        report = {"passed": False, "manifest": str(manifest_path), "mismatches": ["manifest_missing"]}
        print(json.dumps({"reference_audit": report}, indent=2), flush=True)
        raise ValueError("Reference control manifest is missing")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    inherited_config = {
        name: value
        for name, value in model.config.items()
        if name not in {"variant", "q_beta_start", "q_beta_end", "q_beta_hold", "q_beta_ramp_end"}
    }
    expected_arguments = {
        name: getattr(args, name)
        for name in ("steps", "eval_every", "batch", "accum", "seed", "lr", "amp")
    }
    checks = {
        "format": payload.get("format") == parent_common.FRONT_FORMAT,
        "source_hash": payload.get("source_hash") == parent_common.source_hash(),
        "a1_sha256": payload.get("a1_sha256") == digest(args.a1_checkpoint),
        "train_sha256": payload.get("train_sha256") == digest(args.train_list),
        "val_sha256": payload.get("val_sha256") == digest(args.val_list),
        "base_config": payload.get("config") == inherited_config,
    }
    observed_arguments = payload.get("arguments", {})
    for name, expected in expected_arguments.items():
        checks["argument_" + name] = observed_arguments.get(name) == expected
    mismatches = [name for name, passed in checks.items() if not passed]
    report = {
        "passed": not mismatches,
        "manifest": str(manifest_path.resolve()),
        "reference_complete": bool(payload.get("run_complete", False)),
        "checks": checks,
        "mismatches": mismatches,
        "weights_reused": False,
    }
    print(json.dumps({"reference_audit": report}, indent=2), flush=True)
    if mismatches:
        raise ValueError("Reference control protocol mismatch: " + ", ".join(mismatches))
    return report


def main() -> None:
    args = parser().parse_args()
    if not Path(args.a1_checkpoint).is_file():
        raise SystemExit("A1 checkpoint not found")
    if min(args.steps, args.eval_every, args.batch, args.accum, args.dim, args.stride, args.iterations) < 1:
        raise SystemExit("Positive sizes required")
    if args.batch * args.accum != 16:
        raise SystemExit("Effective batch must be 16")
    if args.q_beta_start <= 0 or args.q_beta_end <= 0:
        raise SystemExit("Q beta must be positive")
    if args.q_beta_hold < 0 or args.q_beta_ramp_end <= args.q_beta_hold:
        raise SystemExit("Invalid Q-beta schedule boundaries")
    if not args.smoke:
        if args.steps != 50000:
            raise SystemExit("Formal Q experiment requires exactly 50000 optimizer updates")
        expected = (args.q_beta_start, args.q_beta_end, args.q_beta_hold, args.q_beta_ramp_end)
        if expected != (1.0, 2.0, 1000, 10000):
            raise SystemExit("Formal comparison requires beta 1->2, hold 1000, ramp end 10000")
    if Path(args.output).exists():
        raise SystemExit("Output exists; preserve it and choose a new directory")

    train_names, val_names = lists(args.train_list, args.val_list)
    device = torch.device(args.device)
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise SystemExit("BF16 is unavailable on the selected device")

    # Both variants consume exactly the same initialization RNG.  The control
    # simply sets the effective end beta to one.
    effective_beta_end = 1.0 if args.variant == "control" else args.q_beta_end
    seed_everything(args.seed)
    model = AnnealedFront(
        dim=args.dim,
        decoder_width=args.decoder_width,
        stride=args.stride,
        iterations=args.iterations,
        temperature=args.temperature,
        spatial_weight=args.spatial_weight,
        building_weight=args.building_weight,
        region_context=args.region_context,
        variant=args.variant,
        q_beta_start=args.q_beta_start,
        q_beta_end=effective_beta_end,
        q_beta_hold=args.q_beta_hold,
        q_beta_ramp_end=args.q_beta_ramp_end,
    ).to(device)
    model.set_step(0)
    load_report = load_a1_strict(model, args.a1_checkpoint)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in model.base.parameters()):
        raise RuntimeError("Backbone freeze contract failed")
    reference_audit = (
        audit_parent_reference(args.reference_run, args, model)
        if args.reference_run
        else {
            "passed": None,
            "manifest": None,
            "mismatches": [],
            "weights_reused": False,
            "note": "No external control claimed; run the internal beta=1 control with the same command protocol",
        }
    )

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
        controlled_factor="only the full candidate-logit beta before softmax in every one of the three Q iterations",
        q_beta_policy=(
            "constant beta=1 parent reproduction"
            if args.variant == "control"
            else "beta=1 through step 1000, linear ramp to beta=2 at step 10000, then constant"
        ),
        q_consistency="the same final Q is used for boundary, pooling/statistics and probability projection",
        prohibited_changes="no entropy loss, hard argmax, Memory, new head, class reweighting or backbone change",
        loss="region probability NLL + Lovasz-Softmax",
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        checkpoint_storage="adapter only; beta config and current persistent beta are serialized",
        reference_control_audit=reference_audit,
        smoke=args.smoke,
    )
    save_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)

    initial = evaluate_front(model, validation_data, device)
    observed = initial["metrics"]["a1_pixel"]["valid_miou"]
    reproduction_pass = None if args.smoke else abs(observed - load_report["expected_valid_miou"]) <= 1e-5
    save_json(
        output / "initial_a1_audit.json",
        dict(
            **initial,
            expected=load_report["expected_valid_miou"],
            reproduction_pass=reproduction_pass,
            q_beta_current=model.current_beta,
        ),
    )
    if reproduction_pass is False:
        raise RuntimeError(f"A1 baseline mismatch ({observed}); no optimizer update performed")

    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=5e-3)
    iterator = iter(train_data)
    best = -1.0
    frozen_before = module_fingerprint(model.base)
    for step in range(1, args.steps + 1):
        model.set_step(step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        warmup = min(1000, max(1, args.steps // 10))
        factor = step / warmup if step <= warmup else 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        logged_loss = 0.0
        logged = {name: 0.0 for name in Q_LOG_FIELDS}
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
            for name in Q_LOG_FIELDS:
                logged[name] += float(result[name].detach()) / args.accum
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 20 == 0:
            row = dict(
                step=step,
                crops=step * 16,
                variant=args.variant,
                q_beta=model.current_beta,
                region_loss=logged_loss,
                lr=optimizer.param_groups[0]["lr"],
                grad_norm=float(gradient_norm),
                **logged,
            )
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate_front(model, validation_data, device)
            score = result["metrics"]["front_region"]["valid_miou"]
            with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(step=step, q_beta=model.current_beta, **result)) + "\n")
            state = checkpoint_payload(model, optimizer, step, score, manifest)
            if score > best:
                best = score
                save_pt(output / "best.pth", state)
            if step == args.steps:
                save_pt(output / "last.pth", state)
            print(
                f"VAL step={step} variant={args.variant} beta={model.current_beta:.6f} "
                f"direct_region={score:.6f} A1={result['metrics']['a1_pixel']['valid_miou']:.6f}",
                flush=True,
            )

    if module_fingerprint(model.base) != frozen_before:
        raise RuntimeError("A frozen A1 parameter or buffer changed")
    for checkpoint_name in ("best.pth", "last.pth"):
        checkpoint = torch.load(output / checkpoint_name, map_location="cpu", weights_only=False)
        checkpoint.update(
            run_complete=True,
            completed_steps=args.steps,
            q_beta_current=float(checkpoint["adapter"]["partition.q_beta"]),
        )
        save_pt(output / checkpoint_name, checkpoint)
    save_json(
        output / "summary.json",
        dict(
            complete=True,
            variant=args.variant,
            best_valid_miou=best,
            final_beta=model.current_beta,
            a1_unchanged=True,
            test_accessed=False,
            smoke=args.smoke,
        ),
    )


if __name__ == "__main__":
    main()
