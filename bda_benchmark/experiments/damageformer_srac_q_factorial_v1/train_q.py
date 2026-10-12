"""Matched 2000-update Q fine-tuning from one passed legacy/native E0 front.

No A1 weight or buffer is updated.  All six arms start from the same E0 front,
use the same Train2918 order, optimizer schedule and final-step selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
    transfer_checked_batch, validate_batch_labels,
)
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.numeric import (
    configure_runtime, runtime_settings,
)
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.protocol import reproduction_for_scope
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    FRONT_FORMAT, digest, evaluate_front, lists, load_a1_strict, loader,
    new_output, save_json, save_pt, seed_everything,
    source_hash as inherited_source_hash,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.train_front import module_fingerprint

from .q_losses import auxiliary_q_loss
from .q_variants import ARMS, FORMAT, QVariantFront, q_source_hash


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, help="Passed full E0 protocol directory or protocol.json")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--output", required=True, help="Fresh output; never overwrite previous evidence")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux_weight", type=float, default=0.1)
    parser.add_argument("--cue_gain", type=float, choices=(0.1, 0.25), default=0.1)
    parser.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    parser.add_argument("--smoke", action="store_true", help="Two Val images; short Train run; diagnostic only")
    return parser


def _read_protocol(path: str) -> tuple[Path, dict]:
    source = Path(path)
    if source.is_dir():
        source = source / "protocol.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    if (record.get("format") != "SRAC_HARDQUERY_PROTOCOL_V1"
            or record.get("scope") != "all" or record.get("complete") is not True
            or record.get("smoke") is not False
            or record.get("full_validation_evidence") is not True
            or record.get("image_count") != 450
            or record.get("reproduction", {}).get("passed") is not True
            or record.get("test_accessed") is not False
            or record.get("profile") != "legacy"
            or record.get("pool_mode") != "native"):
        raise ValueError("Requires passed formal Val450 legacy/native E0; failed reports remain failed")
    return source, record


def _source_paths(protocol: dict) -> dict[str, Path]:
    names = ("front", "a1", "train_list", "val_list")
    paths = {key: Path(protocol["input_paths"][key]) for key in names}
    for key, path in paths.items():
        if not path.is_file() or digest(path) != protocol["input_fingerprints"][key]:
            raise ValueError(f"E0 source file changed: {key}")
    return paths


def _front_payload(path: Path, protocol: dict, paths: dict[str, Path]) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FRONT_FORMAT or payload.get("source_hash") != inherited_source_hash():
        raise ValueError("E0 front format or inherited code changed")
    if payload.get("a1_sha256") != digest(paths["a1"]):
        raise ValueError("Front and E0 A1 differ")
    for key in ("train", "val"):
        if payload.get(key + "_sha256") != digest(paths[key + "_list"]):
            raise ValueError(f"Front and E0 {key} split differ")
    if payload.get("smoke") is not False or payload.get("test_accessed") is not False:
        raise ValueError("Only a non-smoke Train/Val front is allowed")
    if protocol.get("source_format") == "SRAC_SNAPSHOT_JOINT123_V1":
        from bda_benchmark.experiments.damageformer_srac_snapshot_joint123_v1.common import snapshot_metadata
        snapshot_metadata(path)
    elif (payload.get("run_complete") is not True
          or payload.get("completed_steps") != payload.get("arguments", {}).get("steps")):
        raise ValueError("Formal non-snapshot front must be complete")
    if not isinstance(payload.get("adapter"), dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Front adapter/config missing")
    return payload


def _validate_formal_arguments(args: argparse.Namespace) -> None:
    if args.workers < 0 or args.steps < 1 or args.eval_every < 1:
        raise ValueError("Invalid positive Train sizes")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.aux_weight) or args.aux_weight < 0:
        raise ValueError("Invalid loss weight or learning rate")
    if args.batch * args.accum != 16:
        raise ValueError("All arms require effective batch 16")
    if not args.smoke and (
        args.steps != 2000 or args.eval_every != 2000 or args.batch != 2 or args.accum != 8
        or args.lr != 1e-4 or args.aux_weight != 0.1
    ):
        raise ValueError("Formal factorial locks 2000 matched updates, batch/order, LR and auxiliary weight")


def _check_inputs_unchanged(paths: dict[str, Path], protocol: dict, source: Path,
                            protocol_sha: str, initial_source_hash: str, initial_runtime: dict) -> dict:
    result = {
        "e0_files": all(digest(path) == protocol["input_fingerprints"][key] for key, path in paths.items()),
        "protocol": digest(source) == protocol_sha,
        "q_code": q_source_hash() == initial_source_hash,
        "runtime": runtime_settings() == initial_runtime,
    }
    if not all(result.values()):
        raise RuntimeError("Q training provenance/runtime changed: " + str(result))
    return result


def run(args: argparse.Namespace) -> dict:
    _validate_formal_arguments(args)
    protocol_path, protocol = _read_protocol(args.protocol)
    paths = _source_paths(protocol)
    initial_payload = _front_payload(paths["front"], protocol, paths)
    train_names, val_names = lists(paths["train_list"], paths["val_list"])
    device = torch.device(protocol["arguments"]["device"])
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 is unavailable on the E0 device")

    overrides = protocol.get("numeric_overrides", {})
    configure_runtime(protocol["profile"], **overrides)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    initial_runtime = runtime_settings()
    if initial_runtime != protocol["runtime_settings"]:
        raise ValueError("Current runtime differs from passed E0; use a fresh matching process")

    seed = int(protocol["arguments"]["seed"])
    seed_everything(seed)
    model = QVariantFront(arm=args.arm, cue_gain=args.cue_gain, **initial_payload["config"]).to(device)
    load_a1_strict(model, paths["a1"])
    model.load_original_adapter(initial_payload["adapter"])
    model.base.requires_grad_(False).eval()
    # The intervention is Q itself. Keep the downstream token construction and
    # classifier fixed so a candidate cannot win by refitting the region head.
    fixed_downstream = {
        "adapter": model.adapter,
        "embedding": model.embedding,
        "region_classifier": model.region_classifier,
    }
    for module in fixed_downstream.values():
        module.requires_grad_(False).eval()
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    trainable = [value for value in model.parameters() if value.requires_grad]
    if (not trainable or any(value.requires_grad for value in model.base.parameters())
            or any(value.requires_grad for module in fixed_downstream.values()
                   for value in module.parameters())
            or any(not name.startswith(("decoder.", "partition."))
                   for name in trainable_names)):
        raise RuntimeError("Q-only trainable parameter contract failed")
    frozen_before = module_fingerprint(model.base)
    downstream_before = {name: module_fingerprint(module)
                         for name, module in fixed_downstream.items()}

    seed_everything(seed + 1)
    train_data = loader(protocol["arguments"]["dataset"], train_names, train=True,
                        batch=args.batch, workers=args.workers, seed=seed + 2)
    validation_data = loader(protocol["arguments"]["dataset"],
                             val_names[:2] if args.smoke else val_names,
                             workers=args.workers, seed=seed + 3)
    output = new_output(args.output)
    protocol_sha = digest(protocol_path)
    source_hash = q_source_hash()
    manifest = dict(
        format=FORMAT, arm=args.arm, arguments=vars(args), config=model.config,
        cue_gain=args.cue_gain, source_hash=source_hash,
        inherited_source_hash=inherited_source_hash(),
        protocol_path=str(protocol_path.resolve()), protocol_sha256=protocol_sha,
        front_checkpoint=str(paths["front"].resolve()), front_sha256=digest(paths["front"]),
        a1_checkpoint=str(paths["a1"].resolve()), a1_sha256=digest(paths["a1"]),
        train_sha256=digest(paths["train_list"]), val_sha256=digest(paths["val_list"]),
        initial_front_step=initial_payload.get("step"),
        initial_front_complete=initial_payload.get("run_complete") is True,
        source_format=protocol.get("source_format"),
        e0_profile=protocol["profile"], e0_pool_mode=protocol["pool_mode"],
        e0_runtime=initial_runtime, frozen_a1=True, test_accessed=False,
        training="matched 2000-update continuation from the exact E0 front; no fresh 50000-step claim",
        best_checkpoint="diagnostic only; formal evaluation must use last.pth",
        region_loss="original region NLL + Lovasz-Softmax",
        auxiliary_loss="Train-only Q label reconstruction" if args.arm in (
            "building_reconstruction", "semantic_reconstruction") else "disabled",
        trainable_scope="Q decoder and partition only; A1, token adapter, embedding and region classifier fixed",
        trainable_parameters=sum(value.numel() for value in trainable), smoke=args.smoke,
    )
    save_json(output / "manifest.json", manifest)
    save_json(output / "summary.json", dict(complete=False, phase="initial_Val", arm=args.arm,
                                           test_accessed=False, smoke=args.smoke))
    current_phase = "initial_Val"
    try:
        initial = evaluate_front(model, validation_data, device)
        observed = initial["metrics"]["a1_pixel"]["valid_miou"]
        expected = protocol["metrics"]["a1_pixel"]["valid_miou"]
        a1_initial_report = reproduction_for_scope(
            {"a1_pixel": initial["metrics"]["a1_pixel"]},
            {"a1_pixel": protocol["metrics"]["a1_pixel"]}, "a1",
            smoke=args.smoke, image_count=initial["image_count"])
        reproduction = a1_initial_report.get("passed")
        save_json(output / "initial_a1_audit.json", dict(
            image_count=initial["image_count"], observed=observed, expected=expected,
            absolute_error=abs(observed - expected), tolerance=1e-5, passed=reproduction,
            class_and_summary_reproduction=a1_initial_report,
            smoke=args.smoke, original_region=initial["metrics"]["front_region"],
        ))
        if reproduction is False:
            raise RuntimeError("E0 A1 reproduction failed before any optimizer update; 1e-5 unchanged")
        if module_fingerprint(model.base) != frozen_before:
            raise RuntimeError("A1 weights/buffers changed during initial replay")
        if any(module_fingerprint(module) != downstream_before[name]
               for name, module in fixed_downstream.items()):
            raise RuntimeError("Downstream region path changed during initial replay")

        from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax

        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=5e-3)
        iterator = iter(train_data)
        best = -1.0
        current_phase = "Train"
        for step in range(1, args.steps + 1):
            model.train()
            for module in fixed_downstream.values():
                module.eval()
            optimizer.zero_grad(set_to_none=True)
            warmup = min(1000, max(1, args.steps // 10))
            factor = step / warmup if step <= warmup else 0.5 * (
                1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            logged_region = logged_aux = logged_entropy = logged_cue = logged_base = 0.0
            for microstep in range(args.accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_data)
                    batch = next(iterator)
                label_report = validate_batch_labels(batch, step, microstep)
                if step == 1 and microstep == 0:
                    save_json(output / "first_batch_labels.json", label_report)
                if not label_report["valid"]:
                    save_json(output / "invalid_labels.json", label_report)
                    raise ValueError("Invalid CPU labels; no remapping was performed")
                pre, post, _loc, target, _names = transfer_checked_batch(batch, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                    result = model(pre, post)
                    if (target != 255).any():
                        region_loss = region_nll(result["prob"], target)
                        region_loss = region_loss + lovasz_softmax(
                            result["prob"], target, classes="present", ignore=255)
                    else:
                        region_loss = result["prob"].sum() * 0
                    auxiliary, _details = auxiliary_q_loss(args.arm, result, target)
                    loss = region_loss + args.aux_weight * auxiliary
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite Q training loss")
                (loss / args.accum).backward()
                logged_region += float(region_loss.detach()) / args.accum
                logged_aux += float(auxiliary.detach()) / args.accum
                logged_entropy += float(result["assignment_entropy"].detach()) / args.accum
                if "cue_score_mean_abs" in result:
                    logged_cue += float(result["cue_score_mean_abs"]) / args.accum
                    logged_base += float(result["base_score_mean_abs"]) / args.accum
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 20 == 0 or step == args.steps:
                row = dict(step=step, crops=step * args.batch * args.accum,
                           region_loss=logged_region, auxiliary_q_loss=logged_aux,
                           assignment_entropy=logged_entropy,
                           cue_score_mean_abs=logged_cue if args.arm in ("sar_cue", "rgb_capacity", "shifted_sar") else None,
                           base_score_mean_abs=logged_base if args.arm in ("sar_cue", "rgb_capacity", "shifted_sar") else None,
                           lr=optimizer.param_groups[0]["lr"], grad_norm=float(gradient_norm))
                with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                current_phase = f"Val_step_{step}"
                result = evaluate_front(model, validation_data, device)
                if not args.smoke and result["image_count"] != 450:
                    raise RuntimeError("Formal Q evaluation requires all 450 Val images")
                a1_val = result["metrics"]["a1_pixel"]["valid_miou"]
                final_a1_report = reproduction_for_scope(
                    {"a1_pixel": result["metrics"]["a1_pixel"]},
                    {"a1_pixel": protocol["metrics"]["a1_pixel"]}, "a1",
                    smoke=args.smoke, image_count=result["image_count"])
                if final_a1_report.get("passed") is False:
                    raise RuntimeError("Frozen A1 class-level replay changed after Q training")
                score = result["metrics"]["front_region"]["valid_miou"]
                with (output / "validation.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(step=step, **result), allow_nan=False) + "\n")
                state = dict(adapter=model.adapter_state_dict(), optimizer=optimizer.state_dict(),
                             step=step, valid_miou=score,
                             initial_a1_reproduction=reproduction,
                             final_a1_reproduction=final_a1_report,
                             full_validation_evidence=bool(not args.smoke and result["image_count"] == 450),
                             **manifest)
                if score > best:
                    best = score
                    save_pt(output / "best.pth", state)
                if step == args.steps:
                    save_pt(output / "last.pth", state)
                print(f"Q {args.arm} Val step={step}: region={score:.6f}, A1={a1_val:.6f}", flush=True)
                current_phase = "Train"

        if module_fingerprint(model.base) != frozen_before:
            raise RuntimeError("Frozen A1 parameter or buffer changed")
        if any(module_fingerprint(module) != downstream_before[name]
               for name, module in fixed_downstream.items()):
            raise RuntimeError("Frozen downstream region parameter or buffer changed")
        immutable = _check_inputs_unchanged(paths, protocol, protocol_path, protocol_sha,
                                            source_hash, initial_runtime)
        immutable["frozen_downstream"] = True
        path = output / "last.pth"
        state = torch.load(path, map_location="cpu", weights_only=False)
        state.update(run_complete=True, completed_steps=args.steps,
                     a1_unchanged=True, downstream_unchanged=True,
                     immutable=immutable)
        save_pt(path, state)
        summary = dict(format=FORMAT, complete=True, arm=args.arm, steps=args.steps,
                       initial_a1_reproduction=reproduction,
                       full_validation_evidence=not args.smoke,
                       best_valid_miou=best, a1_unchanged=True,
                       downstream_unchanged=True, immutable=immutable,
                       smoke=args.smoke, test_accessed=False,
                       formal_checkpoint="last.pth")
        save_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        save_json(output / "summary.json", dict(format=FORMAT, complete=False,
                   arm=args.arm, phase=current_phase, error=f"{type(error).__name__}: {error}",
                   smoke=args.smoke, test_accessed=False))
        raise


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
