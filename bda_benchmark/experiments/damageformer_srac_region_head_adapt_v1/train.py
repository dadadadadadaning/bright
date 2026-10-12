"""Equal-budget P/R head fitting on frozen F/Q, with a secondary T head.

The full Train2918 image stream is shared by all three heads. The source
front, Q, Memory, projection and Reader are never optimized. Initial/final
Val450 audits preserve the original five-arm 1e-5 reproduction requirement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .core import FORMAT, FORMAL_STEPS, SMOKE_STEPS, TRAIN_ARMS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, help="Passed legacy/native all-path E0")
    p.add_argument("--output", required=True, help="Fresh fitting directory")
    p.add_argument("--steps", type=int, default=FORMAL_STEPS,
                   help="Formal runs require exactly 5000 updates per head")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--smoke", action="store_true",
                   help="20 shared updates and two Val images; never formal evidence")
    return p


def validate_training_config(steps, lr, smoke):
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError("Steps must be an integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if smoke:
        if steps not in (FORMAL_STEPS, SMOKE_STEPS):
            raise ValueError("Smoke uses exactly 20 updates per head")
        return SMOKE_STEPS
    if steps != FORMAL_STEPS:
        raise ValueError("Formal matched fitting requires exactly 5000 updates per head")
    return steps


def learning_rate_factor(step, steps):
    if not 1 <= step <= steps:
        raise ValueError("Learning-rate step is outside the update budget")
    warmup = min(1000, max(1, steps // 10))
    if step <= warmup:
        return step / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def validate_trainable_heads(heads, optimizers, frozen_modules=()):
    """Reject shared parameters or an optimizer that touches a frozen source."""
    if set(heads) != set(TRAIN_ARMS) or set(optimizers) != set(TRAIN_ARMS):
        raise ValueError("All three matched-stream heads and optimizers are required")
    source_ids = set()
    for module in frozen_modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise ValueError("A source parameter is trainable")
            source_ids.add(id(parameter))
    seen = set(source_ids)
    for arm in TRAIN_ARMS:
        params = list(heads[arm].parameters())
        ids = {id(p) for p in params}
        if not params or ids & seen:
            raise ValueError("Heads must own independent parameters outside frozen source")
        if any(not p.requires_grad or p.dtype != torch.float32 for p in params):
            raise ValueError("Every private head parameter must be trainable FP32")
        optimizer_params = [p for group in optimizers[arm].param_groups for p in group["params"]]
        if len(optimizer_params) != len(params) or {id(p) for p in optimizer_params} != ids:
            raise ValueError("Optimizer parameters do not match their private head")
        seen.update(ids)


class SharedTrainStream:
    """One loader iterator supplies each update once, then all heads see it.

    A cycle must exhaust all Train names without duplicates. The final cycle
    may be a prefix because the preregistered budget is in optimizer updates.
    """

    def __init__(self, loader, names):
        self.expected = set(names)
        if not self.expected or len(self.expected) != len(names):
            raise ValueError("Train image names must be nonempty and unique")
        self.loader = loader
        self.iterator = iter(loader)
        self.seen = set()
        self.epoch = 1

    def next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            if self.seen != self.expected:
                raise ValueError("Train loader skipped an image before its next cycle")
            self.epoch += 1
            self.seen = set()
            self.iterator = iter(self.loader)
            try:
                batch = next(self.iterator)
            except StopIteration as exc:
                raise ValueError("Train loader is empty") from exc
        names = [str(x) for x in batch[4]]
        if len(names) != 1 or names[0] not in self.expected or names[0] in self.seen:
            raise ValueError("Train image stream contains unexpected/duplicate names or batch size")
        self.seen.add(names[0])
        return batch, names[0], self.epoch


def supervised_loss(probability, target):
    from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll
    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax
    if not bool((target != 255).any()):
        return probability.sum() * 0.0
    return (region_nll(probability, target)
            + lovasz_softmax(probability, target, classes="present", ignore=255))


def matched_update(heads, optimizers, inputs, target, probability_fn=None, loss_fn=None):
    """Compute finite gradients for every arm before any optimizer is advanced."""
    from .core import head_probability
    probability_fn = probability_fn or head_probability
    loss_fn = loss_fn or supervised_loss
    if set(heads) != set(TRAIN_ARMS) or set(optimizers) != set(TRAIN_ARMS):
        raise ValueError("Matched update requires P/R/T")
    if any(value.requires_grad for value in inputs.values() if torch.is_tensor(value)):
        raise ValueError("Frozen features/Q must be detached before head fitting")
    for arm in TRAIN_ARMS:
        heads[arm].train()
        optimizers[arm].zero_grad(set_to_none=True)
    logs = {}
    with torch.enable_grad(), torch.autocast(target.device.type, enabled=False):
        for arm in TRAIN_ARMS:
            probability = probability_fn(arm, heads[arm], inputs, target.shape[-2:])
            if probability.dtype != torch.float32:
                raise ValueError("Head probabilities must be FP32")
            loss = loss_fn(probability, target)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite or nonscalar head loss: " + arm)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                heads[arm].parameters(), 1.0, error_if_nonfinite=True)
            logs[arm] = dict(loss=float(loss.detach()), grad_norm=float(norm))
            del probability, loss
    for arm in TRAIN_ARMS:
        optimizers[arm].step()
    return logs


def _checkpoint(heads, optimizers, bundle, cli, steps, source, complete=False):
    return dict(format=FORMAT, complete=complete, completed_steps=steps,
                smoke=bool(cli.smoke), test_accessed=False,
                protocol_sha256=bundle["protocol_sha256"], source_sha256=source,
                heads=heads.state_dict(),
                optimizer_states={arm: optimizers[arm].state_dict() for arm in TRAIN_ARMS},
                arguments=vars(cli))


def run(cli):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, new_output, verify_unchanged, write_json,
    )
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import save_pt
    from .core import (assert_formal_report, initial_parameter_checks, load_source,
                       make_heads, prepare_inputs, source_hash)
    from .evaluate import evaluate_heads

    steps = validate_training_config(cli.steps, cli.lr, cli.smoke)
    output = new_output(cli.output)
    source_before = source_hash()
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   smoke=bool(cli.smoke), steps=steps, updates_per_arm={arm: 0 for arm in TRAIN_ARMS},
                   source_sha256=source_before, test_accessed=False, phase="load_source",
                   arguments=vars(cli), checkpoint_selection="last_equal_budget_only")
    write_json(output / "summary.json", summary)
    capture = None
    try:
        bundle = load_source(cli.protocol, output)
        args, device = bundle["args"], bundle["device"]
        summary["protocol_sha256"] = bundle["protocol_sha256"]
        heads = make_heads(bundle["front"]).to(device)
        optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), lr=cli.lr,
                                            weight_decay=5e-3) for arm in TRAIN_ARMS}
        frozen = [bundle["front"], bundle["metric"], *bundle["decoders"].values()]
        validate_trainable_heads(heads, optimizers, frozen)
        summary["initial_parameter_checks"] = initial_parameter_checks(bundle["front"], heads)
        if summary["initial_parameter_checks"].get("passed") is not True:
            raise ValueError("Initial P/R/T heads differ from their frozen parents")
        summary["trainable_parameters"] = {
            arm: sum(p.numel() for p in heads[arm].parameters()) for arm in TRAIN_ARMS}
        summary["train_protocol"] = dict(
            count=len(bundle["train_names"]), batch=1, accumulation=1,
            full_images=True, augmentation=False, shared_stream=True,
            seed=args.seed, shuffle_seed=args.seed + 17, dtype="float32",
            optimizer="AdamW", learning_rate=cli.lr, weight_decay=5e-3,
            gradient_clip=1.0, schedule="linear warmup then cosine to zero",
            warmup_steps=min(1000, max(1, steps // 10)),
            loss="unweighted NLL + Lovasz-Softmax (present classes; ignore=255)",
            t_role="auxiliary original-token head; not a matched-capacity P/R arm")
        manifest = dict(
            format=FORMAT, smoke=bool(cli.smoke), test_accessed=False,
            protocol_sha256=bundle["protocol_sha256"], source_sha256=source_before,
            source_hashes=bundle.get("hashes", {}),
            source_head_hashes=bundle.get("head_hashes", {}),
            runtime_settings=bundle.get("runtime_settings", {}),
            source_arguments=vars(args), arguments=vars(cli),
            train_list=dict(path=str(Path(args.train_list).resolve()),
                            sha256=digest(args.train_list), count=len(bundle["train_names"])),
            val_list=dict(path=str(Path(args.val_list).resolve()),
                          sha256=digest(args.val_list), count=len(bundle["val_names"])),
            architecture=dict(p="frozen F -> private A1-initialized 256-to-4 Conv2d -> probability resize",
                              r="frozen Q-mean F -> independent same-initialized 256-to-4 affine head -> Q probability unpool -> resize",
                              t="frozen original 128-dimensional tokens -> private original-initialized 128-to-4 linear head -> Q probability unpool -> resize"),
            trainable_parameters=summary["trainable_parameters"],
            initialization=summary["initial_parameter_checks"],
            frozen_components=["A1 feature extractor and original heads", "Q", "region embedding",
                               "Train Memory bank", "retrieval projection", "J1/J2/J3 Reader"],
            steps_per_arm={arm: steps for arm in TRAIN_ARMS},
            train_protocol=summary["train_protocol"],
            validation=dict(initial_images=2 if cli.smoke else 450,
                            final_images=2 if cli.smoke else 450,
                            historical_metric_tolerance=1e-5,
                            checkpoint_selection="last_equal_budget_only"))
        write_json(output / "manifest.json", manifest)
        summary["manifest_file"] = "manifest.json"
        summary["manifest_sha256"] = digest(output / "manifest.json")
        summary["phase"] = "initial_Val"
        write_json(output / "summary.json", summary)
        initial = evaluate_heads(bundle, heads, output=output / "initial_val",
                                 limit=2 if cli.smoke else 0, initial=True)
        summary["initial"] = initial
        assert_formal_report(initial, smoke=cli.smoke)

        from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
            transfer_checked_batch, validate_batch_labels,
        )
        from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
            FrozenFrontCapture, event, full_image_loader,
        )
        # One shared full-image stream; no augmentation or independent per-arm shuffle.
        data = full_image_loader(args.dataset, bundle["train_names"], True,
                                 args.workers, args.seed + 17)
        stream = SharedTrainStream(data, bundle["train_names"])
        capture = FrozenFrontCapture(bundle["front"])
        guard = PoolGuard(output / "train_pool", mode=args.pool_mode,
                          capture="failure", check_reference=False)
        summary["phase"] = "Train_head_only"
        write_json(output / "summary.json", summary)
        stream_hash = hashlib.sha256()
        with guard, (output / "training.jsonl").open("x", encoding="utf-8") as journal:
            for step in range(1, steps + 1):
                batch, image_name, epoch = stream.next()
                summary["last_image"] = image_name
                label_check = validate_batch_labels(batch, step, 0)
                if label_check.get("valid") is not True:
                    write_json(output / "invalid_labels.json", label_check)
                    raise ValueError("Invalid Train labels before any optimizer update")
                pre, post, _loc, target, _names = transfer_checked_batch(batch, device)
                guard.set_image(image_name)
                with torch.no_grad(), torch.autocast(device.type, enabled=False):
                    result, feature = capture(pre, post)
                    inputs = prepare_inputs(feature, result)
                rate = cli.lr * learning_rate_factor(step, steps)
                for optimizer in optimizers.values():
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                logs = matched_update(heads, optimizers, inputs, target)
                summary["updates_per_arm"] = {arm: step for arm in TRAIN_ARMS}
                row = dict(step=step, image=image_name, event=event(image_name),
                           epoch=epoch, lr=rate, heads=logs)
                encoded = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                journal.write(encoded)
                journal.flush()
                stream_hash.update((str(step) + "\t" + image_name + "\n").encode("utf-8"))
                if step == 1 or step % 20 == 0 or step == steps:
                    print("Head fit " + json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
                    write_json(output / "summary.json", summary)
                del inputs, result, feature, pre, post, target
        capture.close()
        capture = None
        summary["train_stream_sha256"] = stream_hash.hexdigest()
        summary["training_journal_sha256"] = digest(output / "training.jsonl")
        summary["checkpoint_file"] = "heads_last.pth"
        # Preserve completed updates even if the following Val reproduction fails.
        save_pt(output / "heads_last.pth", _checkpoint(
            heads, optimizers, bundle, cli, steps, source_before, complete=False))
        immutable = verify_unchanged(bundle)
        immutable["adaptation_code"] = source_before == source_hash()
        immutable["train_pool"] = not guard.failures and guard.restored
        immutable["manifest"] = digest(output / "manifest.json") == summary["manifest_sha256"]
        summary["immutable"] = immutable
        if not all(v is True for v in immutable.values()):
            raise RuntimeError("Source code or frozen training state changed")
        summary["phase"] = "final_Val"
        write_json(output / "summary.json", summary)
        final = evaluate_heads(bundle, heads, output=output / "final_val",
                               limit=2 if cli.smoke else 0, initial=False)
        summary["final"] = final
        assert_formal_report(final, smoke=cli.smoke)
        immutable.update(verify_unchanged(bundle))
        immutable["adaptation_code"] = source_before == source_hash()
        immutable["manifest"] = digest(output / "manifest.json") == summary["manifest_sha256"]
        if not all(v is True for v in immutable.values()):
            raise RuntimeError("Source changed during final audit")
        save_pt(output / "heads_last.pth", _checkpoint(
            heads, optimizers, bundle, cli, steps, source_before, complete=True))
        summary.update(complete=True, full_validation_evidence=not cli.smoke,
                       checkpoint_sha256=digest(output / "heads_last.pth"),
                       phase="complete", immutable=immutable)
        write_json(output / "summary.json", summary)
        print(json.dumps(dict(complete=True, smoke=bool(cli.smoke), steps=steps,
                              output=str(output)), ensure_ascii=False), flush=True)
        return summary
    except Exception as exc:
        summary.update(complete=False, full_validation_evidence=False,
                       error=type(exc).__name__ + ": " + str(exc))
        checkpoint = output / "heads_last.pth"
        if checkpoint.is_file():
            summary["checkpoint_sha256"] = digest(checkpoint)
        write_json(output / "summary.json", summary)
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
