#!/usr/bin/env python3
"""Train frozen-F/Q pixel corrections; no new Memory, Q or A1 optimization.

local_delta sees F/F; region_delta sees F/Q-mean-unpooled-F. Both have the
same architecture and initialization, but differing input information/rank.
The comparison tests added regional context, not strictly equal effective
capacity, and does not claim to repair the original T128 representation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import torch

from residual_core import ARMS, FORMAT, make_heads, parameter_checks, prepare, probability
from residual_eval import evaluate, source_hash, FORMAT as EVAL_FORMAT
from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.core import (
    assert_formal_report, load_source,
)
from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.train import (
    SharedTrainStream, learning_rate_factor, supervised_loss,
)
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
    digest, new_output, safe_child, verify_unchanged, write_json,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import save_pt

FORMAL_STEPS = 5000
SMOKE_STEPS = 20


def assert_evaluation(result, *, smoke=False, protocol_sha=None, code_sha=None, initial=None):
    assert_formal_report(result, smoke=smoke)
    if (result.get("format") != EVAL_FORMAT or result.get("smoke") is not bool(smoke)
            or result.get("test_accessed") is not False
            or result.get("q_checks_passed") is not True
            or result.get("initial_checks_passed") is not True
            or (not smoke and result.get("e0_metric_comparison", {}).get("passed") is not True)
            or (protocol_sha is not None and result.get("protocol_sha256") != protocol_sha)
            or (code_sha is not None and result.get("source_sha256") != code_sha)
            or (initial is not None and result.get("initial") is not initial)):
        raise ValueError("Residual evaluation provenance/initial/Q/E0 gates failed")
    if initial:
        checks = result.get("initial_probability_checks", {})
        if (result.get("initial_parameters", {}).get("passed") is not True
                or set(checks) != set(ARMS)
                or any(c.get("passed") is not True
                       or c.get("checked_images") != result["image_count"] for c in checks.values())):
            raise ValueError("Missing full initialized grid_probq identity checks")


def training_budget(steps, lr, smoke):
    if type(steps) is not int or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Positive finite LR and integer update budget required")
    expected = SMOKE_STEPS if smoke else FORMAL_STEPS
    if steps != expected:
        raise ValueError(f"This run requires exactly {expected} updates per arm")
    return expected


def validate_optimizers(heads, optimizers, frozen):
    if set(heads) != set(ARMS) or set(optimizers) != set(ARMS):
        raise ValueError("Both matched-stream arms are required")
    seen = set()
    for module in frozen:
        for p in module.parameters():
            if p.requires_grad:
                raise ValueError("Frozen source is unexpectedly trainable")
            seen.add(id(p))
    for arm in ARMS:
        params = list(heads[arm].parameters())
        owned = {id(p) for p in params}
        optimized = [p for group in optimizers[arm].param_groups for p in group["params"]]
        if (not params or owned & seen or len(params) != len(optimized)
                or owned != {id(p) for p in optimized}
                or any(not p.requires_grad or p.dtype != torch.float32 for p in params)):
            raise ValueError("Optimizer must own only an independent private FP32 head")
        seen.update(owned)


def matched_update(heads, optimizers, inputs, target, *, loss_fn=supervised_loss):
    """Validate BOTH gradients before advancing either optimizer."""
    if set(heads) != set(ARMS) or set(optimizers) != set(ARMS):
        raise ValueError("Both arms required")
    if any(v.requires_grad for v in inputs.values() if torch.is_tensor(v)):
        raise ValueError("Frozen inputs must be detached")
    for arm in ARMS:
        heads[arm].train()
        optimizers[arm].zero_grad(set_to_none=True)
    logs = {}
    with torch.enable_grad(), torch.autocast(target.device.type, enabled=False):
        for arm in ARMS:
            p = probability(arm, heads[arm], inputs, target.shape[-2:])
            loss = loss_fn(p, target)
            if loss.ndim or not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite/nonscalar loss: " + arm)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(heads[arm].parameters(), 1.0,
                                                 error_if_nonfinite=True)
            logs[arm] = dict(loss=float(loss.detach()), grad_norm=float(norm))
    for arm in ARMS:
        optimizers[arm].step()
    return logs


def checkpoint(heads, optimizers, bundle, cli, step, code_sha, *, complete):
    return dict(format=FORMAT, complete=bool(complete), completed_steps=step,
                updates_per_arm={arm: step for arm in ARMS}, smoke=bool(cli.smoke),
                test_accessed=False, protocol_sha256=bundle["protocol_sha256"],
                source_sha256=code_sha, config=dict(channels=256, hidden=64, bound=2.0),
                heads=heads.state_dict(), arguments=vars(cli),
                optimizer_states={arm: optimizers[arm].state_dict() for arm in ARMS})


def train(cli):
    from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
        transfer_checked_batch, validate_batch_labels,
    )
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, event, full_image_loader,
    )

    steps = training_budget(cli.steps, cli.lr, cli.smoke)
    output = new_output(cli.output)
    code_sha = source_hash()
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   smoke=bool(cli.smoke), test_accessed=False, phase="load_source",
                   steps=steps, updates_per_arm={arm: 0 for arm in ARMS},
                   source_sha256=code_sha, arguments=vars(cli),
                   checkpoint_selection="last_equal_budget_only")
    write_json(output / "summary.json", summary)
    capture = None
    try:
        bundle = load_source(cli.protocol, output)
        args, device = bundle["args"], bundle["device"]
        summary["protocol_sha256"] = bundle["protocol_sha256"]
        heads = make_heads(bundle["front"]).to(device)
        checks = parameter_checks(bundle["front"], heads)
        if checks.get("passed") is not True:
            raise ValueError("Invalid matched residual initialization")
        optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), lr=cli.lr,
                                            weight_decay=5e-3) for arm in ARMS}
        validate_optimizers(heads, optimizers,
                            [bundle["front"], bundle["metric"], *bundle["decoders"].values()])
        manifest = dict(
            format=FORMAT, smoke=bool(cli.smoke), test_accessed=False,
            protocol_sha256=bundle["protocol_sha256"], source_sha256=code_sha,
            source_hashes=bundle["hashes"], source_head_hashes=bundle["head_hashes"],
            runtime_settings=bundle["runtime_settings"], arguments=vars(cli),
            source_arguments=vars(args), initialization=checks,
            config=dict(channels=256, hidden=64, bound=2.0),
            architecture=dict(local_delta="concat(F,F) -> bounded pixel correction",
                              region_delta="concat(F,Qmean/Qunpool F) -> bounded pixel correction"),
            base="unchanged frozen grid_probq on feature grid, then probability resize",
            interpretation="Added regional mean context versus duplicated local input; equal architecture/parameter count, not equal information rank; not original T128 repair",
            trainable_parameters={arm: sum(p.numel() for p in heads[arm].parameters()) for arm in ARMS},
            train_protocol=dict(count=len(bundle["train_names"]), full_images=True, batch=1,
                                augmentation=False, shared_stream=True, shuffle_seed=args.seed + 17,
                                dtype="float32", optimizer="AdamW", learning_rate=cli.lr,
                                weight_decay=5e-3, gradient_clip=1.0,
                                warmup_steps=min(1000, max(1, steps // 10)),
                                loss="NLL + Lovasz-Softmax; ignore255"),
            frozen_components=["A1", "Q", "original embedding", "original heads",
                               "Train Memory", "retrieval projector", "J1/J2/J3"],
            steps_per_arm={arm: steps for arm in ARMS})
        write_json(output / "manifest.json", manifest)
        summary["manifest_file"] = "manifest.json"
        summary["manifest_sha256"] = digest(output / "manifest.json")
        summary["phase"] = "initial_Val"
        write_json(output / "summary.json", summary)
        initial = evaluate(bundle, heads, output / "initial_val", initial=True,
                           smoke=cli.smoke, source_sha256=code_sha)
        assert_evaluation(initial, smoke=cli.smoke, protocol_sha=bundle["protocol_sha256"],
                          code_sha=code_sha, initial=True)
        summary["initial"] = initial
        data = full_image_loader(args.dataset, bundle["train_names"], True,
                                 args.workers, args.seed + 17)
        stream = SharedTrainStream(data, bundle["train_names"])
        capture = FrozenFrontCapture(bundle["front"])
        guard = PoolGuard(output / "train_pool", mode=args.pool_mode,
                          capture="failure", check_reference=False)
        summary["phase"] = "Train_residual_only"
        write_json(output / "summary.json", summary)
        stream_sha = hashlib.sha256()
        with guard, (output / "training.jsonl").open("x", encoding="utf-8") as journal:
            for step in range(1, steps + 1):
                batch, name, epoch = stream.next()
                summary["last_image"] = name
                labels = validate_batch_labels(batch, step, 0)
                if labels.get("valid") is not True:
                    write_json(output / "invalid_labels.json", labels)
                    raise ValueError("Invalid Train labels before optimizer update")
                pre, post, _loc, target, _names = transfer_checked_batch(batch, device)
                guard.set_image(name)
                with torch.no_grad(), torch.autocast(device.type, enabled=False):
                    result, feature = capture(pre, post)
                    inputs = prepare(feature, result, bundle["front"].base.clf_2)
                rate = cli.lr * learning_rate_factor(step, steps)
                for optimizer in optimizers.values():
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                logs = matched_update(heads, optimizers, inputs, target)
                summary["updates_per_arm"] = {arm: step for arm in ARMS}
                row = dict(step=step, image=name, event=event(name), epoch=epoch,
                           lr=rate, heads=logs)
                journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                stream_sha.update((str(step) + "\t" + name + "\n").encode())
                if step == 1 or step % 20 == 0 or step == steps:
                    print("Residual fit " + json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
                    write_json(output / "summary.json", summary)
                del inputs, result, feature, pre, post, target
        capture.close()
        capture = None
        summary["train_stream_sha256"] = stream_sha.hexdigest()
        summary["training_journal_sha256"] = digest(output / "training.jsonl")
        summary["checkpoint_file"] = "heads_last.pth"
        save_pt(output / "heads_last.pth", checkpoint(heads, optimizers, bundle, cli,
                                                     steps, code_sha, complete=False))
        immutable = verify_unchanged(bundle)
        immutable.update(residual_code=source_hash() == code_sha,
                         manifest=digest(output / "manifest.json") == summary["manifest_sha256"],
                         train_pool=bool(guard.restored and not guard.failures))
        if not all(v is True for v in immutable.values()):
            raise RuntimeError("Frozen source or training provenance changed")
        summary["immutable"] = immutable
        summary["phase"] = "final_Val"
        write_json(output / "summary.json", summary)
        final = evaluate(bundle, heads, output / "final_val", smoke=cli.smoke,
                         source_sha256=code_sha)
        assert_evaluation(final, smoke=cli.smoke, protocol_sha=bundle["protocol_sha256"],
                          code_sha=code_sha, initial=False)
        immutable.update(verify_unchanged(bundle))
        immutable["residual_code"] = source_hash() == code_sha
        immutable["manifest"] = digest(output / "manifest.json") == summary["manifest_sha256"]
        if not all(v is True for v in immutable.values()):
            raise RuntimeError("Provenance changed during final validation")
        summary["final"] = final
        save_pt(output / "heads_last.pth", checkpoint(heads, optimizers, bundle, cli,
                                                     steps, code_sha, complete=True))
        summary.update(complete=True, full_validation_evidence=not cli.smoke,
                       phase="complete", immutable=immutable,
                       checkpoint_sha256=digest(output / "heads_last.pth"))
        write_json(output / "summary.json", summary)
        print(json.dumps(dict(complete=True, smoke=cli.smoke, output=str(output)), ensure_ascii=False), flush=True)
        return summary
    except Exception as exc:
        summary.update(complete=False, full_validation_evidence=False,
                       error=type(exc).__name__ + ": " + str(exc))
        if (output / "heads_last.pth").is_file():
            summary["checkpoint_sha256"] = digest(output / "heads_last.pth")
        write_json(output / "summary.json", summary)
        raise
    finally:
        if capture is not None:
            capture.close()


def load_fit(fit, bundle):
    """Only completed formal, equal-budget, hash-bound last checkpoints."""
    root = Path(fit).resolve()
    summary_path, manifest_path = root / "summary.json", root / "manifest.json"
    s = json.loads(summary_path.read_text())
    if (s.get("format") != FORMAT or s.get("complete") is not True
            or s.get("full_validation_evidence") is not True or s.get("smoke") is not False
            or s.get("test_accessed") is not False or s.get("steps") != FORMAL_STEPS
            or s.get("updates_per_arm") != {a: FORMAL_STEPS for a in ARMS}
            or s.get("protocol_sha256") != bundle["protocol_sha256"]
            or s.get("source_sha256") != source_hash()
            or s.get("manifest_file") != "manifest.json"
            or s.get("manifest_sha256") != digest(manifest_path)
            or not s.get("immutable") or not all(v is True for v in s["immutable"].values())):
        raise ValueError("Incomplete, stale-source, smoke or unmatched fit")
    for key in ("initial", "final"):
        assert_evaluation(s[key], protocol_sha=bundle["protocol_sha256"],
                          code_sha=s["source_sha256"], initial=(key == "initial"))
    m = json.loads(manifest_path.read_text())
    for key, expected in (("format", FORMAT), ("smoke", False), ("test_accessed", False),
                          ("protocol_sha256", bundle["protocol_sha256"]),
                          ("source_sha256", source_hash()), ("source_hashes", bundle["hashes"]),
                          ("source_head_hashes", bundle["head_hashes"]),
                          ("runtime_settings", bundle["runtime_settings"])):
        if m.get(key) != expected:
            raise ValueError("Manifest provenance mismatch: " + key)
    path = safe_child(root, s.get("checkpoint_file"))
    if path.name != "heads_last.pth" or digest(path) != s.get("checkpoint_sha256"):
        raise ValueError("Last checkpoint digest mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (("format", FORMAT), ("complete", True), ("smoke", False),
                          ("test_accessed", False), ("completed_steps", FORMAL_STEPS),
                          ("updates_per_arm", {a: FORMAL_STEPS for a in ARMS}),
                          ("protocol_sha256", bundle["protocol_sha256"]),
                          ("source_sha256", source_hash()),
                          ("config", dict(channels=256, hidden=64, bound=2.0))):
        if payload.get(key) != expected:
            raise ValueError("Checkpoint provenance mismatch: " + key)
    if payload.get("arguments") != s.get("arguments") or m.get("arguments") != s.get("arguments"):
        raise ValueError("Training arguments disagree")
    heads = make_heads(bundle["front"])
    heads.load_state_dict(payload["heads"], strict=True)
    if not all(bool(torch.isfinite(v).all()) for v in heads.state_dict().values()):
        raise ValueError("Nonfinite residual weights")
    for path in (summary_path, manifest_path, root / "heads_last.pth"):
        key = "residual_fit_" + path.name
        bundle["input_paths"][key] = path
        bundle["inputs_before"][key] = digest(path)
    return heads.to(bundle["device"]), dict(
        fit=str(root), completed_steps=FORMAL_STEPS,
        summary_sha256=digest(summary_path), manifest_sha256=digest(manifest_path),
        checkpoint_sha256=s["checkpoint_sha256"], protocol_sha256=s["protocol_sha256"],
        source_sha256=s["source_sha256"], full_validation_evidence=True,
        checkpoint_selection="last_equal_budget_only")


def independent_evaluate(cli):
    bundle = load_source(cli.protocol, cli.output)
    heads, metadata = load_fit(cli.fit, bundle)
    bundle["head_fit_metadata"] = metadata
    result = evaluate(bundle, heads, cli.output)
    # The immutable input ledger above binds the fit; this is a display reference.
    print("Completed independent evaluation of " + metadata["fit"], flush=True)
    return result


def report(output):
    root = Path(output)
    s = json.loads((root / "summary.json").read_text())
    if (s.get("format") != EVAL_FORMAT or s.get("complete") is not True or s.get("full_validation_evidence") is not True
            or s.get("image_count") != 450 or s.get("test_accessed") is not False
            or s.get("source_sha256") != source_hash()
            or s.get("result_sha256") != digest(root / "result.json")):
        raise ValueError("Report requires a complete source-bound formal Val450")
    r = json.loads((root / "result.json").read_text())
    assert_evaluation(r, protocol_sha=s["protocol_sha256"], code_sha=s["source_sha256"])
    print("Formal Val450; foreground mean of Intact/Damaged/Destroyed; percent", flush=True)
    for arm in ("a1_pixel", "grid_probq", "direct_region", *ARMS):
        v = r["metrics"][arm]
        print(arm, "mIoU=" + f'{100*v["valid_miou"]:.4f}',
              "IoU=" + "/".join("null" if x is None else f"{100*x:.4f}" for x in v["iou"]), flush=True)
    for pair, delta in r["miou_delta_percentage_points"].items():
        print(pair, delta, "pp", flush=True)
    print("GT×old×new transitions and event breakdown: result.json", flush=True)
    return r


def preflight():
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    print("CPU contracts passed. No BRIGHT/CUDA training or formal score claimed.", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    t = sub.add_parser("train")
    t.add_argument("--protocol", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--steps", type=int, default=FORMAL_STEPS)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--smoke", action="store_true", help="Require --steps 20; never formal")
    e = sub.add_parser("evaluate")
    e.add_argument("--protocol", required=True)
    e.add_argument("--fit", required=True)
    e.add_argument("--output", required=True)
    r = sub.add_parser("report")
    r.add_argument("--output", required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command == "preflight":
        preflight()
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        independent_evaluate(args)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
