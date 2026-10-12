"""Train three same-capacity, equal-budget corrections of frozen grid_probq.

Only new correction heads are optimized. All arms share one full-image Train
stream and the same frozen front forward per update. Formal runs require
5000 updates per arm plus initial/final passed same-pass Val450 evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .correction import (FIT_FORMAT, FORMAL_STEPS, SMOKE_STEPS, TRAIN_ARMS,
                         assert_evaluation, make_heads)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--keys", required=True, help="Completed paired Train support-key bank")
    p.add_argument("--output", required=True, help="Fresh fit directory")
    p.add_argument("--diagnostics", help="Passed prior diagnose Val450 directory; required for formal fitting")
    p.add_argument("--steps", type=int, default=FORMAL_STEPS)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--bound", type=float, default=2.)
    p.add_argument("--temperature", type=float, default=.1)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--policy", choices=("global", "per_class"), default="global")
    p.add_argument("--smoke", action="store_true", help="20 updates + two Val images; never formal evidence")
    return p


def validate_config(cli):
    if cli.smoke:
        if cli.steps not in (FORMAL_STEPS, SMOKE_STEPS):
            raise ValueError("Smoke requires exactly 20 updates per head")
        steps = SMOKE_STEPS
    else:
        if cli.steps != FORMAL_STEPS:
            raise ValueError("Formal correction fit requires 5000 matched updates per head")
        if not getattr(cli, "diagnostics", None):
            raise ValueError("Formal fitting requires --diagnostics from the completed first-stage Val450 comparison")
        steps = FORMAL_STEPS
    for name in ("lr", "bound", "temperature"):
        value = getattr(cli, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    if type(cli.hidden) is not int or cli.hidden < 1:
        raise ValueError("hidden must be positive")
    if cli.topk != 8:
        raise ValueError("First matched comparison fixes total Top-K to 8")
    return steps


def validate_diagnostics(path, bundle, keys_ledger, *, policy, topk, temperature, own_source):
    """Require completed first-stage evidence before a formal new Reader fit."""
    from .diagnose import FORMAT as DIAGNOSE_FORMAT
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
    root = Path(path).resolve()
    report_path, summary_path = root / "result.json", root / "summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = json.loads(summary_path.read_text(encoding="utf-8"))
    for record in (report, status):
        if (record.get("format") != DIAGNOSE_FORMAT
                or record.get("complete") is not True or record.get("full_validation_evidence") is not True
                or record.get("image_count") != 450 or record.get("smoke") is not False
                or record.get("test_accessed") is not False):
            raise ValueError("First-stage diagnostics must be a completed non-smoke Val450 result")
    bank_sha = next((sha for file, sha in keys_ledger.items() if file.name == "paired_bank.pt"), None)
    manifest_sha = next((sha for file, sha in keys_ledger.items() if file.name == "manifest.json"), None)
    if (report.get("protocol_sha256") != bundle["protocol_sha256"]
            or report.get("source_sha256") != own_source
            or report.get("paired_bank_sha256") != bank_sha
            or report.get("paired_manifest_sha256") != manifest_sha
            or report.get("paired_bank_smoke") is not False
            or policy not in report.get("retrieval_policies", [])
            or report.get("topk") != topk or report.get("temperature") != temperature
            or report.get("reproduction", {}).get("passed") is not True
            or report.get("e0_reproduction", {}).get("passed") is not True):
        raise ValueError("Diagnostics source/protocol/keys/retrieval/gates differ from the proposed correction fit")
    return {report_path: digest(report_path), summary_path: digest(summary_path)}


def validate_trainable_heads(heads, optimizers, frozen_modules=()):
    if set(heads) != set(TRAIN_ARMS) or set(optimizers) != set(TRAIN_ARMS):
        raise ValueError("All three matched heads and optimizers are required")
    seen = set()
    for module in frozen_modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise ValueError("Frozen source has a trainable parameter")
            seen.add(id(parameter))
    counts = {}
    for arm in TRAIN_ARMS:
        parameters = list(heads[arm].parameters())
        own = {id(parameter) for parameter in parameters}
        if own & seen or not own:
            raise ValueError("Private heads share parameters with another arm/source")
        if any(not parameter.requires_grad or parameter.dtype != torch.float32 for parameter in parameters):
            raise ValueError("Correction head parameters must be trainable FP32")
        actual = [p for group in optimizers[arm].param_groups for p in group["params"]]
        if len(actual) != len(parameters) or {id(p) for p in actual} != own:
            raise ValueError("Optimizer does not own exactly its matched head")
        seen.update(own)
        counts[arm] = sum(p.numel() for p in parameters)
    if len(set(counts.values())) != 1:
        raise ValueError("All correction arms must have identical parameter count")
    first = heads[TRAIN_ARMS[0]].state_dict()
    if any(not torch.equal(value, heads[arm].state_dict()[name])
           for arm in TRAIN_ARMS[1:] for name, value in first.items()):
        raise ValueError("Heads must start from exactly identical private copies")
    return counts


def matched_update(heads, optimizers, inputs, target, *, loss_fn=None):
    """Check gradients of every arm before advancing any optimizer."""
    if set(heads) != set(TRAIN_ARMS) or set(inputs) != set(TRAIN_ARMS) or set(optimizers) != set(TRAIN_ARMS):
        raise ValueError("Matched update requires the same three arms")
    if any(value.requires_grad for item in inputs.values() for value in item.values() if torch.is_tensor(value)):
        raise ValueError("Frozen feature/Q/base/evidence inputs must be detached")
    if loss_fn is None:
        from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.train import supervised_loss
        loss_fn = supervised_loss
    for arm in TRAIN_ARMS:
        heads[arm].train()
        optimizers[arm].zero_grad(set_to_none=True)
    logs = {}
    with torch.enable_grad(), torch.autocast(target.device.type, enabled=False):
        for arm in TRAIN_ARMS:
            row = heads[arm](**inputs[arm])
            loss = loss_fn(row["probability"], target)
            # Exact no-support fallback has no model dependency. A connected
            # zero enables a finite zero-gradient matched update, recorded as
            # unsupported; it cannot alter this image's fallback prediction.
            if not loss.requires_grad:
                loss = loss + sum(parameter.sum() * 0 for parameter in heads[arm].parameters())
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite/nonscalar correction loss: " + arm)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(heads[arm].parameters(), 1., error_if_nonfinite=True)
            logs[arm] = dict(loss=float(loss.detach()), gradient_norm=float(norm),
                             mean_gate=float(row["gate"].detach().mean()),
                             supported_coverage=float(inputs[arm]["coverage"].mean()))
    for arm in TRAIN_ARMS:
        optimizers[arm].step()
    return logs


def _checkpoint(heads, optimizers, bundle, summary, complete):
    return dict(format=FIT_FORMAT, complete=complete, smoke=summary["smoke"], test_accessed=False,
                completed_steps=summary["steps"], protocol_sha256=bundle["protocol_sha256"],
                source_sha256=summary["source_sha256"], manifest_sha256=summary["manifest_sha256"],
                key_files_sha256=summary["key_files_sha256"], heads=heads.state_dict(),
                diagnostic_files_sha256=summary.get("diagnostic_files_sha256", {}),
                optimizers={arm: optimizer.state_dict() for arm, optimizer in optimizers.items()},
                architecture=summary["architecture"], retrieval=summary["retrieval"])


def run(cli):
    from .shared import load_bundle, load_paired_bank, source_hash, verify_all
    from .evaluate_correction import bank_ledger, evaluate_correction_heads, retrieve_inputs
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest, new_output, write_json
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import save_pt
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.train import SharedTrainStream, learning_rate_factor
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import FrozenFrontCapture, event, full_image_loader
    from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import transfer_checked_batch, validate_batch_labels
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import NativeA1LogitsCapture, native_grid_probq
    steps = validate_config(cli)
    out = new_output(cli.output)
    own_before = source_hash()
    summary = dict(format=FIT_FORMAT, complete=False, full_validation_evidence=False,
                   test_accessed=False, smoke=bool(cli.smoke), arguments=vars(cli),
                   source_sha256=own_before, steps=steps, updates_per_arm={arm: 0 for arm in TRAIN_ARMS},
                   checkpoint_selection="last_equal_budget_only", phase="load_source",
                   architecture=dict(hidden=cli.hidden, bound=cli.bound, inputs=270,
                                     evidence_channels=10, gate="sigmoid", initial_gate=.5,
                                     initial_delta=0, pixel_gate=True),
                   retrieval=dict(temperature=cli.temperature, topk=cli.topk, policy=cli.policy))
    write_json(out / "summary.json", summary)
    capture = native_capture = None
    try:
        bundle = load_bundle(cli.protocol, out)
        bank, key_manifest = load_paired_bank(cli.keys, bundle, allow_smoke=cli.smoke)
        ledger = bank_ledger(cli.keys)
        summary.update(protocol_sha256=bundle["protocol_sha256"],
                       key_files_sha256={str(path): sha for path, sha in ledger.items()})
        if cli.diagnostics:
            diagnostic_ledger = validate_diagnostics(cli.diagnostics, bundle, ledger, policy=cli.policy,
                                                     topk=cli.topk, temperature=cli.temperature,
                                                     own_source=own_before)
            ledger.update(diagnostic_ledger)
            summary["diagnostic_files_sha256"] = {str(path): sha for path, sha in diagnostic_ledger.items()}
        elif not cli.smoke:
            raise ValueError("Formal fitting requires a completed first-stage diagnostic")
        args, device = bundle["args"], bundle["device"]
        heads = make_heads(cli.hidden, cli.bound).to(device)
        optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), lr=cli.lr, weight_decay=5e-3)
                      for arm in TRAIN_ARMS}
        counts = validate_trainable_heads(heads, optimizers,
                    [bundle["front"], bundle["metric"], *bundle["decoders"].values()])
        summary["trainable_parameters"] = counts
        manifest = dict(format=FIT_FORMAT, source_sha256=own_before,
                        protocol_sha256=bundle["protocol_sha256"],
                        source_hashes=bundle.get("hashes", {}), source_head_hashes=bundle.get("head_hashes", {}),
                        key_files_sha256=summary["key_files_sha256"], arguments=vars(cli),
                        diagnostic_files_sha256=summary.get("diagnostic_files_sha256", {}),
                        source_arguments=vars(args), architecture=summary["architecture"],
                        retrieval=summary["retrieval"], arms=list(TRAIN_ARMS),
                        trainable_parameters=counts, smoke=bool(cli.smoke), test_accessed=False,
                        train_protocol=dict(count=len(bundle["train_names"]), batch=1, augmentation=False,
                                            full_images=True, dtype="float32", shared_stream=True,
                                            shuffle_seed=args.seed + 17, steps_per_arm=steps,
                                            loss="unweighted NLL + Lovasz-Softmax", optimizer="AdamW",
                                            learning_rate=cli.lr, weight_decay=5e-3, gradient_clip=1.,
                                            schedule="linear warmup then cosine to zero"),
                        control="Identical architecture/initialization/F256/base; query-only zeros ten evidence channels and shares F256 coverage",
                        frozen="Entire front/A1/Q/T embedding, original heads/readers/projector and both support keys/labels",
                        gt="Train labels used only for correction loss; Val labels only after prediction scoring",
                        validation=dict(initial_images=2 if cli.smoke else 450,
                                        final_images=2 if cli.smoke else 450, reproduction_tolerance=1e-5),
                        checkpoint_selection="last_equal_budget_only")
        write_json(out / "manifest.json", manifest)
        summary["manifest_sha256"] = digest(out / "manifest.json")
        ledger[out / "manifest.json"] = summary["manifest_sha256"]
        summary["phase"] = "initial_Val"
        write_json(out / "summary.json", summary)
        initial = evaluate_correction_heads(bundle, bank, heads, out / "initial_val", initial=True,
                                             limit=2 if cli.smoke else 0,
                                             extra_ledger=ledger, **summary["retrieval"])
        assert_evaluation(initial, smoke=cli.smoke, initial=True)
        summary["initial"] = initial
        data = full_image_loader(args.dataset, bundle["train_names"], True, args.workers, args.seed + 17)
        stream = SharedTrainStream(data, bundle["train_names"])
        capture = FrozenFrontCapture(bundle["front"])
        native_capture = NativeA1LogitsCapture(bundle["front"])
        guard = PoolGuard(out / "train_pool", mode=args.pool_mode, capture="failure", check_reference=False)
        stream_hash = hashlib.sha256()
        summary["phase"] = "Train_correction_only"
        write_json(out / "summary.json", summary)
        with guard, (out / "training.jsonl").open("x", encoding="utf-8") as journal:
            for step in range(1, steps + 1):
                batch, image_name, epoch = stream.next()
                summary["last_image"] = image_name
                label_check = validate_batch_labels(batch, step, 0)
                if label_check.get("valid") is not True:
                    write_json(out / "invalid_labels.json", label_check)
                    raise ValueError("Invalid Train labels before optimizer updates")
                pre, post, _loc, target, _names = transfer_checked_batch(batch, device)
                guard.set_image(image_name)
                with torch.no_grad(), torch.autocast(device.type, enabled=False):
                    native_capture.clear()
                    result, feature = capture(pre, post)
                    low_logits = native_capture.take()
                    base, replay = native_grid_probq(result, feature, low_logits, target.shape[-2:])
                    if replay.get("passed") is not True:
                        raise RuntimeError("Native A1 same-pass replay failed on Train image " + image_name)
                    inputs, _support = retrieve_inputs(feature, result, base, bank, bundle, image_name,
                                                       **summary["retrieval"])
                rate = cli.lr * learning_rate_factor(step, steps)
                for optimizer in optimizers.values():
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                logs = matched_update(heads, optimizers, inputs, target)
                summary["updates_per_arm"] = {arm: step for arm in TRAIN_ARMS}
                row = dict(step=step, image=image_name, event=event(image_name), epoch=epoch,
                           lr=rate, heads=logs)
                journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                stream_hash.update((str(step) + "\t" + image_name + "\n").encode())
                if step == 1 or step % 20 == 0 or step == steps:
                    print("Memory correction fit " + json.dumps(row, ensure_ascii=False), flush=True)
                    write_json(out / "summary.json", summary)
                del result, feature, base, inputs, pre, post, target
        capture.close()
        capture = None
        native_capture.close()
        native_capture = None
        summary.update(train_stream_sha256=stream_hash.hexdigest(),
                       training_journal_sha256=digest(out / "training.jsonl"),
                       checkpoint_file="heads_last.pth")
        save_pt(out / "heads_last.pth", _checkpoint(heads, optimizers, bundle, summary, False))
        immutable = verify_all(bundle, own_before, ledger)
        immutable["train_pool"] = bool(guard.restored and not guard.failures)
        if not all(immutable.values()):
            raise RuntimeError("Source or training pool changed")
        summary["phase"] = "final_Val"
        write_json(out / "summary.json", summary)
        final = evaluate_correction_heads(bundle, bank, heads, out / "final_val", limit=2 if cli.smoke else 0,
                                           extra_ledger=ledger, **summary["retrieval"])
        assert_evaluation(final, smoke=cli.smoke)
        summary["final"] = final
        immutable.update(verify_all(bundle, own_before, ledger))
        save_pt(out / "heads_last.pth", _checkpoint(heads, optimizers, bundle, summary, True))
        summary.update(complete=True, full_validation_evidence=not cli.smoke, phase="complete",
                       immutable=immutable, checkpoint_sha256=digest(out / "heads_last.pth"))
        write_json(out / "summary.json", summary)
        print(json.dumps(dict(complete=True, smoke=cli.smoke, steps=steps, output=str(out))), flush=True)
        return summary
    except Exception as error:
        summary.update(complete=False, full_validation_evidence=False, error=f"{type(error).__name__}: {error}")
        checkpoint = out / "heads_last.pth"
        if checkpoint.is_file():
            summary["checkpoint_sha256"] = digest(checkpoint)
        write_json(out / "summary.json", summary)
        raise
    finally:
        if capture is not None:
            capture.close()
        if native_capture is not None:
            native_capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
