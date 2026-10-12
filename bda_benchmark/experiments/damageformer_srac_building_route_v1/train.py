"""Fit S0, spatial capacity, hard S2, and soft-B readouts on frozen Q.

All four independently initialized readouts see the same Train image at each
update, use the same optimizer schedule and start from the same E0 region
embedding/classifier. A1, original Q, J1/J2/J3, and Memory never train here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .core import (ARMS, FORMAT, FORMAL_STEPS, assert_independent_heads,
                   head_state, load_route_source, make_heads, source_hash,
                   trainable_parameters, validate_budget)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, help="Passed full legacy/native E0 directory")
    p.add_argument("--output", required=True, help="Fresh Train output directory")
    p.add_argument("--q-checkpoint", help="Optional completed, verified building-Q last.pth for stage 4")
    p.add_argument("--steps", type=int, default=FORMAL_STEPS, choices=(2, 2000, 5000))
    p.add_argument("--eval-every", type=int, default=500,
                   help="Every N steps: diagnostic route-only Val curve; formal five-arm Val is initial/final")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--soft-floor", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true", help="Two updates and two Val images; nonformal")
    return p


def validate_arguments(cli):
    validate_budget(cli.steps, cli.smoke)
    if type(cli.eval_every) is not int or cli.eval_every < 1 or cli.steps % cli.eval_every != 0:
        raise ValueError("--eval-every must divide the matched step budget")
    if not math.isfinite(cli.lr) or cli.lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not math.isfinite(cli.threshold) or not 0 < cli.threshold < 1:
        raise ValueError("Building threshold must be in (0,1)")
    if not math.isfinite(cli.soft_floor) or not 0 <= cli.soft_floor < 0.5:
        raise ValueError("Soft floor must be in [0,0.5)")


def _checkpoint(heads, optimizers, bundle, cli, q_meta, source_before, complete):
    return dict(format=FORMAT, complete=complete, completed_steps=cli.steps,
                protocol_sha256=bundle["protocol_sha256"], source_sha256=source_before,
                smoke=cli.smoke, test_accessed=False, q_source=q_meta,
                heads={arm: head_state(heads[arm]) for arm in ARMS},
                optimizer_states={arm: optimizers[arm].state_dict() for arm in ARMS},
                threshold=cli.threshold, soft_floor=cli.soft_floor,
                arguments=vars(cli))


def assert_completed_audit(report, smoke):
    """A trained checkpoint cannot be certified by a failed or short audit."""
    if report.get("complete") is not True or report.get("image_count") != (2 if smoke else 450):
        raise ValueError("Route audit did not complete its declared Val image set")
    if smoke:
        if report.get("full_validation_evidence") is not False:
            raise ValueError("Smoke audit was incorrectly labelled formal")
    elif (report.get("full_validation_evidence") is not True
          or report.get("reproduction", {}).get("passed") is not True
          or report.get("e0_metric_comparison", {}).get("passed") is not True
          or report.get("q_a1_reproduction", {}).get("passed") is not True
          or report.get("native_a1_replay_ok") is not True
          or not report.get("immutable") or not all(report["immutable"].values())):
        raise ValueError("Route audit failed E0/Q-A1/immutability formal gate")


def periodic_validation_with_capture_reset(evaluate, building_capture, *args, **kwargs):
    """Discard Q-hook calls from periodic Val before the next Train image."""
    try:
        return evaluate(*args, **kwargs)
    finally:
        building_capture.clear()


def run(cli):
    import torch
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
        transfer_checked_batch, validate_batch_labels,
    )
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, new_output, verify_unchanged, write_json,
    )
    from bda_benchmark.experiments.damageformer_srac_q_factorial_v1.evaluate_q import PartitionBuildingCapture
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.core import load_source
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.train import (
        SharedTrainStream, learning_rate_factor, supervised_loss,
    )
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, event, full_image_loader,
    )
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import save_pt
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.train_front import module_fingerprint
    from .evaluate import evaluate_bundle, evaluate_routes_light

    validate_arguments(cli)
    output = new_output(cli.output)
    initial_code = source_hash()
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   smoke=cli.smoke, test_accessed=False, training=True,
                   steps=cli.steps, updates_per_arm={arm: 0 for arm in ARMS},
                   threshold=cli.threshold, soft_floor=cli.soft_floor,
                   source_sha256=initial_code, checkpoint=None, phase="load_source")
    write_json(output / "summary.json", summary)
    capture = building_capture = None
    try:
        bundle = load_source(cli.protocol, output)
        source_model, q_meta = load_route_source(bundle, cli.q_checkpoint, smoke=cli.smoke)
        summary.update(protocol_sha256=bundle["protocol_sha256"], q_source=q_meta)
        source_model.eval().requires_grad_(False)
        front_initial = module_fingerprint(source_model)
        heads = make_heads(source_model, bundle["device"],
                           threshold=cli.threshold, soft_floor=cli.soft_floor)
        assert_independent_heads(heads, source_model)
        optimizers = {arm: torch.optim.AdamW(trainable_parameters(heads[arm]),
                       lr=cli.lr, weight_decay=5e-3) for arm in ARMS}
        summary["trainable_parameters"] = {
            arm: sum(value.numel() for value in trainable_parameters(heads[arm]))
            for arm in ARMS}
        if len(set(summary["trainable_parameters"].values())) != 1:
            raise RuntimeError("Route readouts have unequal trainable parameter counts")
        manifest = dict(format=FORMAT, arguments=vars(cli), q_source=q_meta,
                        protocol_sha256=bundle["protocol_sha256"],
                        source_sha256=initial_code, original_front_sha256=bundle["hashes"]["front_sha256"],
                        original_a1_sha256=bundle["hashes"]["a1_sha256"],
                        train_list_sha256=digest(bundle["args"].train_list),
                        val_list_sha256=digest(bundle["args"].val_list),
                        trainable_parameters=summary["trainable_parameters"],
                        source_model_state_sha256=front_initial,
                        train_protocol="One shared full-image Train2918 stream; four independent readouts; "
                                       "AdamW 0.005 weight decay; warmup/cosine; NLL+Lovasz; no Memory",
                        validation="Initial/final historical five-arm Val450 reproduction at 1e-5; "
                                   "periodic route-only Val is diagnostic",
                        checkpoint_selection="equal-budget last step only; best intermediate is descriptive",
                        test_accessed=False, smoke=cli.smoke)
        write_json(output / "manifest.json", manifest)
        manifest_sha = digest(output / "manifest.json")
        summary["phase"] = "initial_Val"
        write_json(output / "summary.json", summary)
        initial = evaluate_bundle(bundle, source_model, heads, output / "initial_val",
                                  limit=2 if cli.smoke else 0, initial=True, q_metadata=q_meta)
        summary["initial"] = dict(complete=initial["complete"],
                                  full_validation_evidence=initial["full_validation_evidence"],
                                  reproduction=initial["reproduction"],
                                  e0_metric_comparison=initial["e0_metric_comparison"],
                                  q_a1_reproduction=initial["q_a1_reproduction"],
                                  metrics=initial["metrics"])
        assert_completed_audit(initial, cli.smoke)
        if module_fingerprint(source_model) != front_initial:
            raise RuntimeError("Frozen Q source changed during initial Val replay")

        data = full_image_loader(bundle["args"].dataset, bundle["train_names"], True,
                                 bundle["args"].workers, bundle["args"].seed + 17)
        stream = SharedTrainStream(data, bundle["train_names"])
        capture = FrozenFrontCapture(source_model)
        building_capture = PartitionBuildingCapture(source_model)
        guard = PoolGuard(output / "train_pool", mode=bundle["args"].pool_mode,
                          capture="failure", check_reference=False)
        stream_hash = hashlib.sha256()
        summary["phase"] = "Train_route_readouts"
        write_json(output / "summary.json", summary)
        with guard, (output / "training.jsonl").open("x", encoding="utf-8") as journal:
            for step in range(1, cli.steps + 1):
                batch, image_name, epoch = stream.next()
                report = validate_batch_labels(batch, step, 0)
                if report.get("valid") is not True:
                    write_json(output / "invalid_labels.json", report)
                    raise ValueError("Invalid Train label; no remapping")
                pre, post, _loc, target, _names = transfer_checked_batch(batch, bundle["device"])
                guard.set_image(image_name)
                source_model.eval()
                with torch.no_grad(), torch.autocast(bundle["device"].type, enabled=False):
                    result, feature = capture(pre, post)
                    building = building_capture.take(result["shape"])
                lr = cli.lr * learning_rate_factor(step, cli.steps)
                for arm in ARMS:
                    optimizer = optimizers[arm]
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    heads[arm].train()
                logs = {}
                with torch.enable_grad(), torch.autocast(bundle["device"].type, enabled=False):
                    for arm in ARMS:
                        probability = heads[arm].from_frozen(
                            result, feature, building, target.shape[-2:])["prob"]
                        loss = supervised_loss(probability, target)
                        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                            raise RuntimeError("Nonfinite route loss: " + arm)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(
                            trainable_parameters(heads[arm]), 1.0, error_if_nonfinite=True)
                        logs[arm] = dict(loss=float(loss.detach()), grad_norm=float(norm))
                        del probability, loss
                for arm in ARMS:
                    optimizers[arm].step()
                if any(parameter.grad is not None for parameter in source_model.parameters()):
                    raise RuntimeError("Frozen Q/A1 source accumulated a gradient")
                summary["updates_per_arm"] = {arm: step for arm in ARMS}
                encoded = json.dumps(dict(step=step, image=image_name, event=event(image_name),
                                          epoch=epoch, lr=lr, heads=logs), ensure_ascii=False,
                                     allow_nan=False) + "\n"
                journal.write(encoded)
                journal.flush()
                stream_hash.update((str(step) + "\t" + image_name + "\n").encode("utf-8"))
                if step == 1 or step % 20 == 0 or step == cli.steps:
                    print("Route fit " + encoded.strip(), flush=True)
                    write_json(output / "summary.json", summary)
                if step % cli.eval_every == 0:
                    periodic = periodic_validation_with_capture_reset(
                        evaluate_routes_light, building_capture, bundle, source_model, heads,
                        limit=2 if cli.smoke else 0)
                    periodic["step"] = step
                    with (output / "validation_curve.jsonl").open("a", encoding="utf-8") as val_file:
                        val_file.write(json.dumps(periodic, ensure_ascii=False,
                                                  allow_nan=False) + "\n")
                    print("Route diagnostic Val " + json.dumps({arm: periodic["metrics"][arm]["valid_miou"]
                          for arm in ARMS}, ensure_ascii=False), flush=True)
                    if module_fingerprint(source_model) != front_initial:
                        raise RuntimeError("Frozen Q/A1 changed after periodic Val")
                del result, feature, building, pre, post, target
        capture.close()
        building_capture.close()
        capture = building_capture = None
        summary["train_stream_sha256"] = stream_hash.hexdigest()
        summary["training_journal_sha256"] = digest(output / "training.jsonl")
        save_pt(output / "heads_last.pth", _checkpoint(
            heads, optimizers, bundle, cli, q_meta, initial_code, complete=False))
        immutable = verify_unchanged(bundle)
        immutable["route_code"] = initial_code == source_hash()
        immutable["route_q_source"] = front_initial == module_fingerprint(source_model)
        immutable["train_pool"] = bool(guard.restored and not guard.failures)
        immutable["manifest"] = digest(output / "manifest.json") == manifest_sha
        summary["immutable"] = immutable
        if not all(immutable.values()):
            raise RuntimeError("Train source or runtime changed: " + str(immutable))
        summary["phase"] = "final_Val"
        write_json(output / "summary.json", summary)
        final = evaluate_bundle(bundle, source_model, heads, output / "final_val",
                                limit=2 if cli.smoke else 0, initial=False, q_metadata=q_meta)
        summary["final"] = dict(complete=final["complete"],
                                full_validation_evidence=final["full_validation_evidence"],
                                reproduction=final["reproduction"],
                                e0_metric_comparison=final["e0_metric_comparison"],
                                q_a1_reproduction=final["q_a1_reproduction"],
                                metrics=final["metrics"])
        assert_completed_audit(final, cli.smoke)
        immutable.update(verify_unchanged(bundle))
        immutable["route_code"] = initial_code == source_hash()
        immutable["route_q_source"] = front_initial == module_fingerprint(source_model)
        immutable["manifest"] = digest(output / "manifest.json") == manifest_sha
        if not all(immutable.values()):
            raise RuntimeError("Final frozen source/code check failed")
        save_pt(output / "heads_last.pth", _checkpoint(
            heads, optimizers, bundle, cli, q_meta, initial_code, complete=True))
        summary.update(complete=True, full_validation_evidence=not cli.smoke,
                       checkpoint="heads_last.pth", checkpoint_sha256=digest(output / "heads_last.pth"),
                       phase="complete", immutable=immutable, training=False)
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        summary.update(complete=False, full_validation_evidence=False,
                       error=f"{type(exc).__name__}: {exc}")
        if (output / "heads_last.pth").is_file():
            summary["checkpoint_sha256"] = digest(output / "heads_last.pth")
        write_json(output / "summary.json", summary)
        raise
    finally:
        if building_capture is not None:
            building_capture.close()
        if capture is not None:
            capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
