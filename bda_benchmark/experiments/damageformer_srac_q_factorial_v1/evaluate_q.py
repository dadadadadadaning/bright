"""Same-pass Val evaluation of one trained Q arm against the immutable E0 source.

The historical A1/region/J1/J2/J3 predictions are reproduced unchanged and
retain the fixed 1e-5 metric gate. New Q results are additional, clearly
named arms. No Val label enters either forward, routing, or support selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


FORMAT = "SRAC_Q_FACTORIAL_EVAL_V1"
BASE_ARMS = ("a1_pixel", "direct_region", "j1", "j2", "j3")
Q_PROBES = ("pixel_grid", "feature_mean", "region_mean", "grid_probq")
NEW_ARMS = ("q_a1_pixel", "q_pixel_grid", "q_feature_mean", "q_region_mean",
            "q_grid_probq", "q_region")
ALL_ARMS = (*BASE_ARMS, *Q_PROBES, *NEW_ARMS)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, help="Passed legacy/native all-path E0 Val450")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Completed Train-only Q-arm last.pth")
    source.add_argument("--source-front", action="store_true",
                        help="Audit the original frozen E0 Q/B before any Q fitting")
    p.add_argument("--output", required=True, help="Fresh output directory; never overwrite an audit")
    p.add_argument("--limit", type=int, default=0,
                   help="First N Val images for debugging; 0 evaluates all 450 formally")
    return p


def evaluation_source_hash():
    """Hash this package's executable files; separate from inherited E0 hash."""
    root = Path(__file__).parent
    paths = sorted(p for p in root.iterdir()
                   if p.is_file() and (p.suffix == ".py" or p.name == "run.sh")
                   and not p.name.startswith("test_"))
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


class PartitionBuildingCapture:
    """Capture the precise predicted B passed to Q, then resize as Q does."""

    def __init__(self, front):
        self._last = None
        self._calls = 0
        self._handle = front.partition.register_forward_pre_hook(self._capture)

    def _capture(self, _module, inputs):
        from torch.nn import functional as F

        if len(inputs) not in (2, 3):
            raise RuntimeError("Expected structural feature and predicted B at Q input")
        embedding, building = inputs[:2]
        if embedding.ndim != 4 or building.ndim != 4 or building.shape[1] != 1:
            raise ValueError("Unexpected Q input shapes")
        self._last = F.interpolate(building.detach().float(), embedding.shape[-2:],
                                   mode="bilinear", align_corners=False)
        self._calls += 1

    def clear(self):
        self._last, self._calls = None, 0

    def take(self, expected_shape):
        import torch

        if self._calls != 1 or self._last is None:
            raise RuntimeError(f"Expected one Q call, got {self._calls}")
        value = self._last
        if tuple(value.shape[-2:]) != tuple(expected_shape) or not bool(torch.isfinite(value).all()):
            raise ValueError("Captured predicted B does not match Q grid")
        self.clear()
        return value

    def close(self):
        self._handle.remove()


def _checked_checkpoint(payload, bundle, checkpoint_sha, smoke):
    """Only a completed Train-only arm from this exact A1/E0 is evaluable."""
    if not isinstance(payload, dict):
        raise ValueError("Q checkpoint is not a dictionary")
    if payload.get("format") != "SRAC_Q_FACTORIAL_V1":
        raise ValueError("Unknown Q checkpoint format")
    if payload.get("run_complete") is not True or payload.get("test_accessed") is not False:
        raise ValueError("Q checkpoint is incomplete or touched Test")
    immutable = payload.get("immutable")
    if (payload.get("a1_unchanged") is not True
            or payload.get("downstream_unchanged") is not True
            or not isinstance(immutable, dict) or not immutable
            or not all(value is True for value in immutable.values())):
        raise ValueError("Q training did not certify frozen A1 and immutable inputs")
    if payload.get("protocol_sha256") != bundle["protocol_sha256"]:
        raise ValueError("Q checkpoint belongs to a different E0 protocol")
    if payload.get("a1_sha256") != bundle["hashes"]["a1_sha256"]:
        raise ValueError("Q checkpoint belongs to a different A1")
    if payload.get("front_sha256") != bundle["hashes"]["front_sha256"]:
        raise ValueError("Q checkpoint belongs to a different frozen SRAC front")
    if not smoke and payload.get("smoke") is not False:
        raise ValueError("Smoke Q checkpoint cannot produce formal Val evidence")
    if not smoke and payload.get("completed_steps") != 2000:
        raise ValueError("Formal Q comparison requires the matched 2000-step budget")
    if type(payload.get("completed_steps")) is not int or payload["completed_steps"] < 1:
        raise ValueError("Q checkpoint has no completed training budget")
    if (payload.get("step") != payload["completed_steps"]
            or payload.get("arguments", {}).get("steps") != payload["completed_steps"]):
        raise ValueError("Q checkpoint is not its equal-budget last step")
    if not checkpoint_sha or payload.get("arm") not in (
        "baseline", "building_reconstruction", "semantic_reconstruction",
        "sar_cue", "rgb_capacity", "shifted_sar",
    ):
        raise ValueError("Unknown Q training arm")


def _same_base(a, b):
    import torch

    x, y = a.state_dict(), b.state_dict()
    return (set(x) == set(y)
            and all(torch.equal(x[name], y[name]) for name in x))


def _select_images(names, limit):
    if type(limit) is not int or not 0 <= limit < 450:
        raise ValueError("--limit must be 0 (formal) or 1..449 (debug prefix)")
    if len(names) != 450 or len(set(names)) != 450:
        raise ValueError("Expected exactly 450 unique E0 Val names")
    return (names[:limit] if limit else names), bool(limit)


def run(cli):
    import numpy as np
    import torch

    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, new_output, safe_child, verify_unchanged, write_json)
    from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.core import load_source
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.native import (
        NativeA1LogitsCapture, native_replay)
    from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.core import checked_journal
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import feature_branches
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        decoder_forward, event, full_image_loader, gather_support, metrics,
        parse_batch, probability_prediction, select_ids, upsample_probability,
        validate_selected, confusion_update)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import reproduction_report
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.protocol import reproduction_for_scope
    from .q_variants import load_q_checkpoint

    # Fail on invalid debug parameters before creating an output or loading GPU weights.
    limit = cli.limit
    if type(limit) is not int or not 0 <= limit < 450:
        raise ValueError("--limit must be 0 or 1..449")
    if cli.source_front == bool(cli.checkpoint):
        raise ValueError("Use exactly one of --source-front or --checkpoint")
    checkpoint_path = Path(cli.checkpoint).resolve() if cli.checkpoint else None
    if checkpoint_path is not None and (not checkpoint_path.is_file()
                                        or checkpoint_path.name != "last.pth"):
        raise ValueError("Use a completed Train-only Q arm last.pth")
    output = new_output(cli.output)
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   image_count=0, expected_image_count=450, test_accessed=False,
                   training=False, smoke=bool(limit), phase="initialize")
    write_json(output / "summary.json", summary)
    started = time.perf_counter()
    source_before = evaluation_source_hash()
    capture = new_capture = native_capture = new_native_capture = None
    original_building = new_building = guard = None
    image_name = None
    try:
        bundle = load_source(cli.protocol, output)
        selected, smoke = _select_images(bundle["val_names"], limit)
        if bundle["topk"] != 8 or bundle["val_names"] != bundle["protocol"]["selected_images"]:
            raise ValueError("Historical E0 Top-8 or Val ordering changed")
        if cli.source_front:
            checkpoint_path = Path(bundle["paths"]["checkpoint"]).resolve()
            checkpoint_sha = bundle["hashes"]["front_sha256"]
            model = bundle["front"]
            payload = dict(arm="original_q", completed_steps=0)
            model_state_before = {}
        else:
            checkpoint_sha = digest(checkpoint_path)
            model, payload = load_q_checkpoint(checkpoint_path, bundle["device"],
                                               a1_override=bundle["args"].a1_checkpoint,
                                               allow_smoke=smoke)
            _checked_checkpoint(payload, bundle, checkpoint_sha, smoke)
            model.eval().requires_grad_(False)
            if not _same_base(bundle["front"].base, model.base):
                raise ValueError("Trained-Q A1 parameters/buffers differ from E0")
            for name in ("adapter", "embedding", "region_classifier"):
                if not _same_base(getattr(bundle["front"], name), getattr(model, name)):
                    raise ValueError("Trained-Q downstream region path differs from E0: " + name)
            # Include the new checkpoint in the inherited immutable-file ledger.
            bundle["input_paths"]["q_checkpoint"] = checkpoint_path
            bundle["inputs_before"]["q_checkpoint"] = checkpoint_sha
            model_state_before = {name: value.detach().clone() for name, value in
                                  model.state_dict().items() if not name.startswith("base.")}
        args = bundle["args"]
        data = full_image_loader(args.dataset, selected, False, args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        new_capture = None if cli.source_front else FrozenFrontCapture(model)
        native_capture = NativeA1LogitsCapture(bundle["front"])
        new_native_capture = None if cli.source_front else NativeA1LogitsCapture(model)
        original_building = PartitionBuildingCapture(bundle["front"])
        new_building = None if cli.source_front else PartitionBuildingCapture(model)
        cms = {arm: torch.zeros(4, 4, dtype=torch.long) for arm in ALL_ARMS}
        by_event = {}
        (output / "predictions").mkdir()
        summary.update(arm=payload["arm"], checkpoint_sha256=checkpoint_sha,
                       protocol_sha256=bundle["protocol_sha256"],
                       source_sha256=source_before,
                       numerical_profile=bundle["protocol"]["profile"],
                       pool_mode=bundle["protocol"]["pool_mode"],
                       checkpoint_steps=payload["completed_steps"], phase="Val_same_pass")
        write_json(output / "summary.json", summary)
        native_ok, native_max = True, 0.0
        guard = PoolGuard(output / "pool_debug", mode=args.pool_mode,
                          capture="failure", check_reference=False)
        with torch.no_grad(), guard, (output / "images.jsonl").open("x", encoding="utf-8") as journal:
            for index, batch in enumerate(data):
                pre, post, _loc, target, names = parse_batch(batch, bundle["device"])
                image_name = str(names[0])
                if len(pre) != 1 or image_name != selected[index]:
                    raise ValueError("Val image/order/batch differs from E0")
                guard.set_image(image_name)
                native_capture.clear()
                original_building.clear()
                result, feature = capture(pre, post)
                low_logits = native_capture.take()
                building_grid = original_building.take(result["shape"])
                bank = bundle["bank"]
                event_name = event(image_name)
                event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
                query = result["tokens"][0]
                selected_ids = {
                    "j2": select_ids(query, bank, event_id, -1, 8),
                    "j3": select_ids(query, bank, event_id, -1, 8,
                                     projector=bundle["metric"]),
                }
                for ids in selected_ids.values():
                    validate_selected(ids, bank, event_id, -1)
                support = {arm: gather_support(ids, bank) for arm, ids in selected_ids.items()}
                probability = dict(a1_pixel=result["old_pixel"].float().softmax(1),
                                   direct_region=result["prob"])
                for arm in ("j1", "j2", "j3"):
                    decoded = decoder_forward(bundle["decoders"][arm], arm, feature,
                                              result, support["j2"], support["j3"])
                    probability[arm] = upsample_probability(decoded["logits"], target.shape[-2:])
                original_replay = native_replay(low_logits, result["old_pixel"], target.shape[-2:])
                original_probes, _, _ = feature_branches(
                    feature.float(), result["q"], result["ids"], result["count"],
                    bundle["front"].base.clf_2, target.shape[-2:])
                probability.update(original_probes)

                if cli.source_front:
                    new_result, new_feature = result, feature
                    new_building_grid, new_replay = building_grid, original_replay
                else:
                    new_native_capture.clear()
                    new_building.clear()
                    new_result, new_feature = new_capture(pre, post)
                    new_low_logits = new_native_capture.take()
                    new_building_grid = new_building.take(new_result["shape"])
                    new_replay = native_replay(new_low_logits, new_result["old_pixel"], target.shape[-2:])
                native_ok &= original_replay.get("passed") is True and new_replay.get("passed") is True
                native_max = max(native_max, float(original_replay.get("max_abs") or 0),
                                 float(new_replay.get("max_abs") or 0))
                if not native_ok:
                    raise ValueError("Native A1 replay failed within image: " + image_name)
                new_probes, _, _ = feature_branches(
                    new_feature.float(), new_result["q"], new_result["ids"], new_result["count"],
                    model.base.clf_2, target.shape[-2:])
                probability.update({"q_" + arm: value for arm, value in new_probes.items()})
                probability["q_a1_pixel"] = new_result["old_pixel"].float().softmax(1)
                probability["q_region"] = new_result["prob"]
                predictions = {arm: probability_prediction(probability[arm], target.shape[-2:])
                               for arm in ALL_ARMS}

                arrays = {
                    "target": target.detach().cpu().numpy().astype(np.int16),
                    "q": result["q"].detach().cpu().numpy(),
                    "region_ids": result["ids"].detach().cpu().numpy(),
                    "grid_shape": np.asarray(result["shape"], dtype=np.int32),
                    "building_grid": building_grid.detach().cpu().numpy(),
                    "q_new": new_result["q"].detach().cpu().numpy(),
                    "region_ids_new": new_result["ids"].detach().cpu().numpy(),
                    "grid_shape_new": np.asarray(new_result["shape"], dtype=np.int32),
                    "building_grid_new": new_building_grid.detach().cpu().numpy(),
                }
                image_cms = {}
                for arm, prediction in predictions.items():
                    cm = torch.zeros(4, 4, dtype=torch.long)
                    confusion_update(cm, target, prediction)
                    cms[arm].add_(cm)
                    by_event.setdefault(event_name, {}).setdefault(
                        arm, torch.zeros_like(cm)).add_(cm)
                    image_cms[arm] = cm.tolist()
                    arrays["prediction_" + arm] = prediction.detach().cpu().numpy().astype(np.uint8)
                for arm, ids in selected_ids.items():
                    arrays["support_" + arm] = ids.detach().cpu().numpy().astype(np.int32)
                artifact = output / "predictions" / f"{index:04d}.npz"
                np.savez_compressed(artifact, **arrays)
                journal.write(json.dumps(dict(
                    index=index, image=image_name, event=event_name, confusion=image_cms,
                    native_a1_replay=dict(original=original_replay, new=new_replay),
                    prediction_file=artifact.relative_to(output).as_posix(),
                    prediction_sha256=digest(artifact)), ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                summary["image_count"] = index + 1
                if (index + 1) % 10 == 0 or index + 1 == len(selected):
                    print(f"Q factorial {payload['arm']} {index + 1}/{len(selected)}", flush=True)

        summary["phase"] = "final_reproduction_and_integrity"
        write_json(output / "summary.json", summary)
        checked_journal(output, selected, digest, safe_child)
        actual = {arm: metrics(cm) for arm, cm in cms.items()}
        reproduction = reproduction_report({arm: actual[arm] for arm in BASE_ARMS},
                                           bundle["reference"], smoke=smoke,
                                           image_count=len(selected))
        q_a1_reproduction = reproduction_for_scope(
            {"a1_pixel": actual["q_a1_pixel"]},
            {"a1_pixel": bundle["reference"]["a1_pixel"]}, "a1",
            smoke=smoke, image_count=len(selected))
        e0_comparison = reproduction_report({arm: actual[arm] for arm in BASE_ARMS},
                                            bundle["protocol"]["metrics"], smoke=smoke,
                                            image_count=len(selected))
        immutable = verify_unchanged(bundle)
        immutable["q_code"] = source_before == evaluation_source_hash()
        immutable["q_model"] = all(torch.equal(value, model.state_dict()[name])
                                   for name, value in model_state_before.items())
        immutable["q_a1"] = _same_base(bundle["front"].base, model.base)
        immutable["q_downstream"] = all(
            _same_base(getattr(bundle["front"], name), getattr(model, name))
            for name in ("adapter", "embedding", "region_classifier")
        )
        pool_ok = bool(guard.restored and not guard.failures)
        complete = bool(len(selected) == summary["image_count"] and all(immutable.values())
                        and pool_ok and native_ok
                        and (smoke or (reproduction.get("passed") is True
                                       and q_a1_reproduction.get("passed") is True)))
        formal = bool(not smoke and complete and len(selected) == 450
                      and reproduction.get("passed") is True
                      and q_a1_reproduction.get("passed") is True)
        per_event = {event_key: {arm: metrics(cm) for arm, cm in values.items()}
                     for event_key, values in by_event.items()}
        report = dict(
            format=FORMAT, complete=complete, full_validation_evidence=formal,
            image_count=len(selected), expected_image_count=450, smoke=smoke,
            arm=payload["arm"], checkpoint_steps=payload["completed_steps"],
            training_hyperparameters=({key: payload["arguments"][key] for key in
                                       ("steps", "eval_every", "batch", "accum", "workers",
                                        "lr", "aux_weight", "cue_gain", "amp")}
                                      if not cli.source_front else None),
            checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
            protocol_path=str(bundle["protocol_path"]), protocol_sha256=bundle["protocol_sha256"],
            source_sha256=source_before, numerical_profile=bundle["protocol"]["profile"],
            pool_mode=bundle["protocol"]["pool_mode"], metrics=actual, per_event=per_event,
            reproduction=reproduction, e0_metric_comparison=e0_comparison,
            q_a1_reproduction=q_a1_reproduction,
            immutable=immutable, native_a1_replay_ok=native_ok,
            native_a1_replay_max_abs=native_max, pool_guard=guard.report(),
            prediction_artifacts="predictions/*.npz", prediction_journal="images.jsonl",
            gt_rule="Val GT used only after original/new forwards, Q, and old support IDs exist",
            historical_rule="Only the five original E0 arms are subject to the unchanged 1e-5 gate",
            q_rule="New Q is evaluated with the frozen original A1 and original downstream region head",
            comparison_caveat="Comparisons across independently trained Q arms require equal budgets and matched checkpoints",
            test_accessed=False, training=False, elapsed_seconds=time.perf_counter() - started,
        )
        write_json(output / "result.json", report)
        summary.update(report)
        summary.update(result_file="result.json", result_sha256=digest(output / "result.json"),
                       reproduction_passed=reproduction.get("passed") is True,
                       phase="complete" if complete else "failed_checks")
        write_json(output / "summary.json", summary)
        if not complete or (not smoke and not formal):
            raise RuntimeError("Q evaluation failed reproduction/integrity; 1e-5 gate unchanged")
        return report
    except Exception as error:
        summary.update(complete=False, full_validation_evidence=False,
                       phase=summary.get("phase"), image=image_name,
                       error=f"{type(error).__name__}: {error}")
        write_json(output / "summary.json", summary)
        raise
    finally:
        for value in (new_building, original_building, new_native_capture,
                      native_capture, new_capture, capture):
            if value is not None:
                value.close()


if __name__ == "__main__":
    run(parser().parse_args())
