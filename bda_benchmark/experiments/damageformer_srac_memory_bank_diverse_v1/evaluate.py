"""Frozen same-pass Val450 factorial for Memory bank and Top-8 selection.

The original J3 Reader, front, Q, metric projection and checkpoints never
change. Val truth is touched only after all predictions and support IDs exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .retrieve import (TOPK, add_support_diagnostics, final_support_diagnostics,
                       posthoc_query_label_compatibility, select_diverse,
                       support_diagnostics, support_provenance_changes)


FORMAT = "SRAC_MEMORY_BANK_DIVERSE_V1_VAL_V1"
BASE_ARMS = ("a1_pixel", "direct_region", "j1", "j2", "j3")
FACTORIAL_ARMS = ("j3", "j3_old_diverse", "j3_divbank_top8", "j3_divbank_diverse")
ALL_ARMS = (*BASE_ARMS, "grid_probq", *FACTORIAL_ARMS[1:])
PAIRS = ("j3__j3_old_diverse", "j3__j3_divbank_top8",
         "j3__j3_divbank_diverse", "j3_divbank_top8__j3_divbank_diverse",
         "j1__j3", "j1__j3_old_diverse", "j1__j3_divbank_top8",
         "j1__j3_divbank_diverse", "a1_pixel__grid_probq")


def source_hash() -> str:
    """Hash new evaluation code and the pure Q-probability helper it imports."""
    root = Path(__file__).parent
    experiments = root.parent
    paths = [p for p in root.glob("*.py") if not p.name.startswith("test_")]
    paths += list(root.glob("*.sh"))
    paths += [experiments / "damageformer_srac_region_test_v1" / "core.py",
              experiments / "damageformer_srac_stage_diagnostics_v1" / "feature_core.py",
              experiments / "damageformer_srac_stage_diagnostics_v1" / "grouped_pool.py"]
    checksum = hashlib.sha256()
    for path in sorted(paths):
        checksum.update(path.relative_to(experiments).as_posix().encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def protocol_record(path):
    path = Path(path)
    path = path / "protocol.json" if path.is_dir() else path
    record = json.loads(path.read_text(encoding="utf-8"))
    required = (record.get("format") == "SRAC_HARDQUERY_PROTOCOL_V1",
                record.get("complete") is True,
                record.get("full_validation_evidence") is True,
                record.get("reproduction", {}).get("passed") is True,
                record.get("image_count") == 450,
                record.get("scope") == "all",
                record.get("smoke") is False,
                record.get("profile") == "legacy",
                record.get("pool_mode") == "native")
    if not all(required):
        raise ValueError("Requires passed legacy/native full-Val450 E0 protocol")
    if any(record.get(key) is not True for key in
           ("input_files_unchanged", "code_unchanged", "frozen_evidence_unchanged",
            "numerical_unchanged")):
        raise ValueError("E0 source or runtime immutability gate failed")
    names = record.get("selected_images")
    if not isinstance(names, list) or len(names) != 450 or len(set(names)) != 450:
        raise ValueError("E0 Val list/order is incomplete")
    return path, record


def event_class_counts(bank, event_count):
    import torch
    labels = bank["distribution"].float().argmax(-1).long()
    events = bank["events"].long()
    if len(labels) != len(events) or bool(((events < 0) | (events >= event_count)).any()):
        raise ValueError("Bank events or labels are invalid")
    return torch.bincount(events * 4 + labels, minlength=event_count * 4).reshape(
        event_count, 4).tolist()


def check_train_shard_ledger(records, cache_shards, train_names, event_names, event_fn):
    """Compare each built Train shard with the E0 cache path/hash and identity."""
    if (not isinstance(records, list) or len(records) != 2918
            or len(cache_shards) != len(records) or len(train_names) != len(records)):
        raise ValueError("New bank Train shard ledger must cover all 2918 images")
    for index, (row, original, name) in enumerate(zip(records, cache_shards, train_names)):
        expected_event = event_fn(name)
        if expected_event not in event_names:
            raise ValueError(f"Train shard {index} event absent from source bank")
        if (row.get("path") != original.get("path")
                or row.get("sha256") != original.get("sha256")
                or row.get("name") != name
                or row.get("event") != event_names.index(expected_event)):
            raise ValueError(f"Train shard {index} provenance differs from E0 cache")


def check_diverse_bank(root, bundle, protocol_sha):
    """Verify the diverse Train bank, its quota match, origin and tensor payload."""
    import torch
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import device_bank
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import event

    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "SRAC_MEMORY_BANK_DIVERSE_V1"
            or manifest.get("complete") is not True
            or manifest.get("train_only") is not True
            or manifest.get("test_accessed") is not False
            or manifest.get("training") is not False
            or manifest.get("val_labels_used_for_selection") is not False
            or manifest.get("source_protocol_sha256") != protocol_sha
            or manifest.get("old_bank_sha256") != bundle["hashes"]["bank_sha256"]
            or manifest.get("frozen_metric_checkpoint_sha256") != bundle["hashes"]["metric_sha256"]
            or manifest.get("train_shards") != 2918
            or manifest.get("bank_file") != "bank.pt"):
        raise ValueError("New bank manifest provenance/Train-only gate failed")
    check_train_shard_ledger(manifest.get("train_shard_records"),
                             bundle["cache"].get("shards"),
                             bundle["train_names"], bundle["bank_cpu"]["event_names"], event)
    if (not isinstance(manifest.get("original_immutable"), dict)
            or not manifest["original_immutable"]
            or not all(value is True for value in manifest["original_immutable"].values())):
        raise ValueError("New bank builder's immutable checks did not all pass")
    if manifest.get("source_cache_manifest_sha256") != bundle["hashes"]["cache_manifest_sha256"]:
        raise ValueError("New bank source cache differs from E0")
    if manifest.get("selected_rows_sha256") != digest(root / "selected_rows.json"):
        raise ValueError("New bank selected-row ledger changed")
    bank_file = root / "bank.pt"
    bank_hash = digest(bank_file)
    if manifest.get("new_bank_sha256") != bank_hash:
        raise ValueError("New bank payload checksum differs from manifest")
    # The builder records path+bytes SHA for these two new source files.
    from .bank import source_code_hash
    if manifest.get("source_code_sha256") != source_code_hash():
        raise ValueError("Bank-building source changed after construction")
    bank = torch.load(bank_file, map_location="cpu", weights_only=False)
    old = bundle["bank_cpu"]
    fields = ("tokens", "distribution", "events", "images", "regions",
              "geometry", "disasters")
    if any(field not in bank for field in fields):
        raise ValueError("New bank misses the frozen Reader payload")
    count = len(old["tokens"])
    if (manifest.get("capacity") != count or manifest.get("old_capacity") != count
            or len(bank["tokens"]) != count):
        raise ValueError("Bank capacity changed; factorial requires matched capacity")
    if (bank.get("event_names") != old.get("event_names")
            or bank.get("checkpoint_sha256") != old.get("checkpoint_sha256")):
        raise ValueError("New bank event/front identity differs")
    for field in fields:
        if bank[field].shape != old[field].shape or bank[field].dtype != old[field].dtype:
            raise ValueError("New bank field shape/dtype differs: " + field)
        if not bool(torch.isfinite(bank[field]).all()):
            raise ValueError("New bank contains non-finite field: " + field)
    labels = bank["distribution"].float()
    if bool((labels < 0).any()) or not bool(torch.allclose(
            labels.sum(-1), torch.ones(count), atol=1e-5, rtol=0)):
        raise ValueError("New bank soft labels are invalid")
    images = bank["images"].long()
    regions = bank["regions"].long()
    if bool(((images < 0) | (images >= len(bundle["train_names"]))).any()) or bool((regions < 0).any()):
        raise ValueError("New bank contains an invalid Train image/region ID")
    if len(set(zip(images.tolist(), regions.tolist()))) != count:
        raise ValueError("New bank repeats a Train image/region source row")
    names = bank["event_names"]
    expected_ids = torch.tensor([names.index(event(bundle["train_names"][i]))
                                 for i in images.tolist()], dtype=torch.long)
    if not torch.equal(bank["events"].long(), expected_ids):
        raise ValueError("New bank image/event provenance conflicts")
    old_counts = event_class_counts(old, len(names))
    new_counts = event_class_counts(bank, len(names))
    if (manifest.get("old_event_class_counts") != old_counts
            or manifest.get("event_class_counts") != new_counts
            or old_counts != new_counts):
        raise ValueError("Event x class quota differs from original bank")
    return device_bank(bank, bundle["device"]), manifest, manifest_path, bank_file


def sum_nested_counts(dest, src):
    for key, value in src.items():
        if isinstance(value, dict):
            sum_nested_counts(dest.setdefault(key, {}), value)
        elif isinstance(value, list):
            old = dest.setdefault(key, [0] * len(value))
            for i, item in enumerate(value):
                old[i] += item
        else:
            dest[key] = dest.get(key, 0) + value


class NativeA1LogitsCapture:
    """Capture the low-resolution A1 logits from its original forward pass.

    This frozen SRAC front wraps ``DamageFormer_S8_MICB_DSMIL``, which calls
    ``clf_2`` once per forward.  The feature pre-hook records that call's input;
    this hook records its output before A1 upsamples it to ``old_pixel``.  A
    separate ``clf_2(feature)`` invocation can differ under legacy CUDA settings.
    """

    def __init__(self, front):
        self._calls = 0
        self._last = None
        self._handle = front.base.clf_2.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self._calls += 1
        self._last = output.detach()

    def clear(self):
        self._calls = 0
        self._last = None

    def take(self):
        if self._calls != 1 or self._last is None:
            raise RuntimeError(f"Expected exactly one native A1 classifier call, got {self._calls}")
        logits = self._last
        self.clear()
        return logits

    def close(self):
        self._handle.remove()


def native_grid_probq(result, feature, low_logits, output_size):
    """Pool the native A1 grid probabilities through the unchanged Q routes.

    The same-pass replay of A1's interpolation is an identity gate at the
    existing 2e-5 comparison tolerance. The five historical arms separately
    retain their formal 1e-5 reproduction gate after all 450 images.
    """
    import torch
    from torch.nn import functional as F
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import (
        mean_pool, resize_probability, unpool, validate_routes)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.grouped_pool import GroupedRouting
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import close_report

    required = ("q", "ids", "count", "shape", "old_pixel")
    if any(key not in result for key in required):
        raise ValueError("Incomplete front result for native A1/Q probe")
    shape = tuple(result["shape"])
    if (tuple(feature.shape[-2:]) != shape or low_logits.ndim != 4
            or low_logits.shape[0] != feature.shape[0]
            or low_logits.shape[1] != 4 or tuple(low_logits.shape[-2:]) != shape
            or tuple(result["old_pixel"].shape[-2:]) != tuple(output_size)):
        raise ValueError("Native A1 logits, feature, Q grid or output size disagree")
    if low_logits.device != feature.device or not bool(torch.isfinite(low_logits).all()):
        raise ValueError("Native A1 logits have wrong device or non-finite values")
    q, ids, count = result["q"], result["ids"], result["count"]
    validate_routes(q, ids, count, shape)
    native_probability = F.interpolate(low_logits.float(), size=output_size,
                                       mode="bilinear", align_corners=False).softmax(1)
    old_probability = result["old_pixel"].float().softmax(1)
    replay = close_report(native_probability, old_probability)
    if replay.get("finite") is not True:
        raise ValueError("Native A1 replay contains non-finite probabilities")
    routing = GroupedRouting(ids, count)
    pooled, _ = mean_pool(low_logits.float().softmax(1), q, ids, count,
                          routing=routing)
    grid_probq = resize_probability(unpool(pooled, q, ids, shape), output_size)
    return grid_probq, replay


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, help="Passed E0 directory or protocol.json")
    p.add_argument("--bank", required=True, help="New Train-only bank directory")
    p.add_argument("--output", required=True, help="Fresh Val output directory")
    p.add_argument("--limit", type=int, default=0,
                   help="Debug first N Val images only; 0 is formal Val450")
    return p


def run(cli):
    import torch
    with torch.no_grad():
        return _run(cli)


def _run(cli):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, load_verified_bundle, new_output, safe_child, verify_unchanged, write_json)
    from bda_benchmark.experiments.damageformer_srac_region_test_v1.core import Accumulator
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, decoder_forward, event, full_image_loader,
        gather_support, parse_batch, probability_prediction, select_ids,
        upsample_probability, validate_selected)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import (
        region_retrieval_report, reproduction_report, transition_counts)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import (
        close_report, transition_summary)
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import region_targets

    if cli.limit < 0 or cli.limit > 450:
        raise ValueError("--limit must be 0..450")
    protocol_path, protocol = protocol_record(cli.protocol)
    output = new_output(cli.output)
    summary_path = output / "summary.json"
    started = time.perf_counter()
    own_hash = source_hash()
    summary = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                   image_count=0, phase="load", test_accessed=False, training=False)
    write_json(summary_path, summary)
    capture = native_capture = guard = None
    phase, current_image = "load", None
    try:
        args = argparse.Namespace(**protocol["arguments"])
        args.protocol, args.output = str(protocol_path), str(output)
        bundle = load_verified_bundle(args)
        if bundle["topk"] != TOPK:
            raise ValueError("Factorial requires the frozen Top-8 Reader")
        names = bundle["val_names"]
        if names != protocol["selected_images"] or len(names) != 450:
            raise ValueError("Current Val image list/order differs from E0")
        selected_names = names[:cli.limit] if cli.limit else names
        diverse_bank, bank_manifest, bank_manifest_path, bank_file = check_diverse_bank(
            cli.bank, bundle, digest(protocol_path))
        selected_rows_path = Path(cli.bank) / "selected_rows.json"
        bank_before = (digest(bank_manifest_path), digest(bank_file),
                       digest(selected_rows_path))
        data = full_image_loader(args.dataset, selected_names, False, args.workers, args.seed + 2)
        capture = FrozenFrontCapture(bundle["front"])
        native_capture = NativeA1LogitsCapture(bundle["front"])
        guard = PoolGuard(output / "pool_debug", mode=args.pool_mode,
                          capture="failure", check_reference=False)
        counts = Accumulator(arms=ALL_ARMS, pairs=PAIRS)
        support_totals = {arm: {} for arm in FACTORIAL_ARMS}
        relaxation_totals = {arm: {} for arm in ("j3_old_diverse", "j3_divbank_diverse")}
        label_compatibility = {arm: {} for arm in FACTORIAL_ARMS}
        compatibility_by_event = {}
        source_changes = {arm: {} for arm in FACTORIAL_ARMS[1:]}
        per_event_joint = {}
        native_a1_replay_failures = []
        native_a1_replay_max_abs = 0.0
        recomputed_a1_replay_failures = []
        recomputed_a1_replay_max_abs = 0.0
        artifacts = output / "predictions"
        artifacts.mkdir()
        phase = "Val_same_pass"
        with guard:
            with (output / "images.jsonl").open("x", encoding="utf-8") as journal:
                for index, batch in enumerate(data):
                    pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                    current_image = str(image_names[0])
                    if len(pre) != 1 or index >= len(selected_names) or current_image != selected_names[index]:
                        raise ValueError("Val image/order/batch differs from E0")
                    guard.set_image(current_image)
                    native_capture.clear()
                    result, feature = capture(pre, post)
                    native_low_logits = native_capture.take()
                    old_bank = bundle["bank"]
                    event_name = event(current_image)
                    event_id = (old_bank["event_names"].index(event_name)
                                if event_name in old_bank["event_names"] else -1)
                    query = result["tokens"][0]
                    # Exact E0 call order for five historical paths. No new
                    # retrieval or Q probe runs before these frozen predictions.
                    j2_ids = select_ids(query, old_bank, event_id, -1, TOPK)
                    j3_ids = select_ids(query, old_bank, event_id, -1, TOPK,
                                        projector=bundle["metric"])
                    validate_selected(j2_ids, old_bank, event_id, -1)
                    validate_selected(j3_ids, old_bank, event_id, -1)
                    j2_support = gather_support(j2_ids, old_bank)
                    j3_support = gather_support(j3_ids, old_bank)
                    probability = dict(a1_pixel=result["old_pixel"].float().softmax(1),
                                       direct_region=result["prob"])
                    for arm in ("j1", "j2", "j3"):
                        decoded = decoder_forward(bundle["decoders"][arm], arm,
                                                  feature, result, j2_support, j3_support)
                        probability[arm] = upsample_probability(
                            decoded["logits"], target.shape[-2:])
                    grid_probq, native_replay = native_grid_probq(
                        result, feature, native_low_logits, target.shape[-2:])
                    probability["grid_probq"] = grid_probq
                    native_a1_replay_max_abs = max(
                        native_a1_replay_max_abs, float(native_replay["max_abs"]))
                    if native_replay["passed"] is not True:
                        failure = dict(index=index, image=current_image, **native_replay)
                        native_a1_replay_failures.append(failure)
                        summary.update(native_a1_replay_failed_images=len(native_a1_replay_failures),
                                       native_a1_replay_max_abs=native_a1_replay_max_abs,
                                       native_a1_replay_failure=failure)
                        write_json(summary_path, summary)
                        raise ValueError(
                            f"Same-pass native A1 logits disagree with old_pixel: "
                            f"{current_image}, max_abs={native_replay['max_abs']:.9g}")
                    old_diverse_ids, old_relax = select_diverse(
                        query, old_bank, event_id, -1, TOPK, bundle["metric"])
                    divbank_ids = select_ids(query, diverse_bank, event_id, -1,
                                             TOPK, projector=bundle["metric"])
                    divbank_diverse_ids, divbank_relax = select_diverse(
                        query, diverse_bank, event_id, -1, TOPK, bundle["metric"])
                    support_ids = dict(j3=j3_ids, j3_old_diverse=old_diverse_ids,
                                       j3_divbank_top8=divbank_ids,
                                       j3_divbank_diverse=divbank_diverse_ids)
                    bank_for = dict(j3=old_bank, j3_old_diverse=old_bank,
                                    j3_divbank_top8=diverse_bank,
                                    j3_divbank_diverse=diverse_bank)
                    for arm in FACTORIAL_ARMS:
                        validate_selected(support_ids[arm], bank_for[arm], event_id, -1)
                        add_support_diagnostics(support_totals[arm],
                                                support_diagnostics(support_ids[arm], bank_for[arm]))
                    for arm in FACTORIAL_ARMS[1:]:
                        sum_nested_counts(source_changes[arm], support_provenance_changes(
                            j3_ids, old_bank, support_ids[arm], bank_for[arm]))
                    sum_nested_counts(relaxation_totals["j3_old_diverse"], old_relax)
                    sum_nested_counts(relaxation_totals["j3_divbank_diverse"], divbank_relax)
                    for arm in FACTORIAL_ARMS[1:]:
                        payload = gather_support(support_ids[arm], bank_for[arm])
                        decoded = decoder_forward(bundle["decoders"]["j3"], "j3",
                                                  feature, result, payload, payload)
                        probability[arm] = upsample_probability(
                            decoded["logits"], target.shape[-2:])
                    if set(probability) != set(ALL_ARMS):
                        raise RuntimeError("Unexpected factorial arm set")
                    predictions = {arm: probability_prediction(value, target.shape[-2:])
                                   for arm, value in probability.items()}
                    # Diagnose the old failure path only after every prediction
                    # is fixed, so this extra convolution cannot change the
                    # numerical state used by any of the four Memory arms.
                    recomputed_logits = bundle["front"].base.clf_2(feature.float())
                    recomputed_probability = F.interpolate(
                        recomputed_logits.float(), size=target.shape[-2:],
                        mode="bilinear", align_corners=False).softmax(1)
                    recomputed_replay = close_report(
                        recomputed_probability, result["old_pixel"].float().softmax(1))
                    if recomputed_replay["max_abs"] is not None:
                        recomputed_a1_replay_max_abs = max(
                            recomputed_a1_replay_max_abs, float(recomputed_replay["max_abs"]))
                    if recomputed_replay["passed"] is not True:
                        recomputed_a1_replay_failures.append(dict(
                            index=index, image=current_image, **recomputed_replay))
                    # Query GT first appears below, after all support IDs and
                    # four factorial predictions have been fixed.
                    region_distribution, region_weight = region_targets(target, result)
                    for arm in FACTORIAL_ARMS:
                        row = region_retrieval_report(
                            support_ids[arm], bank_for[arm],
                            region_distribution[0], region_weight[0])
                        sum_nested_counts(label_compatibility[arm], row)
                        stratified = posthoc_query_label_compatibility(
                            support_ids[arm], bank_for[arm],
                            region_distribution[0], region_weight[0])
                        sum_nested_counts(label_compatibility[arm].setdefault("soft_compatibility", {}),
                                          stratified)
                        sum_nested_counts(compatibility_by_event.setdefault(event_name, {}).setdefault(arm, {}),
                                          stratified)
                    image_cms = counts.add(current_image, event_name, target, predictions)
                    for arm in FACTORIAL_ARMS[1:]:
                        joint = transition_counts(target, predictions["j3"], predictions[arm])
                        event_joints = per_event_joint.setdefault(event_name, {})
                        event_joints.setdefault(arm, torch.zeros_like(joint)).add_(joint)
                    arrays = {"target": target.detach().cpu().numpy().astype(np.int16),
                              "support_j2": j2_ids.detach().cpu().numpy().astype(np.int32)}
                    for arm in ALL_ARMS:
                        arrays["prediction_" + arm] = (
                            predictions[arm].detach().cpu().numpy().astype(np.uint8))
                    for arm in FACTORIAL_ARMS:
                        arrays["support_" + arm] = (
                            support_ids[arm].detach().cpu().numpy().astype(np.int32))
                    artifact = artifacts / f"{index:04d}.npz"
                    np.savez_compressed(artifact, **arrays)
                    journal.write(json.dumps(dict(
                        index=index, image=current_image, event=event_name,
                        confusion=image_cms, native_a1_replay=native_replay,
                        recomputed_a1_replay_diagnostic=recomputed_replay,
                        prediction_file=artifact.relative_to(output).as_posix(),
                        prediction_sha256=digest(artifact)), ensure_ascii=False,
                        allow_nan=False) + "\n")
                    journal.flush()
                    summary["image_count"] = counts.image_count
                    summary["native_a1_replay_failed_images"] = len(native_a1_replay_failures)
                    summary["native_a1_replay_max_abs"] = native_a1_replay_max_abs
                    summary["recomputed_a1_replay_failed_images"] = len(recomputed_a1_replay_failures)
                    summary["recomputed_a1_replay_max_abs"] = recomputed_a1_replay_max_abs
                    if counts.image_count % 10 == 0 or counts.image_count == len(selected_names):
                        print(f"Memory diverse-bank Val {counts.image_count}/{len(selected_names)}", flush=True)

        phase = "reproduction_and_integrity"
        rows = [json.loads(line) for line in (output / "images.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if (len(rows) != len(selected_names)
                or [row.get("image") for row in rows] != selected_names
                or [row.get("index") for row in rows] != list(range(len(selected_names)))):
            raise ValueError("Per-image Val evidence journal is incomplete or out of order")
        for row in rows:
            path = safe_child(output, row["prediction_file"])
            if digest(path) != row["prediction_sha256"]:
                raise ValueError("Per-image prediction/support checksum mismatch")
            with np.load(path, allow_pickle=False) as saved:
                if "target" not in saved.files:
                    raise ValueError("Saved prediction artifact lacks target")
                for arm, cm in row["confusion"].items():
                    if ("prediction_" + arm not in saved.files
                            or saved["prediction_" + arm].shape != saved["target"].shape):
                        raise ValueError("Saved prediction artifact shape mismatch")
                if any("support_" + arm not in saved.files for arm in FACTORIAL_ARMS):
                    raise ValueError("Saved support IDs missing")
        results = counts.finish()
        historical = {arm: results["metrics"][arm] for arm in BASE_ARMS}
        smoke = len(selected_names) != 450
        reproduction = reproduction_report(historical, bundle["reference"],
                                           smoke=smoke, image_count=len(selected_names))
        e0_comparison = reproduction_report(historical, protocol["metrics"],
                                            smoke=smoke, image_count=len(selected_names))
        immutable = verify_unchanged(bundle)
        immutable["new_bank"] = bank_before == (
            digest(bank_manifest_path), digest(bank_file), digest(selected_rows_path))
        immutable["new_code"] = source_hash() == own_hash
        pool_ok = bool(guard.restored and not guard.failures)
        formal = bool(len(selected_names) == len(names) == 450
                      and reproduction.get("passed") is True
                      and all(immutable.values()) and pool_ok)
        report = dict(format=FORMAT, complete=formal, full_validation_evidence=formal,
                      **results, original_arms=list(BASE_ARMS),
                      factorial_arms=list(FACTORIAL_ARMS), comparison_pairs=list(PAIRS),
                      support_diagnostics={arm: final_support_diagnostics(row)
                                           for arm, row in support_totals.items()},
                      diversity_relaxation=relaxation_totals,
                      support_source_changes_vs_j3=source_changes,
                      posthoc_region_label_compatibility=label_compatibility,
                      posthoc_region_label_compatibility_by_event=compatibility_by_event,
                      per_event_transitions_vs_j3={
                          event_name: {arm: transition_summary(joint)
                                       for arm, joint in rows.items()}
                          for event_name, rows in per_event_joint.items()},
                      reproduction=reproduction,
                      e0_metric_comparison_informative_only=e0_comparison,
                      native_a1_replay=dict(
                          same_pass_identity_gate=True,
                          source="single native A1 clf_2 output, before Q probability pooling",
                          failed_images=native_a1_replay_failures,
                          failed_count=len(native_a1_replay_failures),
                          max_abs=native_a1_replay_max_abs),
                      recomputed_a1_replay_diagnostic=dict(
                          diagnostic_only=True,
                          source="extra clf_2(feature.float()) invocation after every prediction is fixed",
                          failed_images=recomputed_a1_replay_failures,
                          failed_count=len(recomputed_a1_replay_failures),
                          max_abs=recomputed_a1_replay_max_abs),
                      immutable=immutable, pool_guard=guard.report(),
                      protocol_path=str(protocol_path.resolve()),
                      protocol_sha256=digest(protocol_path),
                      old_bank_sha256=bundle["hashes"]["bank_sha256"],
                      new_bank_sha256=bank_before[1],
                      bank_manifest_sha256=bank_before[0],
                      image_journal_sha256=digest(output / "images.jsonl"),
                      source_sha256=own_hash,
                      profile=protocol["profile"], pool_mode=protocol["pool_mode"],
                      original_top8_rule="Fresh same-pass J3 Top-8; E0 row IDs are not an equality gate",
                      test_accessed=False, training=False,
                      inference_rule="All arms share one front, Q, frozen metric and J3 Reader per image",
                      grid_probq_source="Native A1 low-resolution logits captured during the same front forward",
                      gt_rule="Val truth is post-hoc only, after retrieval and predictions are fixed",
                      causal_limit="Bank/retrieval changes are frozen-Reader input interventions; distribution shift may limit their performance without retraining",
                      elapsed_seconds=time.perf_counter() - started)
        write_json(output / "audit.json", report)
        summary.update(complete=formal, full_validation_evidence=formal,
                       reproduction_passed=reproduction.get("passed"),
                       immutable=immutable, audit_file="audit.json",
                       phase="complete" if formal else "failed_formal_gate")
        write_json(summary_path, summary)
        if not smoke and not formal:
            raise RuntimeError("Memory bank Val reproduction/immutability failed; fixed 1e-5 gate unchanged")
        return report
    except Exception as exc:
        summary.update(complete=False, full_validation_evidence=False,
                       phase=phase, image=current_image,
                       error=f"{type(exc).__name__}: {exc}")
        write_json(summary_path, summary)
        raise
    finally:
        if capture is not None:
            capture.close()
        if native_capture is not None:
            native_capture.close()


if __name__ == "__main__":
    run(parser().parse_args())
