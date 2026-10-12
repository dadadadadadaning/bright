"""Observe an unchanged Memory builder once, then bounded Train-only replays."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid


ERROR = "Original T128 supports differ on replay:"
DEFAULT_PROTOCOL = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json"
DEFAULT_RESULTS = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_KEY_COMPARE_V1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def find_failure(base, explicit=None):
    if explicit:
        root = Path(explicit).resolve()
        record = read(root / "summary.json")
        if ERROR not in record.get("error", "") or record.get("complete") is not False:
            raise ValueError("Selected record is not the incomplete T128 replay failure")
        return root
    found = []
    for p in Path(base).glob("*/keys_smoke/summary.json"):
        try:
            record = read(p)
            if ERROR in record.get("error", "") and record.get("complete") is False:
                found.append(p)
        except (OSError, ValueError):
            continue
    if not found:
        raise ValueError("No failed keys_smoke record; specify --failed-build explicitly")
    return max(found, key=lambda p: (p.stat().st_mtime_ns, str(p))).parent.resolve()


def package_check():
    root = Path(__file__).resolve().parent
    manifest = read(root / "SHA256SUMS.json")
    expected = {"run_debug.py", "numeric.py", "test_numeric.py", "test_runner.py", "README.md"}
    if set(manifest) != expected:
        raise ValueError("Unexpected diagnostic package file list")
    for name, expected_sha in manifest.items():
        if sha(root / name) != expected_sha:
            raise ValueError("Diagnostic package checksum mismatch: " + name)
    return {str(root / name): value for name, value in manifest.items()}


def failed_frame(error, builder):
    tb = error.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is builder.run.__code__:
            return tb.tb_frame.f_locals
        tb = tb.tb_next
    raise RuntimeError("No unchanged builder failure frame available")


def prepare(cli):
    ledger = package_check()
    repo = Path(cli.repo).resolve()
    builder_file = repo / "bda_benchmark/experiments/damageformer_srac_memory_key_compare_v1/build_keys.py"
    if not builder_file.is_file():
        raise ValueError("Install the verified original Memory package first")
    failed = find_failure(cli.results_base, cli.failed_build)
    old_summary, manifest = read(failed / "summary.json"), read(failed / "manifest.json")
    if manifest.get("smoke") is not True or manifest.get("expected_source_images") != 4:
        raise ValueError("This helper is bounded to the original four-image build_smoke")
    protocol = Path(cli.protocol).resolve()
    if sha(protocol) != manifest.get("protocol_sha256"):
        raise ValueError("Selected failure and E0 protocol SHA differ")
    sys.path.insert(0, str(repo))
    from bda_benchmark.experiments.damageformer_srac_memory_key_compare_v1 import build_keys, shared
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import protocol_record
    protocol_record(protocol)
    if shared.source_hash() != manifest.get("source_sha256"):
        raise ValueError("Memory source changed since failure; no gate is relaxed")
    ledger.update({str(failed / name): sha(failed / name) for name in ("summary.json", "manifest.json")})
    journal = failed / "images.jsonl"
    if journal.is_file():
        ledger[str(journal)] = sha(journal)
    ledger[str(protocol)] = sha(protocol)
    return build_keys, shared, failed, old_summary, manifest, ledger


def exact_equal(a, b):
    import torch
    a, b = a.detach().cpu(), b.detach().cpu()
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def annotate_outliers(report, row_ids, regions):
    for item in report.get("outliers", []):
        row = item["coordinate"][0]
        item.update(original_bank_row=int(row_ids[row]), region_id=int(regions[row]))


def run(cli):
    started = time.perf_counter()
    builder, shared, failed, previous, manifest, ledger = prepare(cli)
    plan = dict(failed_build=str(failed), historical_failed_image=previous.get("image"),
                protocol=str(Path(cli.protocol).resolve()), limit_images=4,
                original_builder_attempts=1, maximum_additional_forwards=3,
                repeat_order=["no_grad_1", "no_grad_2", "inference_mode_1"],
                unchanged_tolerance=dict(atol=2e-5, rtol=2e-5),
                training=False, test_accessed=False, full_validation_evidence=False,
                old_files_unchanged=True, automatic_repair=False,
                interpretation="Diagnostic completion cannot authorize building, retrieval, or training")
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if cli.plan_only:
        return plan
    output = Path(cli.output).resolve() if cli.output else Path(
        "/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REPLAY_DEBUG_V1"
    ) / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print("OUTPUT_ROOT=" + str(output), flush=True)
    summary = dict(**plan, diagnostic_complete=False, phase="unchanged_builder",
                   historical_failure_resolved=False, downstream_authorized=False,
                   package_and_failure_ledger=ledger, original_builder_source_sha256=shared.source_hash())
    write(output / "summary.json", summary)
    try:
        # No trace, patched function, new model hook or CUDA allocation is added
        # before the unchanged original builder runs and applies its own gate.
        try:
            builder.run(argparse.Namespace(protocol=cli.protocol, output=str(output / "exact_build"), limit_images=4))
        except ValueError as error:
            if not str(error).startswith(ERROR):
                raise
            local = failed_frame(error, builder)
            if local.get("token_agreement", {}).get("finite") is not True:
                raise
            summary["original_attempt"] = dict(passed=False, error=str(error),
                                               image=local.get("current_image"),
                                               token_agreement=local["token_agreement"])
            observe_failure(local, output, summary, shared, ledger)
        else:
            summary.update(phase="history_not_reproduced", original_attempt=dict(passed=True),
                           diagnostic_complete=True,
                           reason="One new smoke pass does not resolve the recorded historical failure")
        if not all(sha(path) == expected for path, expected in ledger.items()):
            raise RuntimeError("Original failure, E0 or diagnostic package changed")
        summary["package_and_failure_unchanged"] = True
        summary["elapsed_seconds"] = time.perf_counter() - started
        write(output / "summary.json", summary)
        print("READ-ONLY DIAGNOSTIC COMPLETE. No repair, training, Test or formal Val evidence.", flush=True)
        print("SUMMARY=" + str(output / "summary.json"), flush=True)
        return summary
    except BaseException as error:
        summary.update(diagnostic_complete=False, phase="failed", error=type(error).__name__ + ": " + str(error),
                       elapsed_seconds=time.perf_counter() - started)
        write(output / "summary.json", summary)
        raise


def observe_failure(local, output, summary, shared, ledger):
    import torch
    from numeric import compare_tokens
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import close_report, tensor_hash
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, region_targets)
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import (
        NativeA1LogitsCapture, native_grid_probq)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import mean_pool
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import safe_child

    bundle, result, feature = local["bundle"], local["result"], local["feature"]
    original, regions = bundle["bank_cpu"], local["region_ids"].cpu().long()
    image_id = local["selected_images"][local["index"]]
    row_ids = (original["images"] == image_id).nonzero().flatten()
    expected = original["tokens"][row_ids].detach().cpu().clone()
    raw = result["tokens"][0].index_select(0, regions.to(result["tokens"].device)).detach().float().cpu().clone()
    already_cast = local["replay_tokens"].detach().cpu().clone()
    current = local["current_image"]
    if current != bundle["train_names"][image_id] or len(raw) != len(row_ids):
        raise ValueError("Failure member/image identities differ")
    source_record = bundle["cache"]["shards"][image_id]
    path = safe_child(bundle["cache_root"], source_record["path"])
    if sha(path) != source_record["sha256"]:
        raise ValueError("Original Train shard hash failed")
    ledger[str(path)] = sha(path)
    shard = torch.load(path, map_location="cpu", weights_only=False)
    if shard.get("name") != current or shard.get("image") != image_id:
        raise ValueError("Original Train shard image identity differs")
    identity = dict(tokens=exact_equal(shard["tokens"][regions], expected),
                    distribution=exact_equal(shard["distribution"][regions], original["distribution"][row_ids]),
                    geometry=exact_equal(shard["geometry"][regions], original["geometry"][row_ids]),
                    region_ids=exact_equal(shard["region_ids"][regions], regions))
    if not all(identity.values()):
        raise ValueError("Old bank and hashed original shard disagree: " + str(identity))
    baseline = compare_tokens(raw, expected)
    baseline["builder_gpu_cast_to_stored"] = close_report(already_cast, expected.float())
    baseline["cpu_vs_builder_gpu_cast"] = close_report(raw.to(expected.dtype).float(), already_cast)
    baseline["cpu_vs_builder_gpu_cast_exact"] = exact_equal(raw.to(expected.dtype).float(), already_cast)
    annotate_outliers(baseline, row_ids, regions)
    builder_stored_outliers = compare_tokens(already_cast.float(), expected)
    annotate_outliers(builder_stored_outliers, row_ids, regions)
    labels, _weights = region_targets(local["target"], result)
    expected_labels = original["distribution"][row_ids].float()
    label_check = close_report(labels[0].index_select(0, regions.to(labels.device)), expected_labels)
    mass_check = close_report(result["mass"][0].index_select(0, regions.to(result["mass"].device)), shard["mass"][regions])
    geometry = result["geometry"][0].index_select(0, regions.to(result["geometry"].device)).float().cpu()
    geometry_check = close_report(geometry.to(shard["geometry"].dtype).float(), shard["geometry"][regions].float())
    scene = dict(image=current, train_image_id=image_id, original_rows=row_ids, regions=regions,
                 old_stored_tokens=expected, current_raw_tokens=raw, builder_gpu_cast=already_cast,
                 current_labels=labels[0].index_select(0, regions.to(labels.device)).cpu(),
                 old_labels=expected_labels, current_q=result["q"].detach().cpu(),
                 current_ids=result["ids"].detach().cpu(),
                 native_feature_sha256=tensor_hash(feature),
                 normalized_pre_sha256=tensor_hash(local["pre"]), normalized_post_sha256=tensor_hash(local["post"]),
                 source_shard_sha256=ledger[str(path)], original_bank_sha256=bundle["hashes"]["bank_sha256"],
                 protocol_sha256=bundle["protocol_sha256"], training=False, test_accessed=False,
                 historical_raw_fp32_available=False)
    torch.save(scene, output / "failure_scene.pt")
    summary.update(phase="bounded_replays", failed_image=current, original_train_image_id=image_id,
                   support_rows=len(row_ids), old_bank_dtype=str(expected.dtype),
                   original_same_pass_native_a1=local.get("replay"),
                   old_bank_matches_original_shard=identity, original_cache_extraction=bundle["cache"].get("extraction"),
                   historical_runtime_settings_recorded="runtime_settings" in bundle["cache"],
                   runtime_settings=bundle["runtime_settings"], baseline=baseline,
                   builder_stored_outliers=builder_stored_outliers,
                   soft_label_replay=label_check, selected_mass_replay=mass_check,
                   selected_stored_geometry_replay=geometry_check,
                   failure_scene_sha256=sha(output / "failure_scene.pt"), repeats=[])
    write(output / "summary.json", summary)
    # The failing original run has already restored its hooks and PoolGuard.
    if not local["guard"].restored or local["guard"].failures:
        raise RuntimeError("Original pool failed or its hooks were not restored")
    shared.verify_all(bundle, local["own"], local["image_ledger"])
    capture = FrozenFrontCapture(bundle["front"])
    native = NativeA1LogitsCapture(bundle["front"])
    try:
        for name, inference in (("no_grad_1", False), ("no_grad_2", False), ("inference_mode_1", True)):
            # Fixed order and count; no retry-until-pass, runtime override or bank update.
            guard = PoolGuard(output / (name + "_pool"), mode=bundle["args"].pool_mode,
                              capture="failure", check_reference=False)
            context = torch.inference_mode() if inference else nullcontext()
            with torch.no_grad(), context, guard:
                guard.set_image(current)
                native.clear()
                rerun, rerun_feature = capture(local["pre"], local["post"])
                native_logits = native.take()
                _grid, native_check = native_grid_probq(rerun, rerun_feature, native_logits, local["target"].shape[-2:])
                if native_check.get("passed") is not True:
                    raise ValueError("Same-pass native A1 gate failed in " + name)
                # Keep the builder's pre-token-check operations in these probes.
                _means, _mass = mean_pool(rerun_feature.float(), rerun["q"], rerun["ids"], rerun["count"])
                if rerun["count"] != result["count"] or not exact_equal(rerun["ids"], result["ids"]):
                    raise ValueError("Repeated routing count/IDs changed")
                rr = rerun["tokens"][0].index_select(0, regions.to(rerun["tokens"].device)).float().cpu()
                ll, _ww = region_targets(local["target"], rerun)
                record = dict(name=name, inference_mode=inference, native_a1=native_check,
                              observed_grad_enabled=torch.is_grad_enabled(),
                              observed_inference_mode=torch.is_inference_mode_enabled(),
                              current_to_old=compare_tokens(rr, expected),
                              raw_to_initial_failure=close_report(rr, raw),
                              cast_to_initial_failure=close_report(rr.to(expected.dtype).float(), raw.to(expected.dtype).float()),
                              exact_cast_to_initial_failure=exact_equal(rr.to(expected.dtype), raw.to(expected.dtype)),
                              feature_to_initial_failure=close_report(rerun_feature, feature),
                              q_to_initial_failure=close_report(rerun["q"], result["q"]),
                              loc_to_initial_failure=close_report(rerun["loc"], result["loc"]),
                              adapted_content_to_initial_failure=close_report(rerun["adapted_content"], result["adapted_content"]),
                              mass_to_initial_failure=close_report(rerun["mass"], result["mass"]),
                              geometry_to_initial_failure=close_report(rerun["geometry"], result["geometry"]),
                              soft_labels_to_old=close_report(ll[0].index_select(0, regions.to(ll.device)), expected_labels))
                torch.save(dict(name=name, raw_tokens=rr, stored_tokens=rr.to(expected.dtype)), output / (name + ".pt"))
            if not guard.restored or guard.failures:
                raise RuntimeError("Repeated pool failed or was not restored")
            record["pool_guard"] = guard.report()
            record["scene_sha256"] = sha(output / (name + ".pt"))
            summary["repeats"].append(record)
            write(output / "summary.json", summary)
            print("Replay " + name + ": " + json.dumps(record["current_to_old"]["cast_to_stored"]), flush=True)
    finally:
        capture.close()
        native.close()
    summary["immutable"] = shared.verify_all(bundle, local["own"], local["image_ledger"])
    summary.update(diagnostic_complete=True, phase="observation_complete",
                   interpretation="Three fixed probes describe current repeatability; the old raw FP32 and full old runtime were not saved. No historical reproduction or repair claim.")


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--repo", default="/root/shared-nvme/BRIGHT-master")
    p.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    p.add_argument("--results-base", default=DEFAULT_RESULTS)
    p.add_argument("--failed-build")
    p.add_argument("--output")
    p.add_argument("--plan-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
