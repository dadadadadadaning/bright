"""Evaluate existing A1/Q/region and frozen J1/J2/J3 on labelled Test878."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
    add_source_arguments,
    digest,
    load_verified_bundle,
    new_output,
    verify_unchanged,
    write_json,
)
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.numeric import runtime_settings
from bda_benchmark.experiments.damageformer_srac_joint123_v1.train import full_image_loader
from bda_benchmark.experiments.damageformer_srac_joint123_v1.joint_runtime import (
    decoder_forward,
    retrieve_pair,
    upsample_probability,
)
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
    probability_prediction,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import event, parse_batch

from .core import ARMS, PAIRS, Accumulator, evaluate_image, read_test_names, sample_paths


FORMAT = "SRAC_EXISTING_REGION_TEST_V1"
TEST_ARMS = (*ARMS, "j1", "j2", "j3")
TEST_PAIRS = (*PAIRS, "a1_pixel__j1", "region__j1", "j1__j2", "j1__j3")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(p)
    p.add_argument("--protocol", required=True, help="Previously passed legacy/native E0 all-path directory")
    p.add_argument("--test_list", required=True, help="Standard labelled Test878 test_set.txt")
    p.add_argument("--max_images", type=int, default=0, help="Diagnostic prefix only; 0 evaluates all 878")
    return p


def source_digest():
    root = Path(__file__).parent
    repository = root.parents[2]
    files = sorted(root.glob("*.py")) + [root / "run.sh"]
    files += [repository / name for name in (
        "bda_benchmark/dataset/make_data_loader.py",
        "bda_benchmark/experiments/damageformer_srac_predsplit_v1/audit.py",
        "bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/feature_core.py",
        "bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/grouped_pool.py",
        "bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/shared.py",
        "bda_benchmark/experiments/damageformer_srac_joint123_v1/joint_runtime.py",
        "bda_benchmark/experiments/damageformer_srac_joint123_v1/train.py",
    )]
    h = hashlib.sha256()
    for path in files:
        h.update(path.relative_to(repository).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def test_file_identity(dataset, names):
    """Check expected files exist and record read-only size/time metadata."""
    result = {}
    for name in names:
        for kind, path in sample_paths(dataset, name).items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing labelled Test input ({name}/{kind}): {path}")
            item = path.stat()
            result[f"{name}/{kind}"] = [item.st_size, item.st_mtime_ns]
    return result


@torch.no_grad()
def existing_joint_probabilities(bundle, result, feature, image_name, output_size):
    """Original frozen J1/J2/J3 inference; GT never enters retrieval."""
    bank = bundle["bank"]
    event_name = event(image_name)
    event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
    _, cosine_support, _, metric_support = retrieve_pair(
        result["tokens"][0], bank, event_id, -1, bundle["topk"], bundle["metric"],
    )
    return {
        arm: upsample_probability(
            decoder_forward(bundle["decoders"][arm], arm, feature, result,
                            cosine_support, metric_support)["logits"], output_size,
        )
        for arm in ("j1", "j2", "j3")
    }


@torch.no_grad()
def run(args):
    if args.smoke:
        raise ValueError("Use --max_images for a labelled Test prefix; source E0 must remain formal")
    if args.max_images < 0 or args.workers < 0:
        raise ValueError("max_images and workers must be nonnegative")
    if args.profile != "legacy" or args.pool_mode != "native":
        raise ValueError("The already passed formal E0 in this run used legacy/native")

    output = new_output(args.output)
    report_path = output / "result.json"
    started = time.perf_counter()
    report = dict(
        format=FORMAT,
        complete=False,
        full_test_evidence=False,
        test_accessed=False,
        memory_artifacts_loaded=False,
        memory_used_in_prediction=False,
        training=False,
        weights_updated=False,
        image_count=0,
        expected_image_count=878,
        arms=list(TEST_ARMS),
        region_probe_arms=list(ARMS),
        existing_joint_arms=["j1", "j2", "j3"],
        comparison_pairs=list(TEST_PAIRS),
        arguments=vars(args),
        source_sha256=source_digest(),
        result_definition="Existing A1, four same-Q/F/A1-head probes, direct region output, and frozen J1/J2/J3 controls",
    )
    write_json(report_path, report)
    capture = None
    guard = None
    bundle = None
    phase = "load_verified_source"
    image_name = None
    try:
        bundle = load_verified_bundle(args)
        report["memory_artifacts_loaded"] = True
        # load_verified_bundle requires the *unchanged*, successful 450-image E0
        # and exact source/runtime matches. Test metrics never feed that Val gate.
        test_names = read_test_names(args.test_list, bundle["train_names"], bundle["val_names"])
        report["test_accessed"] = True
        selected = test_names[:args.max_images] if args.max_images else test_names
        input_identity = test_file_identity(args.dataset, selected)
        test_list_sha256 = digest(args.test_list)
        report.update(
            test_list_sha256=test_list_sha256,
            protocol_sha256=bundle["protocol_sha256"],
            source_hashes=bundle["hashes"],
            source_head_hashes=bundle["head_hashes"],
            runtime_settings=bundle["runtime_settings"],
            front_sha256=bundle["hashes"]["front_sha256"],
            a1_sha256=bundle["hashes"]["a1_sha256"],
            diagnostic_prefix=bool(args.max_images),
            selected_image_count=len(selected),
            input_identity_type="file size and modification time; not full image content hashes",
        )
        write_json(report_path, report)
        source_before = report["source_sha256"]
        data = full_image_loader(args.dataset, selected, False, args.workers, args.seed)
        capture = FrozenFrontCapture(bundle["front"])
        guard = PoolGuard(output / "pool_debug", mode="native", capture="failure", check_reference=False)
        counts = Accumulator(TEST_ARMS, TEST_PAIRS)
        diagnostic_failures = {}
        with guard:
            for batch in data:
                phase = "Test_inference"
                pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                image_name = str(image_names[0])
                if len(pre) != 1 or counts.image_count >= len(selected) or image_name != selected[counts.image_count]:
                    raise ValueError("Unexpected Test image/order/batch")
                guard.set_image(image_name)
                result, feature = capture(pre, post)
                probabilities, checks = evaluate_image(bundle["front"], result, feature, target.shape[-2:])
                probabilities.update(existing_joint_probabilities(
                    bundle, result, feature, image_name, target.shape[-2:],
                ))
                report["memory_used_in_prediction"] = True
                # The native A1 replay is an input-path identity check. Other
                # affine algebra checks can show floating-point differences in
                # legacy mode; keep their exact failure counts as evidence.
                if checks["a1_native_replay"].get("passed") is not True:
                    raise RuntimeError("A1 native replay failed on " + image_name)
                for key, row in checks.items():
                    if isinstance(row, dict) and row.get("passed") is False:
                        item = diagnostic_failures.setdefault(key, dict(count=0, max_abs=0.0))
                        item["count"] += 1
                        if row.get("max_abs") is not None:
                            item["max_abs"] = max(item["max_abs"], float(row["max_abs"]))
                predictions = {
                    arm: probability_prediction(probabilities[arm], target.shape[-2:])
                    for arm in TEST_ARMS
                }
                image_cms = counts.add(image_name, event(image_name), target, predictions)
                with (output / "images.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(
                        index=counts.image_count - 1, image=image_name, event=event(image_name),
                        confusion=image_cms,
                    ), ensure_ascii=False, allow_nan=False) + "\n")
                report["image_count"] = counts.image_count
                if counts.image_count % 10 == 0 or counts.image_count == len(selected):
                    print(f"Test existing region paths: {counts.image_count}/{len(selected)}", flush=True)
        phase = "immutable_source_check"
        immutable = verify_unchanged(bundle)
        immutable.update(
            test_list=digest(args.test_list) == test_list_sha256,
            test_file_metadata=test_file_identity(args.dataset, selected) == input_identity,
            test_code=source_digest() == source_before,
            pool_restored=bool(guard.restored and not guard.failures),
            runtime=runtime_settings() == bundle["runtime_settings"],
        )
        report.update(
            **counts.finish(),
            diagnostics=dict(failed_numerical_checks=diagnostic_failures,
                             pool_guard=guard.report()),
            immutable=immutable,
            wall_seconds=time.perf_counter() - started,
        )
        report["complete"] = bool(counts.image_count == len(selected) and all(immutable.values()))
        report["full_test_evidence"] = bool(report["complete"] and not args.max_images
                                            and counts.image_count == len(test_names) == 878)
        write_json(report_path, report)
        if not report["complete"]:
            raise RuntimeError("Test evaluation or source immutability check failed; partial evidence preserved")
        return report
    except Exception as exc:
        report.update(
            complete=False,
            full_test_evidence=False,
            phase=phase,
            failing_image=image_name,
            error=f"{type(exc).__name__}: {exc}",
            wall_seconds=time.perf_counter() - started,
        )
        if guard is not None:
            report["pool_guard"] = guard.report()
        write_json(report_path, report)
        raise
    finally:
        if capture is not None:
            capture.close()


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
