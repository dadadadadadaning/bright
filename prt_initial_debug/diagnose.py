"""Read-only, selected-image P/R initialization diagnostics; never formal Val.

Run in a fresh process against the unmodified original repository. No training,
E0 profile changes, retries, threshold changes or complete-image tensor dumps.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="/root/shared-nvme/BRIGHT-master")
    p.add_argument("--failed-run", required=True)
    p.add_argument("--output", required=True, help="New directory outside failed run")
    p.add_argument("--max-images", type=int, default=2,
                   help="Default selection: highest failing P/R images, deduplicated")
    p.add_argument("--image", action="append", help="Exact Val name; repeat to override selection")
    p.add_argument("--head-precision-probes", action="store_true",
                   help="Additional head-only backend contrasts on a frozen tensor snapshot")
    return p


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")


def checked_failure(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output == root or root in output.parents:
        raise ValueError("Output must be a fresh directory outside the failed run")
    files = {name: root / relative for name, relative in {
        "protocol": "fresh_e0/protocol.json", "manifest": "prt_fit/manifest.json",
        "fit_summary": "prt_fit/summary.json", "initial_result": "prt_fit/initial_val/result.json",
        "journal": "prt_fit/initial_val/images.jsonl"}.items()}
    report, fit, manifest = [json.loads(files[key].read_text())
                             for key in ("initial_result", "fit_summary", "manifest")]
    if (report.get("format") != "SRAC_REGION_HEAD_ADAPT_EVAL_V1"
            or fit.get("format") != "SRAC_REGION_HEAD_ADAPT_V1"
            or manifest.get("format") != "SRAC_REGION_HEAD_ADAPT_V1"
            or any(record.get("smoke") is not False or record.get("test_accessed") is not False
                   for record in (report, fit, manifest))):
        raise ValueError("Requires original formal Train/Val head-adaptation evidence")
    if (report.get("complete") is not False or report.get("initial") is not True
            or report.get("full_validation_evidence") is not False
            or report.get("image_count") != 450 or fit.get("complete") is not False
            or fit.get("phase") != "initial_Val"
            or fit.get("updates_per_arm") != {"p": 0, "r": 0, "t": 0}):
        raise ValueError("Requires an untrained, failed full initial Val450 run")
    if (report.get("initial_parameters", {}).get("passed") is not True
            or report.get("native_a1_replay_ok") is not True
            or report.get("reproduction", {}).get("passed") is not True
            or report.get("e0_metric_comparison", {}).get("passed") is not True
            or not report.get("immutable") or not all(report["immutable"].values())
            or report.get("initial_checks_passed") is not False):
        raise ValueError("Failure must be isolated to initialized-head probabilities")
    hashes = {name: digest(path) for name, path in files.items()}
    if manifest.get("protocol_sha256") != hashes["protocol"]:
        raise ValueError("Failed fit manifest does not bind the supplied E0 protocol")
    if report.get("protocol_sha256") != hashes["protocol"]:
        raise ValueError("Initial result does not bind the supplied E0 protocol")
    if fit.get("protocol_sha256") != hashes["protocol"]:
        raise ValueError("Fit summary does not bind the supplied E0 protocol")
    if report.get("source_sha256") != manifest.get("source_sha256") or fit.get("source_sha256") != manifest.get("source_sha256"):
        raise ValueError("Failed report/source manifest hashes disagree")
    rows = [json.loads(line) for line in files["journal"].read_text().splitlines()]
    if len(rows) != 450 or len({row["image"] for row in rows}) != 450:
        raise ValueError("Failed journal must contain exactly 450 unique Val images")
    return files, hashes, manifest, rows


def select_images(rows, names, maximum, requested=None):
    if not isinstance(maximum, int) or not 1 <= maximum < 450:
        raise ValueError("--max-images must be 1..449")
    if requested:
        if len(requested) != len(set(requested)) or any(name not in names for name in requested):
            raise ValueError("Explicit images must be unique E0 Val names")
        return list(requested)
    failures = [(float(check.get("max_abs") or 0), row["image"], arm)
                for row in rows for arm, check in row["initial_head_checks"].items()
                if arm in ("p", "r") and check.get("passed") is not True]
    winners = []
    for arm in ("p", "r"):
        pool = [entry for entry in failures if entry[2] == arm]
        if pool:
            winners.append(max(pool))
    candidates = sorted(winners, reverse=True) + sorted(failures, reverse=True)
    selected = []
    for _score, name, _arm in candidates:
        if name not in selected:
            selected.append(name)
        if len(selected) >= maximum:
            break
    if not selected:
        raise ValueError("No P/R initial probability failures found")
    return selected


def call_with_locals(function, arguments, wanted):
    """Observe the real function's return-frame locals, restoring profiling.

    This does not repeat arithmetic or replace project functions. Instrumented
    selected-image execution is explicitly not a formal replay of old history.
    """
    target = inspect.unwrap(function).__code__
    previous, captured = sys.getprofile(), {}

    def observe(frame, event, arg):
        if frame.f_code is target and event == "return":
            for name in wanted:
                if name in frame.f_locals:
                    captured[name] = frame.f_locals[name]
        if previous is not None:
            previous(frame, event, arg)

    try:
        sys.setprofile(observe)
        result = function(*arguments)
    finally:
        sys.setprofile(previous)
    missing = set(wanted) - set(captured)
    if missing:
        raise RuntimeError("Actual readout locals not captured: " + str(sorted(missing)))
    return result, captured


def package_hash():
    checksum = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.startswith("test_"):
            checksum.update(path.name.encode())
            checksum.update(path.read_bytes())
    return checksum.hexdigest()


def run(cli):
    # Do not import torch/project modules until repository resolution is done.
    repo = Path(cli.repo).resolve()
    if not (repo / "bda_benchmark").is_dir():
        raise ValueError("Repository has no bda_benchmark directory")
    if Path(cli.output).resolve() == repo or repo in Path(cli.output).resolve().parents:
        raise ValueError("Keep diagnostic outputs outside the original repository")
    sys.path.insert(0, str(repo))
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1 import core
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.evaluate import initial_closeness
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.native import NativeA1LogitsCapture, native_replay
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import feature_branches, unpool
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        FrozenFrontCapture, decoder_forward, event, full_image_loader, gather_support,
        parse_batch, select_ids, upsample_probability, validate_selected)
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import verify_unchanged
    from diag_core import backend_flags, close_report, compare_fixed_heads, tensor_meta
    import torch

    files, hashes, manifest, old_rows = checked_failure(cli.failed_run, cli.output)
    out = Path(cli.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(format="PRT_INITIAL_DEBUG_V1", complete=False, diagnostic_only=True,
                  full_validation_evidence=False, training=False, test_accessed=False, smoke=True,
                  source_failure=str(Path(cli.failed_run).resolve()), evidence_sha256=hashes,
                  head_precision_probes=bool(cli.head_precision_probes), images=[],
                  scope="Fresh selected-image instrumented execution; not exact old process history",
                  threshold_rule="No formal gate changed; no training or E0 profile override")
    write_json(out / "result.json", report)
    bundle = capture = native = guard = heads = None
    source_before, diagnostic_before = core.source_hash(), package_hash()
    started = time.perf_counter()
    try:
        if manifest.get("source_sha256") != source_before:
            raise ValueError("Original head-adaptation executable changed since the failed run")
        bundle = core.load_source(files["protocol"], out)
        report["backend_before_head_probes"] = backend_flags()
        if bundle["protocol_sha256"] != hashes["protocol"]:
            raise ValueError("E0 protocol changed while loading")
        if bundle["args"].pool_mode != "native":
            raise ValueError("This diagnostic requires the failed native-pool contract")
        if [row["image"] for row in old_rows] != bundle["val_names"]:
            raise ValueError("Failed initial journal order differs from E0 Val450")
        selected = select_images(old_rows, bundle["val_names"], cli.max_images, cli.image)
        report.update(selected_images=selected, runtime_settings=bundle["runtime_settings"],
                      original_source_sha256=source_before, diagnostic_source_sha256=diagnostic_before)
        heads = core.make_heads(bundle["front"]).to(bundle["device"]).eval()
        report["head_execution_state"] = {
            arm: dict(training=head.training, requires_grad=[p.requires_grad for p in head.parameters()])
            for arm, head in heads.items()}
        report["initial_parameters"] = core.initial_parameter_checks(bundle["front"], heads)
        if report["initial_parameters"]["passed"] is not True:
            raise ValueError("Copied initial head parameters differ")
        saved_heads = {key: tensor_meta(value)["sha256"] for key, value in heads.state_dict().items()}
        report["head_state_sha256"] = saved_heads
        capture, native = FrozenFrontCapture(bundle["front"]), NativeA1LogitsCapture(bundle["front"])
        guard = PoolGuard(out / "pool_debug", mode="native", capture="failure", check_reference=False)
        data = full_image_loader(bundle["args"].dataset, selected, False,
                                 bundle["args"].workers, bundle["args"].seed + 2)
        with torch.no_grad(), guard:
            for index, batch in enumerate(data):
                pre, post, _loc, target, image_names = parse_batch(batch, bundle["device"])
                image = str(image_names[0])
                report["current_image"] = image
                if len(pre) != 1 or image != selected[index]:
                    raise ValueError("Diagnostic loader image/order mismatch")
                print("Diagnosing " + image, flush=True)
                guard.set_image(image)
                native.clear()
                result, feature = capture(pre, post)
                native_low_logits = native.take()
                size, bank = target.shape[-2:], bundle["bank"]
                event_name = event(image)
                event_id = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
                query = result["tokens"][0]
                ids = {"j2": select_ids(query, bank, event_id, -1, 8),
                       "j3": select_ids(query, bank, event_id, -1, 8, projector=bundle["metric"])}
                for selected_ids in ids.values():
                    validate_selected(selected_ids, bank, event_id, -1)
                support = {arm: gather_support(selected_ids, bank) for arm, selected_ids in ids.items()}
                probability = dict(a1_pixel=result["old_pixel"].float().softmax(1), direct_region=result["prob"])
                for arm in ("j1", "j2", "j3"):
                    decoded = decoder_forward(bundle["decoders"][arm], arm, feature, result, support["j2"], support["j3"])
                    probability[arm] = upsample_probability(decoded["logits"], size)
                replay = native_replay(native_low_logits, result["old_pixel"], size)
                if replay.get("passed") is not True:
                    raise ValueError("Captured native A1 replay failed")
                probes, captured = call_with_locals(feature_branches,
                    (feature.float(), result["q"], result["ids"], result["count"], bundle["front"].base.clf_2, size),
                    ("token", "mass", "direct_logits", "token_logits"))
                probability.update(probes[0])
                inputs = core.prepare_inputs(feature, result)
                candidate_p_logits = []
                hook = heads["p"].register_forward_hook(
                    lambda _module, _inputs, output: candidate_p_logits.append(output))
                try:
                    probability["p"] = core.head_probability("p", heads["p"], inputs, size)
                finally:
                    hook.remove()
                if len(candidate_p_logits) != 1:
                    raise RuntimeError("Expected exactly one candidate P classifier call")
                probability["r"], candidate_r = call_with_locals(
                    core.head_probability, ("r", heads["r"], inputs, size), ("logits",))
                probability["t"] = core.head_probability("t", heads["t"], inputs, size)
                checks = initial_closeness(probability)
                original_checks = next(row["initial_head_checks"] for row in old_rows if row["image"] == image)
                item = dict(image=image, native_a1_replay=replay, initial_head_checks=checks,
                            original_head_checks=original_checks,
                            current_failure_reproduced={arm: checks[arm].get("passed") is not True
                                for arm in ("p", "r") if original_checks[arm].get("passed") is not True},
                            native_vs_feature_branch_logits=close_report(captured["direct_logits"], native_low_logits),
                            actual_original_vs_candidate_logits={
                                "p": close_report(candidate_p_logits[0], captured["direct_logits"]),
                                "r": close_report(candidate_r["logits"], captured["token_logits"])},
                            feature_probe_checks=probes[1],
                            original_mean_vs_recomputed=close_report(inputs["mean_tokens"], captured["token"]),
                            original_mass_vs_recomputed=close_report(inputs["mass"], captured["mass"]))
                snapshots = {"feature": feature, "q": result["q"], "ids": result["ids"],
                             "original_mean_tokens": captured["token"], "recomputed_mean_tokens": inputs["mean_tokens"]}
                item["snapshot_metadata"] = {key: tensor_meta(value) for key, value in snapshots.items()}
                saved_inputs = {key: value["sha256"] for key, value in item["snapshot_metadata"].items()}
                item["p_fixed_input"] = compare_fixed_heads("p", feature, bundle["front"].base.clf_2,
                    heads["p"], size, run_backend_probes=cli.head_precision_probes,
                    captured_stages={"logits": captured["direct_logits"], "resize": probes[0]["pixel_grid"]})
                project = lambda p: unpool(p, result["q"], result["ids"], result["shape"])
                item["r_fixed_input"] = compare_fixed_heads("r", captured["token"], bundle["front"].base.clf_2,
                    heads["r"], size, unpool=project, run_backend_probes=cli.head_precision_probes,
                    captured_stages={"logits": captured["token_logits"], "resize": probes[0]["region_mean"]})
                item["snapshot_unchanged"] = all(tensor_meta(snapshots[key])["sha256"] == value
                                                  for key, value in saved_inputs.items())
                if not item["snapshot_unchanged"]:
                    raise RuntimeError("Diagnostic snapshot mutated")
                report["images"].append(item)
                write_json(out / "result.json", report)
        report["image_count"] = len(report["images"])
        report["immutable"] = verify_unchanged(bundle)
        report["immutable"].update(original_head_adapt_code=core.source_hash() == source_before,
            diagnostic_code=package_hash() == diagnostic_before,
            failed_evidence=all(digest(files[key]) == value for key, value in hashes.items()),
            private_heads=all(tensor_meta(value)["sha256"] == saved_heads[key]
                              for key, value in heads.state_dict().items()))
        report["pool_guard"] = guard.report()
        if not all(report["immutable"].values()) or not guard.restored or guard.failures:
            raise RuntimeError("Diagnostic immutable/restore checks failed")
        report["complete"] = report["image_count"] == len(selected)
        return report
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        if capture is not None:
            capture.close()
        if native is not None:
            native.close()
        if guard is not None:
            report["pool_guard"] = guard.report()
        if "error" in report and "immutable" not in report:
            report["immutable_not_checked_due_to_error"] = True
        # After a device error do not hash tensors, query CUDA statistics, retry
        # kernels or verify GPU fingerprints. Preserve the primary exception.
        if "backend_before_head_probes" in report:
            report["backend_restored"] = backend_flags() == report["backend_before_head_probes"]
        report["old_evidence_unchanged"] = {}
        for key, expected in hashes.items():
            try:
                report["old_evidence_unchanged"][key] = digest(files[key]) == expected
            except Exception as error:
                report["old_evidence_unchanged"][key] = False
                report.setdefault("evidence_read_errors", {})[key] = type(error).__name__ + ": " + str(error)
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(out / "result.json", report)


if __name__ == "__main__":
    run(parser().parse_args())
