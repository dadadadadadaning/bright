"""One uninstrumented initial Val450, or a separate instrumented scene capture.

Neither mode fits parameters. Capture keeps Python references to actual outputs
until the first observed P/R probability mismatch; it never manufactures a
failure by recomputing selected images or consults labels to choose a scene.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

FORMAT = "PRT_SEQUENCE_CHECK_V1"
SCENE_FORMAT = "PRT_SEQUENCE_SCENE_V1"
ARMS = ("p", "r", "t")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--failed-run", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("full", "capture"), required=True)
    return p


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")


def read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def isolated_initial_probability_failure(report):
    """Do not misclassify a metric, native, pool or integrity failure."""
    checks = report.get("initial_probability_checks", {})
    immutable, pool = report.get("immutable"), report.get("pool_guard", {})
    return bool(
        report.get("format") == "SRAC_REGION_HEAD_ADAPT_EVAL_V1"
        and report.get("initial") is True and report.get("smoke") is False
        and report.get("test_accessed") is False
        and report.get("complete") is False and report.get("image_count") == 450
        and report.get("initial_parameters", {}).get("passed") is True
        and report.get("initial_checks_passed") is False
        and set(checks) == set(ARMS)
        and all(row.get("checked_images") == 450 for row in checks.values())
        and any(checks[arm].get("passed") is False for arm in ("p", "r"))
        and checks["t"].get("passed") is True
        and report.get("reproduction", {}).get("passed") is True
        and report.get("e0_metric_comparison", {}).get("passed") is True
        and report.get("native_a1_replay_ok") is True
        and isinstance(immutable, dict) and immutable
        and all(value is True for value in immutable.values())
        and pool.get("restored") is True and pool.get("failures") == []
    )


def full_initial_pass(report):
    checks, immutable = report.get("initial_probability_checks", {}), report.get("immutable")
    pool = report.get("pool_guard", {})
    return bool(
        report.get("format") == "SRAC_REGION_HEAD_ADAPT_EVAL_V1"
        and report.get("complete") is True
        and report.get("full_validation_evidence") is True
        and report.get("image_count") == report.get("expected_image_count") == 450
        and report.get("initial") is True and report.get("smoke") is False
        and report.get("test_accessed") is False and report.get("training") is False
        and report.get("initial_parameters", {}).get("passed") is True
        and report.get("initial_checks_passed") is True and set(checks) == set(ARMS)
        and all(row.get("passed") is True and row.get("checked_images") == 450
                and row.get("failed_images") == 0 for row in checks.values())
        and report.get("reproduction", {}).get("passed") is True
        and report.get("e0_metric_comparison", {}).get("passed") is True
        and report.get("native_a1_replay_ok") is True
        and isinstance(immutable, dict) and immutable
        and all(value is True for value in immutable.values())
        and pool.get("restored") is True and pool.get("failures") == []
    )


def bind_failed_evidence(failed_run, protocol, output, repo):
    root, out, repo = (Path(value).resolve() for value in (failed_run, output, repo))
    supplied = Path(protocol).resolve()
    supplied = supplied / "protocol.json" if supplied.is_dir() else supplied
    expected = root / "fresh_e0/protocol.json"
    if supplied != expected:
        raise ValueError("Protocol must be the unchanged fresh_e0/protocol.json of the failed run")
    if out.exists() or out == root or root in out.parents or out == repo or repo in out.parents:
        raise ValueError("Output must be new and outside source repository and failed run")
    paths = {"protocol": expected, "manifest": root / "prt_fit/manifest.json",
             "fit_summary": root / "prt_fit/summary.json",
             "initial_result": root / "prt_fit/initial_val/result.json",
             "journal": root / "prt_fit/initial_val/images.jsonl"}
    values = {name: read_object(path) for name, path in paths.items() if name != "journal"}
    hashes = {name: digest(path) for name, path in paths.items()}
    if not isolated_initial_probability_failure(values["initial_result"]):
        raise ValueError("Old failure is not isolated to untrained P/R initialization probabilities")
    for name in ("manifest", "fit_summary"):
        row = values[name]
        if (row.get("format") != "SRAC_REGION_HEAD_ADAPT_V1"
                or row.get("smoke") is not False or row.get("test_accessed") is not False):
            raise ValueError("Expected original formal head-adaptation artifact: " + name)
    fit = values["fit_summary"]
    if (fit.get("complete") is not False or fit.get("phase") != "initial_Val"
            or fit.get("updates_per_arm") != {arm: 0 for arm in ARMS}):
        raise ValueError("Source run must have zero optimizer updates")
    source = values["manifest"].get("source_sha256")
    if not isinstance(source, str) or len(source) != 64:
        raise ValueError("Missing original source hash")
    for name in ("manifest", "fit_summary", "initial_result"):
        if (values[name].get("protocol_sha256") != hashes["protocol"]
                or values[name].get("source_sha256") != source):
            raise ValueError("Failed artifacts disagree on source/protocol identity: " + name)
    names = values["protocol"].get("selected_images")
    rows = [json.loads(line) for line in paths["journal"].read_text().splitlines() if line.strip()]
    if (not isinstance(names, list) or len(names) != 450 or len(set(names)) != 450
            or [row.get("image") for row in rows] != names
            or [row.get("index") for row in rows] != list(range(450))):
        raise ValueError("Old initial journal does not preserve complete E0 Val450 order")
    return paths, hashes, source, names


class CapturedSceneStop(RuntimeError):
    """Controlled diagnostic stop after preserving actual failed outputs."""


def tensor_metadata(value):
    """Metadata only; called after an observed failure, never in the prefix."""
    import torch
    return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                stride=list(value.stride()), storage_offset=value.storage_offset(),
                contiguous=value.is_contiguous(),
                channels_last=value.ndim == 4 and value.is_contiguous(memory_format=torch.channels_last),
                pointer_mod_64=value.data_ptr() % 64, pointer_mod_256=value.data_ptr() % 256,
                version=value._version, requires_grad=bool(value.requires_grad))


def cpu_tensor_digest(value):
    """Pure CPU content checksum after scene transfer."""
    if value.device.type != "cpu":
        raise ValueError("Value checksum must never transfer or hash a CUDA tensor")
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    h.update(value.detach().contiguous().view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


class SceneObserver:
    """Python-only references before mismatch; CPU snapshot only after it."""

    def __init__(self, feature_branches, head_probability, initial_closeness,
                 heads, front, output, record, val_names):
        self.branch_code = inspect.unwrap(feature_branches).__code__
        self.head_code = inspect.unwrap(head_probability).__code__
        self.close_code = inspect.unwrap(initial_closeness).__code__
        self.heads, self.front = heads, front
        self.output, self.record = Path(output), record
        self.val_names = list(val_names)
        self.refs = {}
        self.previous = None
        self.hook = None

    def p_output(self, _module, _inputs, output):
        self.refs["p_private_logits"] = output

    def profile(self, frame, event, result):
        if frame.f_code is self.branch_code:
            if event == "call":
                self.refs = {}
            elif event == "return" and result is not None:
                for actual, stored in (("token", "original_mean_tokens"),
                                       ("mass", "original_mass"),
                                       ("direct_logits", "p_original_logits"),
                                       ("token_logits", "r_original_logits")):
                    if actual in frame.f_locals:
                        self.refs[stored] = frame.f_locals[actual]
        elif frame.f_code is self.head_code and event == "return" and result is not None:
            if frame.f_locals.get("arm") == "r" and "logits" in frame.f_locals:
                self.refs["r_private_logits"] = frame.f_locals["logits"]
        elif frame.f_code is self.close_code and event == "return" and isinstance(result, dict):
            failed = [arm for arm in ("p", "r") if result.get(arm, {}).get("passed") is False]
            if failed:
                caller = frame.f_back
                if caller is None:
                    raise RuntimeError("Missing original evaluator frame at probability mismatch")
                self.save(caller.f_locals, result, failed)
                raise CapturedSceneStop("First observed P/R initial probability mismatch captured")
        if self.previous is not None:
            self.previous(frame, event, result)

    def save(self, live, checks, failed):
        import torch
        expected = ("current_image", "index", "feature", "result", "inputs", "probability",
                    "low_logits", "target")
        missing = set(expected) - set(live)
        missing |= {key for key in ("p_original_logits", "p_private_logits", "r_original_logits",
                                    "r_private_logits", "original_mean_tokens", "original_mass")
                    if key not in self.refs}
        if missing:
            raise RuntimeError("Cannot capture actual failure tensors: " + str(sorted(missing)))
        result, inputs, probability = live["result"], live["inputs"], live["probability"]
        if (type(live["index"]) is not int or not 0 <= live["index"] < len(self.val_names)
                or live["current_image"] != self.val_names[live["index"]]):
            raise RuntimeError("Observed mismatch is outside the original Val order")
        tensors = {**self.refs, "feature": live["feature"], "q": result["q"], "ids": result["ids"],
                   "recomputed_mean_tokens": inputs["mean_tokens"], "recomputed_mass": inputs["mass"],
                   "native_low_logits": live["low_logits"]}
        tensors.update({name: probability[name] for name in ("pixel_grid", "p", "region_mean", "r")})
        metadata = {name: tensor_metadata(value) for name, value in tensors.items()}
        # No tensor hashing, extra classifiers, pools, statistics or precision changes
        # precede the actual failure. CPU conversion begins only now.
        cpu = {name: value.detach().cpu().clone() for name, value in tensors.items()}
        parent_state, private_state = self.front.base.clf_2.state_dict(), self.heads.state_dict()
        state_meta = {group: {name: tensor_metadata(value)
                              for name, value in state.items()}
                      for group, state in (("parent_classifier_state", parent_state),
                                           ("private_heads_state", private_state))}
        cpu_parent = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
        cpu_private = {name: value.detach().cpu().clone() for name, value in private_state.items()}
        scene = dict(format=SCENE_FORMAT, diagnostic_only=True, full_validation_evidence=False,
                     training=False, test_accessed=False, image=live["current_image"],
                     index=live["index"], index_one_based=live["index"] + 1,
                     checks=checks, failed_arms=failed, labels_saved=False,
                     ordered_prefix_images=self.val_names[:live["index"] + 1],
                     original_val_order_sha256=self.record["val_order_sha256"],
                     tensors=cpu, tensor_meta=metadata,
                     tensor_sha256={name: cpu_tensor_digest(value) for name, value in cpu.items()},
                     count=result["count"],
                     shape=tuple(result["shape"]), output_size=tuple(live["target"].shape[-2:]),
                     parent_classifier_state=cpu_parent, private_heads_state=cpu_private,
                     state_meta=state_meta,
                     state_sha256={group: {name: cpu_tensor_digest(value) for name, value in state.items()}
                                   for group, state in (("parent_classifier_state", cpu_parent),
                                                        ("private_heads_state", cpu_private))},
                     provenance={name: self.record[name] for name in
                                 ("protocol_sha256", "original_source_sha256", "runtime_settings",
                                  "evidence_sha256", "source_failed_run", "helper_sha256")},
                     selection_rule="First observed P/R initial probability mismatch in original Val order; no GT selection",
                     observation_limit="Python profiling/retained tensors can alter timing and allocation history")
        path = self.output / "scene.pt"
        torch.save(scene, path)
        self.record.update(scene_captured=True, scene_file="scene.pt", scene_sha256=digest(path),
                           scene_image=scene["image"], scene_index=scene["index"],
                           captured_checks=checks, failure_kind="initial_probability_gate")
        write_json(self.output / "check.json", self.record)

    @contextmanager
    def installed(self):
        self.previous = sys.getprofile()
        self.hook = self.heads["p"].register_forward_hook(self.p_output)
        try:
            sys.setprofile(self.profile)
            yield self
        finally:
            sys.setprofile(self.previous)
            self.hook.remove()
            self.record["observer_restored"] = dict(profile=sys.getprofile() is self.previous,
                                                   p_hook=self.hook.id not in self.heads["p"]._forward_hooks)
            self.hook = None


def run(cli):
    started = time.perf_counter()
    repo, output = Path(cli.repo).resolve(), Path(cli.output).resolve()
    if not (repo / "bda_benchmark").is_dir():
        raise ValueError("Repository has no bda_benchmark")
    paths, hashes, old_source, names = bind_failed_evidence(cli.failed_run, cli.protocol, output, repo)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(format=FORMAT, mode=cli.mode, complete=False, diagnostic_only=True,
                  full_validation_evidence=False, training=False, test_accessed=False,
                  protocol_sha256=hashes["protocol"], original_source_sha256=old_source,
                  failure_kind=None, full_order_passed=False, scene_captured=False,
                  optimizer_steps=0, updates_per_arm={arm: 0 for arm in ARMS},
                  phase="load_source", evidence_sha256=hashes,
                  val_order_sha256=hashlib.sha256(json.dumps(names, ensure_ascii=False,
                                                   separators=(",", ":")).encode()).hexdigest(),
                  source_failed_run=str(Path(cli.failed_run).resolve()),
                  scope="One fixed full-order initialization attempt; no retry or fitting",
                  runtime_changed=False, threshold_changed=False,
                  helper_sha256=digest(__file__), original_evaluator_report="initial_val/result.json")
    write_json(output / "check.json", record)
    sys.path.insert(0, str(repo))
    source_function = None
    try:
        import torch
        from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1 import core, evaluate, train
        source_function = core.source_hash
        if core.source_hash() != old_source:
            raise ValueError("Original executable source changed since failed fit")
        old_args = read_object(paths["manifest"]).get("arguments", {})
        lr = old_args.get("lr", 1e-4)
        train.validate_training_config(old_args.get("steps", train.FORMAL_STEPS), lr, False)
        # This order mirrors old train.run, including all three optimizers and
        # the first parent-parameter check. No optimizer is advanced.
        bundle = core.load_source(paths["protocol"], output)
        if bundle["protocol_sha256"] != hashes["protocol"] or bundle["val_names"] != names:
            raise ValueError("Loaded source/protocol or complete Val order changed")
        heads = core.make_heads(bundle["front"]).to(bundle["device"])
        optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), lr=lr, weight_decay=5e-3)
                      for arm in core.TRAIN_ARMS}
        frozen = [bundle["front"], bundle["metric"], *bundle["decoders"].values()]
        train.validate_trainable_heads(heads, optimizers, frozen)
        record["initial_parameter_checks"] = core.initial_parameter_checks(bundle["front"], heads)
        if record["initial_parameter_checks"].get("passed") is not True:
            raise ValueError("Private heads do not match their frozen parents")
        record.update(runtime_settings=bundle["runtime_settings"], phase="initial_Val450")
        write_json(output / "check.json", record)
        if cli.mode == "full":
            original = evaluate.evaluate_heads(bundle, heads, output=output / "initial_val", limit=0, initial=True)
            core.assert_formal_report(original, smoke=False)
            if not full_initial_pass(original):
                raise ValueError("Initial evaluator report did not satisfy all original formal gates")
            record.update(complete=True, full_order_passed=True, phase="full_order_passed",
                          outcome="One uninstrumented Val450 initial check passed; no fitting executed")
        else:
            from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import feature_branches
            observer = SceneObserver(feature_branches, core.head_probability, evaluate.initial_closeness,
                                     heads, bundle["front"], output, record, names)
            record["instrumentation"] = "sys.setprofile and P forward hook; actual outputs retained, never recomputed before failure"
            record["original_evaluator_report_is_instrumented"] = True
            try:
                with observer.installed():
                    original = evaluate.evaluate_heads(bundle, heads, output=output / "initial_val", limit=0, initial=True)
                record["instrumented_pool_guard"] = read_object(output / "initial_val/pool_debug/guard.json")
                if not all(record.get("observer_restored", {}).values()):
                    raise RuntimeError("Observer restoration checks failed")
                # Even a complete instrumented evaluator can never authorize fitting.
                record.update(complete=True, phase="capture_unreproduced", outcome="All 450 images had no observed P/R mismatch in instrumented process; not a fix")
            except CapturedSceneStop:
                pool = read_object(output / "initial_val/pool_debug/guard.json")
                record["instrumented_pool_guard"] = pool
                if (pool.get("restored") is not True or pool.get("failures") != []
                        or pool.get("restore_conflicts") != []
                        or not all(record.get("observer_restored", {}).values())):
                    raise RuntimeError("Instrumented capture restoration checks failed")
                record.update(complete=True, phase="capture_complete",
                              outcome="Instrumented Val prefix stopped at first actual P/R mismatch; scene is diagnostic only")
        return record
    except Exception as error:
        record.update(error=type(error).__name__ + ": " + str(error), phase="failed",
                      complete=False, failure_kind="other")
        result_path = output / "initial_val/result.json"
        if cli.mode == "full" and result_path.is_file():
            result = read_object(result_path)
            if isolated_initial_probability_failure(result):
                record["failure_kind"] = "initial_probability_gate"
        return record
    finally:
        # Do not retry GPU work, fingerprint CUDA tensors or query allocator state
        # after any error; these checks use only CPU file I/O.
        record["old_evidence_unchanged"] = {}
        for name, expected in hashes.items():
            try:
                record["old_evidence_unchanged"][name] = digest(paths[name]) == expected
            except Exception as error:
                record["old_evidence_unchanged"][name] = False
                record.setdefault("cpu_check_errors", {})[name] = str(error)
        try:
            record["original_code_unchanged"] = bool(source_function and source_function() == old_source)
            record["helper_unchanged"] = digest(__file__) == record["helper_sha256"]
        except Exception as error:
            record["original_code_unchanged"] = False
            record["cpu_code_check_error"] = str(error)
        if (not all(record["old_evidence_unchanged"].values())
                or record["original_code_unchanged"] is not True or record.get("helper_unchanged") is not True):
            record.update(complete=False, full_order_passed=False, failure_kind="other",
                          evidence_or_code_changed=True)
        record["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "check.json", record)
        print("INITIAL_SEQUENCE " + json.dumps({key: record.get(key) for key in
              ("mode", "complete", "failure_kind", "full_order_passed", "scene_captured", "phase", "error")},
              ensure_ascii=False), flush=True)


if __name__ == "__main__":
    result = run(parser().parse_args())
    raise SystemExit(0 if result["complete"] else 1)
