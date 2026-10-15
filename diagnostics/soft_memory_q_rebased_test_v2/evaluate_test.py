"""Exploratory Test878 for one explicitly selected, frozen fix2 checkpoint.

This is independent of the final-only v1 protocol. Training artifacts are
read-only; completion here certifies this Test pass, not completion of training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

import source_support as support

require = support.require
read_json = support.read_json
ARMS = support.ARMS
HEADS = support.JOINT_ARMS
FORMAT = "SRAC_SOFT_MEMORY_Q_REBASED_EXPLORATORY_TEST_V2"
DEFAULT_TOOL = "/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2"
DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def verify_files(ledger, label):
    require(isinstance(ledger, dict) and bool(ledger), label + " ledger is empty")
    for index, (name, expected) in enumerate(ledger.items(), 1):
        require(isinstance(name, str) and Path(name).is_absolute()
                and isinstance(expected, str) and support.re.fullmatch(r"[0-9a-f]{64}", expected),
                "Invalid " + label + " ledger entry")
        require(Path(name).is_file() and digest(name) == expected,
                label + " file missing or changed: " + name)
        if index % 1000 == 0:
            print("VERIFY_%s %d/%d" % (label, index, len(ledger)), flush=True)


def validate_checkpoint(root, step, tool):
    """Use immutable checkpoint evidence, never current progress as step evidence."""
    root = Path(root).expanduser().resolve(strict=True)
    require(type(step) is int and 0 < step <= 5000, "Checkpoint step must be 1..5000")
    train = root / "train"
    name = "checkpoint_last.pt" if step == 5000 else "checkpoint_%06d.pt" % step
    checkpoint = train / "checkpoints" / name
    require(checkpoint.is_file(), "Requested checkpoint is missing; no fallback: " + str(checkpoint))
    sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    sidecar = read_json(sidecar_path)
    manifest_path = train / "manifest.json"
    manifest = read_json(manifest_path)
    code = tool.common.source_hash()
    require(manifest.get("format") == tool.common.FORMAT and manifest.get("source_hash") == code,
            "Training manifest/code differs from current fix2")
    for label, count in (("train_names", 2918), ("val_names", 450)):
        names = manifest.get(label)
        require(isinstance(names, list) and len(names) == count and len(set(names)) == count
                and all(isinstance(n, str) and support.SAFE_NAME.fullmatch(n) for n in names),
                "Invalid training manifest split: " + label)
    require(not set(manifest["train_names"]) & set(manifest["val_names"]), "Train/Val overlap")
    require(sidecar.get("format") == tool.common.FORMAT and sidecar.get("file") == name
            and sidecar.get("sha256") == digest(checkpoint) and sidecar.get("step") == step
            and sidecar.get("source_hash") == code and sidecar.get("mode") == "train"
            and sidecar.get("smoke") is False
            and sidecar.get("training_complete") is (step == 5000)
            and sidecar.get("protocol_info") == manifest.get("protocol_info"),
            "Checkpoint sidecar, hash, step or provenance differs")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(saved, dict) and saved.get("format") == tool.common.FORMAT
            and saved.get("source_hash") == code and saved.get("step") == step
            and saved.get("mode") == "train" and saved.get("smoke") is False
            and saved.get("training_complete") is (step == 5000)
            and saved.get("test_accessed") is False
            and saved.get("updates_per_arm") == {a: step for a in HEADS}
            and saved.get("protocol_info") == sidecar.get("protocol_info")
            and saved.get("config") == manifest.get("arguments")
            and saved.get("old_three_arm_baseline") == "not_comparable_new_Q_source"
            and set(saved.get("heads", {})) == set(HEADS),
            "Serialized checkpoint differs from the matched training evidence")
    require(set(sidecar.get("head_hashes", {})) == set(HEADS)
            and all(support.state_hash(saved["heads"][a]) == sidecar["head_hashes"][a] for a in HEADS),
            "Saved A/B/C/D tensor hash differs")
    config = saved["config"]
    require(config.get("steps") == 5000 and config.get("topk") == 64
            and config.get("key_name") in ("t128", "f256")
            and config.get("baseline_policy", "independent") == "independent"
            and config.get("baseline_root") is None and config.get("baseline_evidence") is None
            and config.get("allow_smoke_values") is False
            and Path(config["output"]).resolve() == train
            and Path(config["bank_root"]).resolve() == root / "parent"
            and Path(config["values"]).resolve() == root / "values",
            "Checkpoint configuration does not bind the full independent four-arm run")
    require(Path(config["protocol"]).resolve() == Path(saved["protocol_info"]["source_protocol"]).resolve()
            and digest(config["protocol"]) == saved["protocol_info"]["source_protocol_sha256"],
            "Saved source protocol path/hash differs")
    verify_files(saved.get("input_file_sha256"), "CHECKPOINT_INPUTS")
    # No current train/summary or workflow completion requirement: they may
    # legitimately advance while this already-saved checkpoint is evaluated.
    frozen_paths = [manifest_path, checkpoint, sidecar_path, Path(config["protocol"])]
    hashes = {str(p.resolve()): digest(p) for p in frozen_paths}
    return SimpleNamespace(root=root, train=train, manifest=manifest, saved=saved,
        checkpoint=checkpoint, checkpoint_manifest=sidecar, hashes=hashes, step=step)


def configure_runtime(formal, threads):
    require(threads > 0, "CPU threads must be positive")
    settings = formal.saved["protocol_info"]["execution_settings"]
    require(torch.__version__ == settings["torch_version"] and torch.version.cuda == settings["cuda_version"],
            "PyTorch/CUDA versions must match the checkpoint's recorded runtime")
    require(not torch.cuda.is_initialized(), "Configure the saved numeric profile before CUDA initialization")
    for key, field in (("CUDA_LAUNCH_BLOCKING", "cuda_launch_blocking"),
                       ("CUBLAS_WORKSPACE_CONFIG", "cublas_workspace_config")):
        value = settings[field]
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(settings["deterministic_algorithms"])
    torch.set_float32_matmul_precision(settings["float32_matmul_precision"])
    torch.backends.cudnn.allow_tf32 = settings["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = settings["matmul_allow_tf32"]


def test_runtime_class(tool, names, dataset, formal):
    """Explicit Test loader/extraction; original Train and Val sets stay intact."""
    allowed = frozenset(names)
    class TestRuntime(tool.server_runtime.Runtime):
        def __enter__(self):
            super().__enter__()
            try:
                require(self.protocol_info == formal.saved["protocol_info"],
                        "Runtime/source/bank/numeric profile differs from saved checkpoint")
                require(self.train_names == formal.manifest["train_names"]
                        and self.val_names == formal.manifest["val_names"], "Runtime Train/Val split changed")
                for path, sha in self.input_ledger.items():
                    require(formal.saved["input_file_sha256"].get(str(path)) == sha,
                            "Loaded source is not bound by checkpoint: " + str(path))
                for name in names:
                    actual = {k: Path(p).resolve() for k, p in self.deps.sample_paths(dataset, name).items()}
                    expected = {"pre": dataset / "pre-event" / (name + "_pre_disaster.tif"),
                        "post": dataset / "post-event" / (name + "_post_disaster.tif"),
                        "label": dataset / "target" / (name + "_building_damage.tif")}
                    require(actual == expected, "Unexpected Test data mapping: " + name)
                return self
            except BaseException:
                self.close(*sys.exc_info())
                raise

        def batches(self, requested, train=False, seed=3407):
            require(train is False and list(requested) == list(names),
                    "Test loader must use exactly the declared ordered Test878 list")
            print("Hashing %d Test input files before inference" % (3 * len(names)), flush=True)
            for index, name in enumerate(names, 1):
                for path in self.deps.sample_paths(dataset, name).values():
                    path = Path(path).resolve()
                    self.input_ledger.setdefault(path, digest(path))
                if index % 100 == 0 or index == len(names):
                    print("TEST_INPUT_HASHES %d/%d images" % (index, len(names)), flush=True)
            return self.deps.full_image_loader(dataset, list(names), False, self.args.workers, seed)

        @torch.no_grad()
        def extract(self, batch):
            # Same frozen forward/slot calculation as FrozenSlotRuntime.extract;
            # only split authorization is explicit Test, without rewriting Val.
            d, module = self.deps, tool.server_runtime
            module.validate_raw_labels(batch)
            pre, post, _loc, target, batch_names = d.parse_batch(batch, self.device)
            require(len(pre) == 1 and len(batch_names) == 1, "Test expects one full image")
            name = str(batch_names[0])
            require(name in allowed and name not in self._train_ids and name not in self._val_set,
                    "Query is outside disjoint Test878")
            for path in d.sample_paths(dataset, name).values():
                path = Path(path).resolve()
                if path not in self.input_ledger:
                    self.input_ledger[path] = digest(path)
            self.guard.set_image(name)
            self.native_capture.clear()
            self.building_capture.clear()
            result, feature = self.capture(pre, post)
            logits, building = self.native_capture.take(), self.building_capture.take()
            baseline, check = d.native_grid_probq(result, feature, logits, target.shape[-2:])
            require(check.get("passed") is True, "Native probability check failed: " + name)
            pair = d.extract_pair(result, feature)  # No Test GT in key or feature extraction.
            require(pair["q_check"].get("passed") is True, "Frozen Q check failed: " + name)
            slots = {rule: module.prepare_slots(feature, result["q"], result["ids"], result["count"],
                     tuple(result["shape"]), building, rule) for rule in module.RULES}
            self.prepare_count += 1
            return dict(name=name, event=d.event(name), target=target, result=result, feature=feature,
                        pair=pair, native_logits=logits, building=building, slots=slots,
                        base_probability_full=baseline, native_check=check)
    return TestRuntime


def counts(array, shape):
    value = np.asarray(array)
    require(value.shape == shape and np.issubdtype(value.dtype, np.integer)
            and np.all(value >= 0), "Invalid nonnegative integer count array")
    return value.astype(np.int64)


def reopen_journal(path, names, expected, transitions, events, tool):
    rebuilt = {a: np.zeros((4, 4), dtype=np.int64) for a in ARMS}
    changes = {a: np.zeros((4, 4, 4), dtype=np.int64) for a in HEADS}
    event_cm = {}
    seen = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            require(index < len(names) and row["index"] == index and row["image"] == names[index],
                    "Test journal identity/order differs")
            require(isinstance(row["event"], str) and bool(row["event"]), "Invalid event identity")
            current = {a: counts(row["confusion"][a], (4, 4)) for a in ARMS}
            for a in HEADS:
                t = counts(row["transitions"][a], (4, 4, 4))
                require(np.array_equal(t.sum(2), current["grid_probq"])
                        and np.array_equal(t.sum(1), current[a]), "Transition marginal differs")
                changes[a] += t
            event = event_cm.setdefault(row["event"], {a: np.zeros((4, 4), dtype=np.int64) for a in ARMS})
            for a in ARMS:
                rebuilt[a] += current[a]
                event[a] += current[a]
            seen.append(row["image"])
    require(seen == list(names), "Test journal is incomplete")
    require(all(np.array_equal(rebuilt[a], expected[a]) for a in ARMS)
            and all(np.array_equal(changes[a], transitions[a]) for a in HEADS)
            and set(event_cm) == set(events)
            and all(np.array_equal(event_cm[e][a], events[e][a]) for e in events for a in ARMS),
            "Reopened Test journal counts differ")
    return rebuilt, changes, event_cm


def infer(runtime, heads, names, tool, output, seed):
    confusion = {a: np.zeros((4, 4), dtype=np.int64) for a in ARMS}
    transitions = {a: np.zeros((4, 4, 4), dtype=np.int64) for a in HEADS}
    events = {}
    journal_path = output / "images.jsonl"
    with torch.no_grad(), journal_path.open("x", encoding="utf-8") as journal:
        seen = 0
        for index, batch in enumerate(runtime.batches(names, train=False, seed=seed)):
            example = runtime.prepare(batch)
            tool.train._assert_example(example)
            require(index < len(names) and example.image == names[index], "Unexpected Test image/order")
            target = example.target.detach().cpu().numpy()
            base = example.common["base_probability_full"]
            require(bool(torch.isfinite(base).all()), "Nonfinite grid_probq")
            predictions = {"grid_probq": base.argmax(1).cpu().numpy()}
            for a in HEADS:
                p = heads[a](**example.common, **example.supports[a])["probabilities"]
                require(bool(torch.isfinite(p).all()), "Nonfinite probability: " + a)
                predictions[a] = p.argmax(1).cpu().numpy()
            per_image = {a: tool.train._confusion(target, predictions[a]) for a in ARMS}
            changes = {a: tool.train._transition(target, predictions["grid_probq"], predictions[a]) for a in HEADS}
            event = events.setdefault(example.event, {a: np.zeros((4, 4), dtype=np.int64) for a in ARMS})
            for a in ARMS:
                confusion[a] += per_image[a]
                event[a] += per_image[a]
            for a in HEADS:
                transitions[a] += changes[a]
            row = dict(index=index, image=example.image, event=example.event,
                       confusion={a: per_image[a].tolist() for a in ARMS},
                       transitions={a: changes[a].tolist() for a in HEADS}, diagnostics=example.diagnostics)
            journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            seen += 1
            if seen == 1 or seen % 10 == 0 or seen == len(names):
                print("Test %d/%d; final integrity pending" % (seen, len(names)), flush=True)
                write_json(output / "progress.json", dict(image_count=seen, expected=len(names),
                    complete=False, final_integrity_pending=True, last_image=example.image))
            del example, base, p, predictions, target
        require(seen == len(names), "Test loader ended early")
    return reopen_journal(journal_path, names, confusion, transitions, events, tool)


def metric_result(confusion, transitions, events, tool):
    metric = {a: support.metrics(confusion[a], tool.train) for a in ARMS}
    return dict(metrics=metric,
        per_event={e: {a: support.metrics(cm, tool.train) for a, cm in arms.items()} for e, arms in events.items()},
        transitions={a: tool.train._transition_report(transitions[a]) for a in HEADS},
        same_forward_baseline_deltas_pp={a: 100 * (metric[a]["foreground_miou"] - metric["grid_probq"]["foreground_miou"]) for a in HEADS},
        comparisons_pp=tool.train._comparisons(metric))


def report(folder):
    folder = Path(folder)
    summary = read_json(folder / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("complete") is not True:
        progress = folder / "progress.json"
        if progress.is_file(): print(progress.read_text())
        return
    require(digest(folder / "result.json") == summary["result_sha256"], "Test result file changed")
    result = read_json(folder / "result.json")
    print("checkpoint_step=%s; exploratory Test; training_complete_at_checkpoint=%s" %
          (result["checkpoint_step"], result["training_complete_at_checkpoint"]))
    print("arm | foreground3_mIoU% | fourclass_mIoU% | BG / Intact / Damaged / Destroyed IoU%")
    for a in ARMS:
        m = result["metrics"][a]
        print("%s | %.4f | %.4f | %s" % (a, 100*m["foreground_miou"], 100*m["four_class_miou"],
              " / ".join("NA" if x is None else "%.4f" % (100*x) for x in m["iou"])))
    print("Delta vs same-Test grid_probq (pp):", result["same_forward_baseline_deltas_pp"])
    print("Matched comparisons (pp):", result["comparisons_pp"])


def evaluate(formal, tool, output, threads=2, workers=1):
    output = support._protect_output(output, formal, tool)
    require(not output.is_relative_to(Path(__file__).resolve().parent), "Output must be outside evaluator code")
    output.mkdir(parents=True)
    summary = dict(format=FORMAT, evaluation_kind="exploratory_checkpoint", complete=False,
        full_test_evidence=False, final_integrity_pending=True, test_accessed=False,
        training=False, weights_updated=False, primary_endpoint=False,
        checkpoint_step=formal.step, checkpoint_sha256=formal.checkpoint_manifest["sha256"],
        checkpoint=str(formal.checkpoint), training_complete_at_checkpoint=formal.saved["training_complete"],
        source_final_training_integrity="not_asserted_by_checkpoint_evaluation",
        code_sha256=formal.saved["source_hash"], source_run=str(formal.root),
        saved_predictions_reopened=False, prediction_arrays_saved=False,
        cpu_threads=threads, data_workers=workers, package_manifest_sha256=support.verify_release())
    write_json(output / "summary.json", summary)
    start = time.monotonic()
    try:
        configure_runtime(formal, threads)
        require(workers >= 0, "Workers cannot be negative")
        locked = dict(checkpoint=str(formal.checkpoint), step=formal.step,
            sha256=formal.checkpoint_manifest["sha256"], head_hashes=formal.checkpoint_manifest["head_hashes"],
            source_hashes=formal.hashes, pathways=list(ARMS),
            selection="explicit checkpoint chosen before Test; all five pathways reported",
            selected_on_test=False, exploratory=True)
        write_json(output / "checkpoint_lock.json", locked)
        lock_hash = digest(output / "checkpoint_lock.json")
        summary.update(phase="Test_split_check", test_accessed=True)
        write_json(output / "summary.json", summary)
        test_list, names, dataset = support._canonical_test_split(formal.saved["config"]["protocol"], formal)
        test_hash = digest(test_list)
        args = SimpleNamespace(**formal.saved["config"])
        args.mode, args.smoke, args.output, args.workers = "train", False, str(output), workers
        runtime = test_runtime_class(tool, names, dataset, formal)(args, output)
        heads = {}
        with runtime:
            require(runtime.device.type == "cuda", "This checkpoint requires its original CUDA runtime")
            runtime.input_ledger[test_list.resolve()] = test_hash
            for a in HEADS:
                head = tool.model.JointRegionMemoryClassifier(chunk_size=args.chunk_size).to(runtime.device)
                head.load_state_dict(formal.saved["heads"][a], strict=True)
                head.requires_grad_(False).eval()
                require(tool.common.module_hash(head) == formal.checkpoint_manifest["head_hashes"][a],
                        "Loaded checkpoint head hash differs: " + a)
                heads[a] = head
            summary.update(phase="inference", image_count=0, expected_test_images=878,
                           test_list=str(test_list), test_list_sha256=test_hash)
            write_json(output / "summary.json", summary)
            cm, changes, events = infer(runtime, heads, names, tool, output, args.seed)
            summary.update(phase="final_integrity", image_count=878)
            write_json(output / "summary.json", summary)
            integrity = tool.train._checked_verify(runtime)
            require(all(tool.common.module_hash(heads[a]) == formal.checkpoint_manifest["head_hashes"][a] for a in HEADS),
                    "Frozen head changed during Test")
        guard = runtime.guard_report
        require(isinstance(guard, dict) and guard.get("restored") is True and not guard.get("restore_conflicts"),
                "PoolGuard did not restore cleanly")
        verify_files(formal.hashes, "FROZEN_ARTIFACTS")
        require(tool.common.source_hash() == formal.saved["source_hash"]
                and support._verify_tool_before_import(tool.root, None) == formal.saved["source_hash"],
                "Original training code changed during Test")
        require(support.verify_release() == summary["package_manifest_sha256"], "Test package changed")
        require(digest(test_list) == test_hash and digest(output / "checkpoint_lock.json") == lock_hash,
                "Test split or checkpoint lock changed")
        write_json(output / "input_fingerprints.json", {str(p): h for p, h in runtime.input_ledger.items()})
        result = dict(summary, **metric_result(cm, changes, events, tool))
        result.update(phase="complete", complete=True, execution_complete=True, full_test_evidence=True,
            final_integrity_pending=False, image_count=878, split="Test878",
            journal_aggregates_reopened=True, images_journal_sha256=digest(output / "images.jsonl"),
            input_fingerprints_sha256=digest(output / "input_fingerprints.json"),
            runtime_integrity=integrity, pool_guard=guard, checkpoint_lock_sha256=lock_hash,
            elapsed_seconds=time.monotonic()-start)
        write_json(output / "result.json", result)
        with (output / "metrics.csv").open("x", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["arm", "foreground_mIoU_percent", "four_class_mIoU_percent", "background_IoU_percent", "intact_IoU_percent", "damaged_IoU_percent", "destroyed_IoU_percent"])
            for a in ARMS:
                m = result["metrics"][a]
                writer.writerow([a, 100*m["foreground_miou"], 100*m["four_class_miou"],
                                 *[None if x is None else 100*x for x in m["iou"]]])
        summary.update(phase="complete", complete=True, execution_complete=True, full_test_evidence=True,
            final_integrity_pending=False, image_count=878, result_sha256=digest(output / "result.json"),
            elapsed_seconds=time.monotonic()-start)
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", dict(image_count=878, expected=878,
                   complete=True, final_integrity_pending=False))
        report(output)
        return result
    except BaseException as exc:
        summary.update(phase="failed", complete=False, full_test_evidence=False,
                       error=type(exc).__name__ + ": " + str(exc))
        write_json(output / "summary.json", summary)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "run", "report"))
    p.add_argument("--train-root", default=DEFAULT_ROOT)
    p.add_argument("--tool", default=DEFAULT_TOOL)
    p.add_argument("--step", type=int, default=2500)
    p.add_argument("--output")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--workers", type=int, default=1)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    support.verify_release()
    if args.command == "report":
        require(args.output, "--output required")
        return report(args.output)
    tool = support._load_tool(args.tool, None)
    formal = validate_checkpoint(args.train_root, args.step, tool)
    if args.command == "plan":
        print(json.dumps(dict(ready_for_runtime_check=True, test_accessed=False,
            checkpoint=str(formal.checkpoint), checkpoint_step=formal.step,
            checkpoint_sha256=formal.checkpoint_manifest["sha256"],
            training_complete_at_checkpoint=formal.saved["training_complete"],
            note="CPU source check only; real CUDA inference still untested"), ensure_ascii=False, indent=2))
        return
    require(args.output, "--output required; choose a new output directory")
    return evaluate(formal, tool, args.output, args.cpu_threads, args.workers)


if __name__ == "__main__":
    main()
