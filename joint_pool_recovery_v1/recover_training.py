"""New, explicitly labelled continuation of a validated failed joint run.

The original training package, failed output and saved checkpoint are read-only.
Changing the frozen pooling implementation invalidates any claim of an exact
historical resume, even though weights, optimizer, RNG and observed order are
restored. This module is an engine; the release entry point validates artifacts
and selects the candidate runtime before calling it.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def _batch_name(batch):
    if isinstance(batch, str):
        return batch
    if isinstance(batch, (tuple, list)) and len(batch) >= 5:
        names = batch[4]
        if isinstance(names, (tuple, list)) and len(names) == 1:
            return str(names[0])
    raise ValueError("Cannot verify image identity in reconstructed single-image batch")


def _checked_ledger(module, ledger):
    result = {}
    for path, expected in ledger.items():
        path = str(Path(path).expanduser().resolve())
        if not isinstance(expected, str) or module.digest(path) != expected:
            raise ValueError("Saved input fingerprint changed: " + path)
        result[path] = expected
    return result


def _merge_ledger(module, inherited, runtime, *, verify=False):
    result = dict(inherited)
    for path, expected in getattr(runtime, "input_ledger", {}).items():
        path = str(Path(path).expanduser().resolve())
        if path in result and result[path] != expected:
            raise ValueError("Candidate runtime input fingerprint differs: " + path)
        result[path] = expected
    # The inherited files are fully verified before any update and again at
    # final integrity. Rehashing all image files at every checkpoint creates
    # substantial redundant I/O; checkpointing only merges immutable bindings.
    return _checked_ledger(module, result) if verify else result


def _history_state(module, saved, history_rows, args):
    arms = tuple(module.ARMS)
    step = saved.get("step")
    if type(step) is not int or step < 1 or step >= int(args.steps):
        raise ValueError("Continuation requires an incomplete positive-step checkpoint")
    if saved.get("updates_per_arm") != {a: step for a in arms}:
        raise ValueError("Checkpoint does not contain equal updates for all three arms")
    if set(saved.get("heads", {})) != set(arms) or set(saved.get("optimizers", {})) != set(arms):
        raise ValueError("Checkpoint must preserve all three heads and optimizers")
    if saved.get("mode") != args.mode:
        raise ValueError("Continuation mode differs from checkpoint")
    config = saved.get("config", {})
    for key in ("steps", "seed", "topk", "chunk_size", "lr", "weight_decay", "eval_every"):
        if config.get(key) != getattr(args, key):
            raise ValueError("Continuation changes checkpoint configuration: " + key)
    if len(history_rows) != step:
        raise ValueError("Training journal must have exactly the checkpoint prefix")
    coverage = module._blank_coverage()
    gradients = {a: {"finite_updates": 0, "positive_gradient_updates": 0,
                     "reader_positive_gradient_updates": 0} for a in arms}
    seen = set()
    for number, row in enumerate(history_rows, 1):
        if row.get("step") != number or not isinstance(row.get("image"), str):
            raise ValueError("Training journal steps/image identities are not a complete prefix")
        if type(row.get("epoch")) is not int or row["epoch"] < 0:
            raise ValueError("Training journal epoch is invalid")
        expected_lr = module._lr(number, int(args.steps), float(args.lr))
        if not math.isclose(float(row.get("lr", math.nan)), expected_lr, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Training journal learning-rate schedule differs at step %d" % number)
        if set(row.get("heads", {})) != set(arms) or set(row.get("coverage", {})) != set(arms):
            raise ValueError("Training journal must include all three arms")
        for arm in arms:
            value = row["heads"][arm]
            scalars = [float(value.get(k, math.nan)) for k in ("loss", "grad_norm", "reader_grad_abs_sum")]
            if not all(math.isfinite(x) for x in scalars) or any(x < 0 for x in scalars[1:]):
                raise ValueError("Invalid saved gradient evidence: " + arm)
            gradients[arm]["finite_updates"] += 1
            gradients[arm]["positive_gradient_updates"] += int(scalars[1] > 0)
            gradients[arm]["reader_positive_gradient_updates"] += int(scalars[2] > 0)
            values = row["coverage"][arm]
            if set(values) != {"queries", "supported_queries", "valid_slots"}:
                raise ValueError("Invalid saved coverage schema: " + arm)
            if any(type(v) is not int or v < 0 for v in values.values()):
                raise ValueError("Invalid saved coverage value: " + arm)
            if values["supported_queries"] > values["queries"]:
                raise ValueError("Saved support coverage exceeds query count")
        module._merge_coverage(coverage, row["coverage"])
        seen.add(row["image"])
    for arm in arms:
        if gradients[arm]["positive_gradient_updates"] == 0:
            raise ValueError("Saved prefix has no nonzero gradient: " + arm)
        if step > 1 and gradients[arm]["reader_positive_gradient_updates"] == 0:
            raise ValueError("Saved prefix has no Reader gradient after initialization: " + arm)
    return step, coverage, gradients, seen


def _reconstruct_iterator(runtime, names, rows, seed):
    epoch, cursor = 0, 0
    iterator = iter(runtime.batches(names, train=True, seed=int(seed)))
    total = len(rows)
    print("Checking saved training data order on CPU: 0/%d; no optimizer updates" % total, flush=True)
    for index, row in enumerate(rows, 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            cursor = 0
            iterator = iter(runtime.batches(names, train=True, seed=int(seed) + epoch))
            try:
                batch = next(iterator)
            except StopIteration:
                raise ValueError("Reconstructed training loader returned no batches")
        cursor += 1
        if _batch_name(batch) != row["image"] or epoch != row["epoch"]:
            raise ValueError("Reconstructed data order differs at saved step %d" % row["step"])
        if index % 250 == 0 or index == total:
            print("Saved training data order checked %d/%d (epoch=%d, cursor=%d); no optimizer updates" %
                  (index, total, epoch, cursor), flush=True)
    return epoch, cursor, iterator


def _restore_rng(saved):
    rng = saved.get("rng", {})
    if set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Checkpoint RNG state is incomplete")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if rng["torch_cuda"]:
        if not torch.cuda.is_available() or len(rng["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG device count differs")
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])


def _checkpoint(module, folder, filename, heads, optimizers, step, args, protocol,
                initial_hashes, original_source_hash, ledger, binding, resume_state):
    path = folder / filename
    record = {"format": module.FORMAT, "step": step, "mode": args.mode,
              "config": module._config(args), "smoke": args.mode == "smoke",
              "training_complete": step == int(args.steps), "source_hash": original_source_hash,
              "protocol_info": protocol, "initial_head_hashes": initial_hashes,
              "input_file_sha256": ledger,
              "heads": {a: h.state_dict() for a, h in heads.items()},
              "optimizers": {a: o.state_dict() for a, o in optimizers.items()},
              "updates_per_arm": {a: step for a in module.ARMS},
              "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch_cpu": torch.get_rng_state(),
                      "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
              "test_accessed": False, "training_resume_supported": False,
              "continuation_state_saved": True, "resume_state": copy.deepcopy(resume_state),
              "recovery_binding": binding, "exact_historical_resume_claim": False}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, temporary)
    os.replace(temporary, path)
    manifest = {"format": module.FORMAT, "file": filename, "sha256": module.digest(path),
                "step": step, "mode": args.mode, "smoke": args.mode == "smoke",
                "training_complete": step == int(args.steps), "source_hash": original_source_hash,
                "protocol_info": protocol, "recovery_binding": binding,
                "exact_historical_resume_claim": False,
                "head_hashes": {a: module.module_hash(h) for a, h in heads.items()}}
    module.write_json(path.with_suffix(path.suffix + ".json"), manifest)
    return manifest


def continue_training(original_train_module, args, runtime_factory, saved_checkpoint,
                      history_rows, recovery_binding):
    """Continue all three validated heads in a NEW candidate experiment directory."""
    module = original_train_module
    saved = saved_checkpoint
    smoke = args.mode == "smoke"
    if args.mode not in ("smoke", "train") or (not smoke and int(args.steps) != 5000):
        raise ValueError("Formal continuation requires the original 5000-update budget")
    if int(args.eval_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Positive eval_every and learning rate are required")
    saved_step, coverage, gradients, train_seen = _history_state(module, saved, history_rows, args)
    inherited_ledger = _checked_ledger(module, saved.get("input_file_sha256", {}))
    if module.source_hash() != saved.get("source_hash"):
        raise ValueError("Original training source binding differs from saved checkpoint")
    engine_sha256 = module.digest(__file__)
    binding = copy.deepcopy(recovery_binding)
    binding.update(engine_sha256=engine_sha256, exact_historical_resume_claim=False,
                   experiment_kind="candidate_pool_new_continuation", prefix_updates=saved_step)
    output = module._new_output(args.output)
    summary = {"format": module.FORMAT, "phase": "initializing_continuation", "complete": False,
               "execution_complete": False, "full_validation_evidence": False,
               "smoke": smoke, "training": True, "test_accessed": False,
               "updates_per_arm": {a: saved_step for a in module.ARMS},
               "prefix_updates_per_arm": {a: saved_step for a in module.ARMS},
               "new_updates_per_arm": {a: 0 for a in module.ARMS},
               "exact_historical_resume_claim": False,
               "original_run_complete": False, "recovery_binding": binding,
               "arguments": module._config(args)}
    module.write_json(output / "summary.json", summary)
    try:
        module._seed(int(args.seed))
        with runtime_factory(args, output) as runtime:
            module._seed(int(args.seed))
            train_names, val_names = module._names(runtime, smoke)
            original_protocol = copy.deepcopy(saved["protocol_info"])
            original_protocol.pop("recovery_binding", None)
            runtime_original_protocol = copy.deepcopy(runtime.protocol_info)
            runtime_original_protocol.pop("recovery_binding", None)
            if runtime_original_protocol != original_protocol:
                raise ValueError("Candidate runtime source/bank/data binding differs from checkpoint")
            if not train_seen <= set(train_names):
                raise ValueError("Saved training history includes images outside Train selection")
            heads, _ = module._new_heads(args, runtime.device)
            optimizers = {a: torch.optim.AdamW(h.parameters(), lr=args.lr,
                                              weight_decay=args.weight_decay) for a, h in heads.items()}
            for arm in module.ARMS:
                heads[arm].load_state_dict(saved["heads"][arm], strict=True)
                optimizers[arm].load_state_dict(saved["optimizers"][arm])
                if not all(bool(torch.isfinite(p).all()) for p in heads[arm].parameters()):
                    raise ValueError("Loaded checkpoint parameters are nonfinite: " + arm)
            protocol = copy.deepcopy(runtime.protocol_info)
            protocol["recovery_binding"] = binding
            initial_hashes = saved["initial_head_hashes"]
            module.write_json(output / "manifest.json", {
                "format": module.FORMAT, "source_hash": saved["source_hash"],
                "protocol_info": protocol, "arguments": module._config(args),
                "train_names": train_names, "val_names": val_names,
                "initial_head_hashes": initial_hashes, "recovery_binding": binding,
                "loaded_head_hashes": {a: module.module_hash(h) for a, h in heads.items()},
                "claims": {"exact_historical_resume_claim": False,
                           "shared_frozen_forward": True, "query_GT_not_model_input": True,
                           "new_continuation_experiment": True}})
            summary["phase"] = "checkpoint_revalidation"
            module.write_json(output / "summary.json", summary)
            checkpoint_eval = module._evaluate(runtime, heads, val_names, output / "checkpoint_reval",
                                               seed=args.seed, smoke=smoke, save_arrays=True)
            integrity = module._checked_verify(runtime)
            checkpoint_eval.update(checkpoint_step=saved_step, primary_endpoint=False,
                                   full_validation_evidence=False, final_integrity_pending=False,
                                   final_integrity=integrity, recovery_binding=binding,
                                   exact_historical_resume_claim=False)
            module.write_json(output / "checkpoint_reval" / "result.json", checkpoint_eval)
            summary["checkpoint_revalidation_complete"] = True
            summary["checkpoint_revalidation_integrity"] = integrity
            _merge_ledger(module, inherited_ledger, runtime)
            summary["phase"] = "reconstructing_training_iterator"
            module.write_json(output / "summary.json", summary)
            epoch, cursor, iterator = _reconstruct_iterator(runtime, train_names, history_rows, args.seed)
            # Revalidation and prefix loading may consume RNG. Restore immediately
            # before the next training batch/update, never before those operations.
            _restore_rng(saved)
            summary["phase"] = "training_continuation"
            module.write_json(output / "summary.json", summary)
            start = time.monotonic()
            if str(runtime.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(runtime.device)
            checkpoints = output / "checkpoints"
            checkpoints.mkdir()
            with (output / "train.jsonl").open("w", encoding="utf-8") as journal:
                for prior_row in history_rows:
                    journal.write(json.dumps(prior_row, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                for step in range(saved_step + 1, int(args.steps) + 1):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        epoch += 1
                        cursor = 0
                        iterator = iter(runtime.batches(train_names, train=True, seed=int(args.seed) + epoch))
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            raise ValueError("Training loader returned no batches")
                    cursor += 1
                    example = runtime.prepare(batch)
                    module._assert_example(example)
                    if example.image not in train_names or _batch_name(batch) != example.image:
                        raise ValueError("Training loader image identity differs from Train selection/batch")
                    train_seen.add(example.image)
                    image_coverage = module._coverage(example)
                    module._merge_coverage(coverage, image_coverage)
                    lr = module._lr(step, int(args.steps), float(args.lr))
                    row = {"step": step, "epoch": epoch, "image": example.image, "event": example.event,
                           "lr": lr, "heads": {}, "coverage": image_coverage,
                           "recovery_continuation": True}
                    for arm in module.ARMS:
                        head, optimizer = heads[arm], optimizers[arm]
                        head.train()
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                        optimizer.zero_grad(set_to_none=True)
                        result = head(**example.common, **example.supports[arm])
                        loss = runtime.loss(result["probabilities"], example.target)
                        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                            raise ValueError("Non-finite or non-scalar training loss: " + arm)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0,
                                                                  error_if_nonfinite=True)
                        evidence = gradients[arm]
                        evidence["finite_updates"] += 1
                        evidence["positive_gradient_updates"] += int(float(grad_norm) > 0)
                        reader_grad = sum(float(p.grad.detach().abs().sum()) for name, p in head.named_parameters()
                                          if p.grad is not None and name.startswith(("query_projection",
                                             "key_projection", "value_projection", "label_embedding",
                                             "context_projection")))
                        evidence["reader_positive_gradient_updates"] += int(reader_grad > 0)
                        optimizer.step()
                        if not all(bool(torch.isfinite(p).all()) for p in head.parameters()):
                            raise ValueError("Non-finite updated parameters: " + arm)
                        summary["updates_per_arm"][arm] = step
                        summary["new_updates_per_arm"][arm] = step - saved_step
                        row["heads"][arm] = {"loss": float(loss.detach()), "grad_norm": float(grad_norm),
                                             "reader_grad_abs_sum": reader_grad}
                        del result, loss
                    row["elapsed_seconds"] = time.monotonic() - start
                    row["peak_cuda_allocated_bytes"] = (int(torch.cuda.max_memory_allocated(runtime.device))
                                                        if str(runtime.device).startswith("cuda") else None)
                    journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    if step % 20 == 0 or step == int(args.steps):
                        print("Joint recovery fit " + json.dumps(row, ensure_ascii=False), flush=True)
                    if step % int(args.eval_every) == 0 or step == int(args.steps):
                        name = "checkpoint_last.pt" if step == int(args.steps) else "checkpoint_%06d.pt" % step
                        ledger = _merge_ledger(module, inherited_ledger, runtime)
                        resume_state = {"epoch": epoch, "cursor": cursor,
                                        "coverage": coverage, "gradient_evidence": gradients,
                                        "train_seen": sorted(train_seen),
                                        "journal_rows": step, "prefix_updates": saved_step}
                        manifest = _checkpoint(module, checkpoints, name, heads, optimizers, step, args,
                                               protocol, initial_hashes, saved["source_hash"], ledger,
                                               binding, resume_state)
                        summary.update(last_checkpoint=str(checkpoints / name), checkpoint=manifest)
                        if step == int(args.steps):
                            summary["training_complete"] = True
                        else:
                            summary["phase"] = "intermediate_val"
                            module.write_json(output / "summary.json", summary)
                            evaluation = module._evaluate(runtime, heads, val_names,
                                        output / ("val_step_%06d" % step), seed=args.seed,
                                        smoke=smoke, save_arrays=False)
                            evaluation.update(primary_endpoint=False, recovery_binding=binding,
                                              exact_historical_resume_claim=False)
                            module.write_json(output / ("val_step_%06d" % step) / "result.json", evaluation)
                            summary["phase"] = "training_continuation"
                        module.write_json(output / "summary.json", summary)
                    del example
            if any(coverage[a]["supported_queries"] == 0 for a in ("rt", "rf")):
                raise ValueError("No legal historical support was consumed during training")
            if not smoke and train_seen != set(train_names):
                raise ValueError("Formal continuation did not visit all Train2918 query images")
            for arm in module.ARMS:
                if gradients[arm]["finite_updates"] != int(args.steps):
                    raise ValueError("Total gradient evidence does not match full budget: " + arm)
                if gradients[arm]["positive_gradient_updates"] == 0 or gradients[arm]["reader_positive_gradient_updates"] == 0:
                    raise ValueError("No nonzero training/Reader gradient: " + arm)
            summary["phase"] = "final_val"
            module.write_json(output / "summary.json", summary)
            if module.digest(summary["last_checkpoint"]) != summary["checkpoint"]["sha256"]:
                raise RuntimeError("Last checkpoint hash changed before final evaluation")
            last = torch.load(summary["last_checkpoint"], map_location="cpu", weights_only=False)
            for arm in module.ARMS:
                heads[arm].load_state_dict(last["heads"][arm], strict=True)
            final = module._evaluate(runtime, heads, val_names, output / "final_val", seed=args.seed,
                                     smoke=smoke, save_arrays=True)
            summary["phase"] = "final_integrity"
            module.write_json(output / "summary.json", summary)
            integrity = module._checked_verify(runtime)
            if module.source_hash() != saved["source_hash"] or module.digest(__file__) != engine_sha256:
                raise RuntimeError("Original or recovery engine code changed while experiment was running")
            if module.digest(summary["last_checkpoint"]) != summary["checkpoint"]["sha256"]:
                raise RuntimeError("Final checkpoint changed")
            if any(module.module_hash(heads[a]) != summary["checkpoint"]["head_hashes"][a] for a in module.ARMS):
                raise RuntimeError("Final heads differ from saved last checkpoint")
            fingerprints = _merge_ledger(module, inherited_ledger, runtime, verify=True)
            summary.update(phase="complete_smoke" if smoke else "complete_new_continuation",
                           complete=True, execution_complete=True, full_validation_evidence=not smoke,
                           primary_endpoint=not smoke,
                           primary_endpoint_kind="new_repaired_continuation" if not smoke else "smoke",
                           image_count=len(val_names), train_image_count=len(train_seen),
                           training_coverage=coverage, gradient_evidence=gradients, integrity=integrity,
                           metrics=final["metrics"], elapsed_continuation_and_final_seconds=time.monotonic()-start,
                           feature_operator_updates_per_arm={a: {"original_operator_prefix": saved_step,
                               "candidate_operator_suffix": int(args.steps)-saved_step} for a in module.ARMS},
                           comparisons_pp={"r0_minus_grid_probq": 100*(final["metrics"]["r0"]["foreground_miou"]-final["metrics"]["grid_probq"]["foreground_miou"]),
                                           "rt_minus_r0": 100*(final["metrics"]["rt"]["foreground_miou"]-final["metrics"]["r0"]["foreground_miou"]),
                                           "rf_minus_r0": 100*(final["metrics"]["rf"]["foreground_miou"]-final["metrics"]["r0"]["foreground_miou"])})
        summary["guard_report"] = module._restored_guard(runtime)
        module.write_json(output / "input_fingerprints.json", fingerprints)
        summary["input_fingerprints_sha256"] = module.digest(output / "input_fingerprints.json")
        final.update(final_integrity_pending=False, full_validation_evidence=not smoke,
                     primary_endpoint=not smoke,
                     primary_endpoint_kind="new_repaired_continuation" if not smoke else "smoke",
                     final_integrity=integrity, guard_report=summary["guard_report"],
                     recovery_binding=binding, exact_historical_resume_claim=False,
                     checkpoint_step=int(args.steps))
        module.write_json(output / "final_val" / "result.json", final)
        module.write_json(output / "summary.json", summary)
        print("JOINT NEW CONTINUATION COMPLETE. No exact historical resume claim. No Test.", flush=True)
        return summary
    except BaseException as exc:
        module._failure(output, summary, exc)
        raise
