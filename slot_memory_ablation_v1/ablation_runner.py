#!/usr/bin/env python3
"""Matched single/dual-slot and support-source training; no Test access."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from runtime_bridge import (
    DEFAULT_SOURCE, check_package, digest, inspect_source, load_runtime,
    named, read_json, runtime_arguments, verify_ledgers, write_json,
)

HERE = Path(__file__).resolve().parent
FORMAT = "BRIGHT_SLOT_MEMORY_ABLATION_V1"
DEFAULT_OUTPUT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_V1"
GROUPS = {
    "core": ("e", "f", "b", "d"),
    "main": ("e", "f", "b", "d", "r", "l"),
    "memory": ("d", "r", "l", "n"),
    "spatial": ("a", "b", "c", "d"),
    "all": ("e", "f", "a", "b", "c", "d", "r", "l", "n"),
}


def options(args):
    arms = tuple(args.arms.lower().split(",")) if args.arms else GROUPS[args.group]
    if len(arms) != len(set(arms)) or not arms or not set(arms) <= set(GROUPS["all"]):
        raise ValueError("Use unique arm IDs from e,f,a,b,c,d,r,l,n")
    if not 1 <= args.steps <= 5000:
        raise ValueError("Endpoint must be 1..5000; scheduler horizon remains 5000")
    if not math.isfinite(args.post500_lr) or not 0 < args.post500_lr <= 1:
        raise ValueError("--post500-lr must be in (0,1]")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise ValueError("Use distinct nonnegative seeds")
    if not 1 <= args.fixed_train <= 2918 or not 1 <= args.smoke_steps <= 20:
        raise ValueError("Invalid fixed Train count/smoke update count")
    return arms, seeds


def schedule_steps(endpoint):
    return sorted({s for s in (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, endpoint)
                   if 0 < s <= endpoint})


def training_batches(runtime, names, seed):
    """Repeat the original finite loader, matching its seed+epoch convention."""
    epoch = 0
    while True:
        found = False
        for batch in runtime.batches(names, train=True, seed=int(seed) + epoch):
            found = True
            yield epoch, batch
        if not found:
            raise RuntimeError("Training loader yielded no samples at epoch %s" % epoch)
        epoch += 1


def addon_release():
    return check_package(HERE)


def plan(args):
    arms, seeds = options(args)
    source = inspect_source(args.source_run, args.package)
    result = dict(
        format=FORMAT, source=source, addon_release_sha256=addon_release(),
        arms=arms, seeds=seeds, endpoint_step=args.steps, schedule_horizon=5000,
        maximum_lr=1e-4, warmup_steps=500, post500_lr_multiplier=args.post500_lr,
        weight_decay=0.005, optimizer="AdamW", grad_clip=1.0,
        common_initialization="identical fresh full joint modules; independent optimizer storage",
        original_NLL_and_Lovasz_unchanged=True, anchor_weight=0.0,
        validation_steps=[0] + schedule_steps(args.steps),
        fixed_train_count=args.fixed_train,
        frozen_bank_reused=True, old_head_weights_reused=False,
        old_training_trajectory_claim=False, training_started=False,
        cuda_forward_verified=False, test_accessed=False,
    )
    return result


def model_copies(torch, original_train, state, arms, config, device):
    prototype = original_train.JointRegionMemoryClassifier(chunk_size=int(config["chunk_size"])).to(device)
    initial = state.state_fingerprint(prototype.state_dict())
    heads, optimizers = {}, {}
    for arm in arms:
        model = copy.deepcopy(prototype)
        if state.state_fingerprint(model.state_dict()) != initial:
            raise ValueError("Fresh model initialization changed")
        heads[arm] = model
        optimizers[arm] = torch.optim.AdamW(
            model.parameters(), lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
    seen = set()
    for head in heads.values():
        for p in head.parameters():
            if p.data_ptr() in seen:
                raise ValueError("Arms share trainable parameter storage")
            seen.add(p.data_ptr())
    return heads, optimizers, initial


def update_arms(heads, optimizers, example, runtime, factory, state, *, step, multiplier):
    """One shared frozen forward; independent equal-budget optimizer updates."""
    import torch
    supports = factory.prepare(example, tuple(heads))
    rng = state.capture_rng()
    after = None
    rows = {}
    lr = state.lr_at_step(step, maximum=1e-4, horizon=5000, multiplier=multiplier)
    for arm, head in heads.items():
        state.restore_rng(rng)
        head.train()
        opt = optimizers[arm]
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        output = factory.forward(head, example, arm, supports[arm])
        loss = runtime.loss(output["probabilities"], example.target)
        if loss.ndim or not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite/nonscalar loss: " + arm)
        with torch.no_grad():
            if bool((example.target != 255).any()):
                nll = runtime.deps.region_nll(output["probabilities"], example.target)
            else:
                nll = loss.detach() * 0
        before = [p.detach().clone() for p in head.parameters()] if step == 1 or step % 20 == 0 else None
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        reader_grad = sum(
            float(p.grad.detach().abs().sum())
            for name, p in head.named_parameters()
            if p.grad is not None and name.startswith(
                ("query_projection", "key_projection", "value_projection", "label_embedding", "context_projection")
            )
        )
        opt.step()
        counts = [int(v["step"]) for v in opt.state.values() if "step" in v]
        if not counts or any(n != step for n in counts):
            raise ValueError("Optimizer update count differs: " + arm)
        if not all(bool(torch.isfinite(p).all()) for p in head.parameters()):
            raise ValueError("Nonfinite updated head: " + arm)
        magnitude = (
            math.sqrt(sum(float((p.detach().double() - old.double()).square().sum())
                          for p, old in zip(head.parameters(), before)))
            if before is not None else None
        )
        rows[arm] = dict(
            original_loss=float(loss.detach()), nll=float(nll),
            lovasz=float(loss.detach() - nll), kl=0.0, lr=lr,
            grad_norm_preclip=float(grad_norm), clip_threshold=1.0,
            reader_grad_abs_sum=reader_grad, parameter_update_l2=magnitude,
            parameter_l2=(
                math.sqrt(sum(float(p.detach().double().square().sum()) for p in head.parameters()))
                if before is not None else None
            ),
            optimizer_step=step,
        )
        current = state.capture_rng()
        if after is not None and state.state_fingerprint(current) != state.state_fingerprint(after):
            raise ValueError("Arms consumed different global RNG streams")
        after = current
        del output, loss
    state.restore_rng(after)
    return rows


def save_checkpoint(folder, heads, optimizers, state, runtime, step, config, study_plan, factory):
    import torch
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("checkpoint_%06d.pt" % step)
    if path.exists():
        raise FileExistsError(path)
    payload = dict(
        format=FORMAT, step=step, heads={a: m.state_dict() for a, m in heads.items()},
        optimizers={a: o.state_dict() for a, o in optimizers.items()}, rng=state.capture_rng(),
        protocol_info=runtime.protocol_info, runtime_config=config,
        study_plan=study_plan, support_provenance=factory.provenance(),
        input_file_sha256={str(p): h for p, h in runtime.input_ledger.items()},
        test_accessed=False,
    )
    temp = path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    reopened = torch.load(temp, map_location="cpu", weights_only=False)
    hashes = {}
    for a in heads:
        h = state.state_fingerprint(heads[a].state_dict())
        if h != state.state_fingerprint(reopened["heads"][a]):
            raise ValueError("Checkpoint head roundtrip differs")
        if state.state_fingerprint(optimizers[a].state_dict()) != state.state_fingerprint(reopened["optimizers"][a]):
            raise ValueError("Checkpoint optimizer roundtrip differs")
        hashes[a] = h
    temp.replace(path)
    record = dict(step=step, sha256=digest(path), head_hashes=hashes, roundtrip_verified=True)
    write_json(path.with_suffix(".pt.json"), record)
    return record


def train_worker(args, study_plan):
    source = study_plan["source"]
    bindings = load_runtime(source)
    torch, original_train, legacy = bindings.torch, bindings.original_train, bindings.legacy
    if not torch.cuda.is_available():
        raise RuntimeError("Real BRIGHT training requires CUDA. CPU contracts are not training evidence.")
    # Load addons privately after the upstream bare-import namespaces are established.
    support_module = named(HERE / "supports.py", "supports")
    evaluation = named(HERE / "evaluation.py", "evaluation")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    smoke = args.mode == "smoke"
    seed = int(args.seed)
    endpoint = args.smoke_steps if smoke else study_plan["endpoint_step"]
    arms = tuple(study_plan["arms"])
    status = dict(
        format=FORMAT, complete=False, execution_complete=False,
        training_complete=False, full_validation_evidence=False,
        final_integrity_pending=True, phase="loading", smoke=smoke,
        test_accessed=False, seed=seed, arms=arms, endpoint_step=endpoint,
        primary_endpoint=study_plan["endpoint_step"], schedule_horizon=5000,
        updates_per_arm={a: 0 for a in arms}, evaluations={},
        trainable="full joint module, including slot encoders and Reader",
        source_run=source["source_run"], pid=os.getpid(),
    )
    write_json(output / "summary.json", status)
    results = []
    started = time.monotonic()
    runtime_args = runtime_arguments(source, seed=seed, output=output)
    try:
        runtime = bindings.Runtime(runtime_args, output)
        with runtime:
            original_train._formal_runtime(runtime, False)
            original_train._names(runtime, False)
            original_train._checked_verify(runtime)
            factory = support_module.SupportFactory(runtime, seed=seed)
            original_train._seed(seed)
            heads, optimizers, initial_hash = model_copies(
                torch, original_train, legacy.state, arms, vars(runtime_args), runtime.device
            )
            train_names, val_names = list(runtime.train_names), list(runtime.val_names)
            if smoke:
                train_names, val_names = runtime.smoke_names()
            # A declared deterministic sample of the full Train list, evaluated
            # with inference transforms. It is never used as the training loader.
            import random
            fixed_names = sorted(random.Random(seed + 191).sample(
                list(runtime.train_names), min(args.fixed_train, len(runtime.train_names))
            ))
            if smoke:
                fixed_names = list(train_names)
            parameter_count = {a: sum(p.numel() for p in h.parameters()) for a, h in heads.items()}
            write_json(output / "manifest.json", dict(
                format=FORMAT, plan=study_plan, seed=seed,
                runtime_config=vars(runtime_args), protocol_info=runtime.protocol_info,
                train_names=list(runtime.train_names), val_names=list(runtime.val_names),
                fixed_train_names=fixed_names, initial_head_sha256=initial_hash,
                parameter_count=parameter_count, arm_specs={
                    a: support_module.ARM_SPECS[a] for a in arms
                },
                fresh_initialization=True, no_common_trained_D_prefix=True,
                historical_trajectory_claim=False, test_accessed=False,
            ))
            write_json(output / "support_provenance.json", factory.provenance())

            def assess(step, split, names, subdir, primary=False):
                status["phase"] = f"evaluate_{split}_{step}"
                write_json(output / "summary.json", status)
                report = evaluation.evaluate(
                    runtime, heads, names, output / subdir, step=step, split=split,
                    seed=seed, original_train=original_train, legacy=legacy,
                    factory=factory, primary_endpoint=primary, smoke=smoke,
                )
                results.append((output / subdir / "result.json", report))
                status["evaluations"][subdir] = dict(
                    step=step, split=split, image_count=report["image_count"],
                    metrics=report["metrics"],
                )
                write_json(output / "summary.json", status)
                evaluation.print_table(report)

            assess(0, "val", val_names, "initial_val")
            assess(0, "fixed_train", fixed_names, "fixed_train_initial")
            iterator = training_batches(runtime, train_names, seed)
            allowed_train = set(train_names)
            checkpoints = [endpoint] if smoke else schedule_steps(endpoint)
            status["phase"] = "training"
            write_json(output / "summary.json", status)
            with (output / "train.jsonl").open("x", encoding="utf-8") as journal:
                for step in range(1, endpoint + 1):
                    epoch, batch = next(iterator)
                    example = runtime.prepare(batch)
                    original_train._assert_example(example)
                    if example.image not in allowed_train:
                        raise ValueError("Training loader returned an image outside declared Train: " + str(example.image))
                    rows = update_arms(
                        heads, optimizers, example, runtime, factory, legacy.state,
                        step=step, multiplier=study_plan["post500_lr_multiplier"],
                    )
                    record = dict(step=step, epoch=epoch, epoch_seed=seed + epoch,
                                  image=example.image, event=example.event, arms=rows)
                    journal.write(json.dumps(record, allow_nan=False) + "\n")
                    journal.flush()
                    status["updates_per_arm"] = {a: step for a in arms}
                    if step == 1 or step % 20 == 0 or step == endpoint:
                        print("ABLATION TRAIN", json.dumps(record, allow_nan=False), flush=True)
                        write_json(output / "summary.json", status)
                    if step in checkpoints:
                        verify_ledgers(source)
                        checkpoint = save_checkpoint(
                            output / "checkpoints", heads, optimizers, legacy.state, runtime,
                            step, vars(runtime_args), study_plan, factory,
                        )
                        status["last_checkpoint"] = checkpoint
                        assess(
                            step, "val", val_names, "val_step_%06d" % step,
                            primary=step == endpoint and not smoke,
                        )
                        status["phase"] = "training"
                        write_json(output / "summary.json", status)
            status["training_complete"] = True
            assess(endpoint, "fixed_train", fixed_names, "fixed_train_endpoint")
            status["phase"] = "final_integrity"
            write_json(output / "summary.json", status)
            checks = original_train._checked_verify(runtime)
            verify_ledgers(source)
            if addon_release() != study_plan["addon_release_sha256"]:
                raise ValueError("Ablation addon changed during execution")
            bindings.legacy_bridge.verify_dependencies(legacy)
            write_json(output / "input_fingerprints.json", {str(p): h for p, h in runtime.input_ledger.items()})
            write_json(output / "support_provenance.json", factory.provenance())
            status["integrity_checks"] = checks
        guard = getattr(runtime, "guard_report", {})
        if guard.get("failures") or guard.get("restore_conflicts") or guard.get("restored") is not True:
            raise RuntimeError("PoolGuard close reported a failure")
        status["pool_guard"] = guard
        # Promote reports only after Runtime exit and all source/model checks.
        for path, result in results:
            result.update(
                complete=True, final_integrity_pending=False,
                phase="complete_smoke" if smoke else "complete",
                primary_endpoint=bool(result.get("primary_endpoint_requested")) and not smoke,
                full_validation_evidence=(
                    not smoke and result["split"] == "val"
                    and result.get("whole_val450_processed") is True
                ),
            )
            write_json(path, result)
            path.with_name("result.json.sha256").write_text(digest(path) + "  result.json\n")
        status.update(
            complete=True, execution_complete=True, training_complete=True,
            full_validation_evidence=not smoke, final_integrity_pending=False,
            phase="complete_smoke" if smoke else "complete", error=None,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(output / "summary.json", status)
        print("ABLATION COMPLETE; smoke=%s; no Test; %s" % (smoke, output), flush=True)
    except BaseException as exc:
        status.update(
            complete=False, execution_complete=False, full_validation_evidence=False,
            phase="failed", error=type(exc).__name__ + ": " + str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(output / "summary.json", status)
        (output / "traceback.txt").write_text(traceback.format_exc())
        raise


def report(root):
    root = Path(root).resolve()
    workflow = root / "workflow_state.json"
    print("结果目录:", root)
    if workflow.is_file():
        w = read_json(workflow)
        print(json.dumps({k: w.get(k) for k in ("complete", "phase", "current_stage", "error")}, ensure_ascii=False, indent=2))
    for p in sorted(root.glob("seed_*/train/summary.json")):
        d = read_json(p)
        print(p.parent.parent.name, "完成:", d.get("complete"),
              "Val证据:", d.get("full_validation_evidence"), "更新:", d.get("updates_per_arm"),
              "错误:", d.get("error"))
        for result in sorted(p.parent.glob("val_step_*/result.json")):
            r = read_json(result)
            print("step", r.get("step"), "正式终点", r.get("primary_endpoint"), {
                a: None if m.get("foreground_miou") is None else round(100*m["foreground_miou"], 4)
                for a, m in r.get("metrics", {}).items()
            })
            print("前景mIoU差值/百分点:", r.get("comparisons_pp"))


def pipeline(args):
    study_plan = plan(args)
    output_base = Path(args.output_base).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    lock_path = output_base / "active.lock"
    # Acquire BEFORE creating run/latest pointers. Kernel releases the lock on
    # exit; never delete lock files to bypass a live process.
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An ablation pipeline holds active.lock; no duplicate was started") from exc
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = output_base / ("run_%s_%s" % (stamp, os.getpid()))
        root.mkdir()
        (root / "logs").mkdir()
        write_json(root / "plan.json", study_plan)
        (output_base / "latest.txt").write_text(str(root) + "\n")
        w = dict(format=FORMAT, complete=False, phase="running", current_stage=None, stages={}, test_accessed=False)
        write_json(root / "workflow_state.json", w)
        print("结果目录:", root, flush=True)
        print("正式终点:", args.steps, "调度长度: 5000; 各组从0步独立训练; 不访问Test", flush=True)
        try:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", BRIGHT_ABLATION_REFERENCE=study_plan["source"]["package"])
            tasks = [("cpu_contracts", [sys.executable, "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_*.py", "-q"])]
            for seed in study_plan["seeds"]:
                for mode in (("smoke", "train") if args.command == "full" else ("smoke",)):
                    (root / ("seed_%s" % seed)).mkdir(exist_ok=True)
                    destination = root / ("seed_%s" % seed) / mode
                    tasks.append((
                        "%s_%s" % (seed, mode),
                        [sys.executable, "-u", str(HERE / "ablation_runner.py"), "worker",
                         "--plan", str(root / "plan.json"), "--seed", str(seed), "--mode", mode,
                         "--output", str(destination), "--fixed-train", str(args.fixed_train),
                         "--smoke-steps", str(args.smoke_steps)],
                    ))
            for stage, command in tasks:
                log = root / "logs" / (stage + ".log")
                w["current_stage"] = stage
                w["stages"][stage] = dict(status="running", command=command, log=str(log))
                write_json(root / "workflow_state.json", w)
                print("START", stage, "log", log, flush=True)
                with log.open("x") as stream:
                    process = subprocess.Popen(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT, env=env)
                    w["stages"][stage]["pid"] = process.pid
                    write_json(root / "workflow_state.json", w)
                    code = process.wait()
                w["stages"][stage].update(returncode=code, status="complete" if code == 0 else "failed", log_sha256=digest(log))
                if code:
                    raise RuntimeError("%s failed (%s); see %s" % (stage, code, log))
                if stage != "cpu_contracts":
                    finished = read_json(Path(command[command.index("--output") + 1]) / "summary.json")
                    if not finished.get("complete") or (stage.endswith("_train") and not finished.get("full_validation_evidence")):
                        raise ValueError("Child exited without complete evidence: " + stage)
                write_json(root / "workflow_state.json", w)
            w.update(complete=True, phase="complete", current_stage=None, error=None)
            write_json(root / "workflow_state.json", w)
            print("PIPELINE COMPLETE; no Test:", root, flush=True)
        except BaseException as exc:
            w.update(complete=False, phase="failed", error=type(exc).__name__ + ": " + str(exc))
            write_json(root / "workflow_state.json", w)
            (root / "traceback.txt").write_text(traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "full", "smoke", "worker", "report"))
    parser.add_argument("--source-run", default=DEFAULT_SOURCE)
    parser.add_argument("--package")
    parser.add_argument("--group", choices=tuple(GROUPS), default="core")
    parser.add_argument("--arms", help="Explicit comma-separated arm IDs; overrides group")
    parser.add_argument("--seeds", default="3407")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--post500-lr", type=float, default=0.2)
    parser.add_argument("--fixed-train", type=int, default=450)
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT)
    parser.add_argument("--root")
    parser.add_argument("--plan")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("smoke", "train"))
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "report":
        if not args.root:
            args.root = (Path(args.output_base) / "latest.txt").read_text().strip()
        return report(args.root)
    if args.command == "plan":
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
        return
    if args.command == "worker":
        if not all((args.plan, args.mode, args.output)) or args.seed is None:
            parser.error("worker needs --plan --mode --output --seed")
        study_plan = read_json(args.plan)
        if addon_release() != study_plan["addon_release_sha256"]:
            raise ValueError("Addon differs from the pipeline plan")
        return train_worker(args, study_plan)
    return pipeline(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOPPED:", type(exc).__name__ + ":", str(exc), file=sys.stderr, flush=True)
        raise
