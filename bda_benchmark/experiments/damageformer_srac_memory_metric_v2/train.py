"""Fit matched Train-only SRAC search projections; no Reader, Q or gate updates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from ..damageformer_srac_hardquery_memory_v1.io_utils import (
    add_source_arguments, artifact_identity, digest, fail_summary,
    load_verified_bundle, new_output, read_json, verify_unchanged, write_json,
)
from ..damageformer_srac_hardquery_memory_v1.metric_learning import (
    HardQuerySampler, soft_cross_event_loss,
)
from ..damageformer_srac_hardquery_memory_v1.train_metric import QueryStore, reliable_support
from ..damageformer_srac_qmetric_v1.metric import RetrievalProjector
from .sampling import RetrievalMisleadSampler, mine_reference_risk

FORMAT = "SRAC_MEMORY_METRIC_V2"
ARMS = ("j1_hard", "mislead")
FAILED_FINALIZATION_SOURCE_SHA256 = (
    "e29550c0ca0b9a49d31f8067c5f28c73d5ccc54240e362453fb804f9a347c0b5"
)


def experiment_source_hash() -> str:
    """Bind a fitted projector to the exact new training implementation."""
    root = Path(__file__).resolve().parent
    paths = sorted(p for p in root.iterdir() if p.suffix == ".py" and not p.name.startswith("test_"))
    paths += [root / "run.sh"]
    value = hashlib.sha256()
    for path in paths:
        value.update(path.name.encode("utf-8"))
        value.update(path.read_bytes())
    return value.hexdigest()


def parser():
    p = add_source_arguments(argparse.ArgumentParser(__doc__))
    p.add_argument("--protocol", required=True, help="Passed legacy/native all-path E0")
    p.add_argument("--queries", required=True, help="Completed Train-only hardquery query cache")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--per_class", type=int, default=16)
    p.add_argument("--purity", type=float, default=.8, help="Reliable support and risk-query purity")
    p.add_argument("--temperature", type=float, default=.1, help="Matched metric-loss temperature")
    p.add_argument("--risk_temperature", type=float, default=.1, help="Frozen Top-8 vote temperature")
    p.add_argument("--risk_margin", type=float, default=0., help="Require vote[D]-vote[I] greater than margin")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hard_fraction", type=float, default=.5)
    p.add_argument("--hard_quantile", type=float, default=.75)
    p.add_argument("--mine_batch", type=int, default=512)
    return p


def _validate_args(args):
    args.steps = (20 if args.smoke else 2000) if args.steps is None else args.steps
    if args.steps != (20 if args.smoke else 2000):
        raise ValueError("Matched budget is exactly 2000 updates (20 for smoke)")
    if args.per_class < 2 or args.mine_batch < 1 or not 0 <= args.purity <= 1:
        raise ValueError("Invalid per_class, mine_batch or purity")
    if not 0 < args.hard_fraction < 1 or not 0 < args.hard_quantile < 1:
        raise ValueError("Invalid hard_fraction or hard_quantile")
    for name in ("temperature", "risk_temperature", "lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    if not math.isfinite(args.risk_margin) or args.risk_margin < 0:
        raise ValueError("risk_margin must be finite and nonnegative")


def _train_arm(args, arm, store, support, sampler, identity, output, coverage, source_sha):
    device = support["tokens"].device
    # Both arms start from the same identity weights; only sampled Train
    # queries differ. The bank, objective, optimizer and Reader stay fixed.
    projector = RetrievalProjector(support["tokens"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=0.)
    generator = torch.Generator().manual_seed(args.seed + 1)
    active_updates = used_anchors = zero_batches = 0
    sampled_by_class = [0] * 4
    for step in range(1, args.steps + 1):
        indices, sampled = sampler.sample(args.per_class, generator, args.hard_fraction)
        query = store.batch(indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss, statistics = soft_cross_event_loss(
            query["tokens"], query["distribution"], query["events"], query["images"],
            support["tokens"], support["distribution"], support["events"], support["images"],
            projector, temperature=args.temperature,
        )
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1., error_if_nonfinite=True)
        factor = .5 * (1 + math.cos(math.pi * (step - 1) / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        optimizer.step()
        active_updates += int(float(grad) > 0)
        used_anchors += statistics["used_anchor_count"]
        zero_batches += int(statistics["zero_loss"])
        counts = torch.bincount(sampler.classes[indices], minlength=4).tolist()
        sampled_by_class = [a + b for a, b in zip(sampled_by_class, counts)]
        if step == 1 or step % 20 == 0 or step == args.steps:
            row = dict(arm=arm, step=step, loss=float(loss.detach()), grad_norm=float(grad),
                       lr=args.lr * factor, sampling=sampled, objective=statistics)
            with (output / f"training_{arm}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
    if not args.smoke and (active_updates == 0 or used_anchors == 0):
        raise RuntimeError(f"{arm}: no cross-event metric learning occurred")
    result = dict(optimizer_updates=args.steps, nonzero_gradient_updates=active_updates,
                  used_anchors=used_anchors, zero_anchor_batches=zero_batches,
                  sampled_by_class=sampled_by_class)
    checkpoint = dict(format=FORMAT, arm=arm, complete=True, smoke=args.smoke,
                      train_only=True, test_accessed=False, dim=projector.dim,
                      steps=args.steps, model=projector.state_dict(),
                      arguments=vars(args), source_experiment_sha256=source_sha,
                      query_manifest_sha256=store.manifest_sha256,
                      support_coverage=coverage, initialization="identity",
                      selection="fixed 2000-update final state; no Val/Test tuning",
                      objective="normalized_soft_overlap_cross_entropy",
                      support_subset="original Train bank, purity >= threshold; full inference bank unchanged",
                      training_summary=result, **identity)
    return checkpoint, result


def run(args):
    _validate_args(args)
    bundle = load_verified_bundle(args)
    identity = artifact_identity(bundle)
    store = QueryStore(args.queries, bundle, args.smoke)
    device = torch.device(args.device)
    support, coverage = reliable_support(bundle["bank_cpu"], args.purity, device)
    if not args.smoke and any(row["events"] < 2 for row in coverage.values()):
        raise ValueError("Formal fitting needs reliable supports from >=2 Train events per class")
    output = new_output(args.output)
    source_sha = experiment_source_hash()
    manifest = dict(format=FORMAT, complete=False, train_only=True, test_accessed=False,
                    smoke=args.smoke, arms=list(ARMS), arguments=vars(args),
                    source_experiment_sha256=source_sha,
                    query_manifest_sha256=store.manifest_sha256,
                    support_coverage=coverage, query_inventory=store.inventory,
                    fixed="front/Q/Train query cache/original bank membership/Reader/J1/J2/J3",
                    learned="two matched 128x128 retrieval projections only",
                    runtime_settings=bundle["runtime_settings"], **identity)
    write_json(output / "manifest.json", manifest)
    try:
        misleading, missed, mining = mine_reference_risk(
            store, bundle["bank"], bundle["metric"], device,
            topk=bundle["topk"], temperature=args.risk_temperature,
            purity=args.purity, risk_margin=args.risk_margin,
            batch_size=args.mine_batch,
        )
        baseline = HardQuerySampler(store.fields["distribution"],
                                    store.fields["base_probability"],
                                    hard_quantile=args.hard_quantile)
        mislead = RetrievalMisleadSampler(
            store.fields["distribution"], store.fields["base_probability"],
            misleading, missed, hard_quantile=args.hard_quantile,
            require_target=not args.smoke,
        )
        manifest.update(mining=mining, query_strata=dict(
            j1_hard=baseline.inventory, mislead=mislead.inventory))
        write_json(output / "manifest.json", manifest)
        payloads, results = {}, {}
        for arm, sampler in (("j1_hard", baseline), ("mislead", mislead)):
            payloads[arm], results[arm] = _train_arm(
                args, arm, store, support, sampler, identity, output, coverage, source_sha)
        checks = verify_unchanged(bundle)
        checks["new_training_code"] = experiment_source_hash() == source_sha
        checks["query_manifest"] = digest(store.path) == store.manifest_sha256
        if not all(checks.values()):
            raise RuntimeError("Immutable source or query cache changed: " + str(checks))
        hashes = {}
        for arm in ARMS:
            payloads[arm]["immutable_checks"] = checks
            path = output / f"metric_{arm}_last.pth"
            torch.save(payloads[arm], path)
            hashes[arm] = digest(path)
        write_json(output / "summary.json", dict(
            format=FORMAT, complete=True, smoke=args.smoke, train_only=True,
            test_accessed=False, full_validation_evidence=False,
            arms=results, steps=args.steps, checkpoint_sha256=hashes,
            query_manifest_sha256=store.manifest_sha256,
            source_experiment_sha256=source_sha,
            immutable_checks=checks, mining=mining, **identity,
        ))
        manifest["complete"] = True
        write_json(output / "manifest.json", manifest)
    except Exception as error:
        fail_summary(output, error)
        raise
    print("MATCHED TRAIN-ONLY METRIC FIT COMPLETE. No Reader/Q/gate fitted.", flush=True)
    return results


def load_metric_checkpoint(path, bundle, expected_arm: str, smoke: bool = False) -> RetrievalProjector:
    """Load one trained search key only after hash/provenance/budget checks."""
    if expected_arm not in ARMS:
        raise ValueError("Unknown metric arm")
    path = Path(path)
    if path.name != f"metric_{expected_arm}_last.pth":
        raise ValueError("Metric checkpoint filename/arm disagree")
    summary = read_json(path.parent / "summary.json")
    manifest = read_json(path.parent / "manifest.json")
    if (summary.get("format") != FORMAT or summary.get("complete") is not True
            or summary.get("smoke") is not bool(smoke)
            or summary.get("train_only") is not True
            or summary.get("test_accessed") is not False
            or manifest.get("format") != FORMAT or manifest.get("complete") is not True
            or manifest.get("smoke") is not bool(smoke)
            or manifest.get("train_only") is not True
            or manifest.get("test_accessed") is not False):
        raise ValueError("Incomplete, mismatched or changed metric run")
    checkpoint_hashes = summary.get("checkpoint_sha256", {})
    if set(checkpoint_hashes) != set(ARMS) or any(
        checkpoint_hashes[arm] != digest(path.parent / f"metric_{arm}_last.pth")
        for arm in ARMS
    ):
        raise ValueError("Incomplete, mismatched or changed metric run")
    checks = summary.get("immutable_checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("Metric immutable checks did not all pass")
    expected_steps = 20 if smoke else 2000
    if summary.get("steps") != expected_steps or summary.get("protocol_sha256") != bundle["protocol_sha256"]:
        raise ValueError("Metric budget or E0 protocol mismatch")
    identity = artifact_identity(bundle)
    for key, value in identity.items():
        if summary.get(key) != value or manifest.get(key) != value:
            raise ValueError("Metric source provenance mismatch: " + key)
    recorded_source = summary.get("source_experiment_sha256")
    failed_summary_sha = summary.get("recovery_failed_summary_sha256")
    recovered_prior_release = (
        recorded_source == FAILED_FINALIZATION_SOURCE_SHA256
        and summary.get("recovered_from_failed_finalization") is True
        and manifest.get("recovered_from_failed_finalization") is True
        and isinstance(failed_summary_sha, str)
        and len(failed_summary_sha) == 64
        and manifest.get("recovery_failed_summary_sha256") == failed_summary_sha
        and digest(path.parent / "failed_summary_original.json") == failed_summary_sha
    )
    if (not (recorded_source == experiment_source_hash() or recovered_prior_release)
            or manifest.get("source_experiment_sha256") != recorded_source
            or manifest.get("query_manifest_sha256") != summary.get("query_manifest_sha256")):
        raise ValueError("Metric V2 training source changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in dict(format=FORMAT, arm=expected_arm, complete=True,
                           smoke=smoke, train_only=True, test_accessed=False,
                           steps=expected_steps,
                           source_experiment_sha256=summary["source_experiment_sha256"],
                           query_manifest_sha256=summary["query_manifest_sha256"],
                           **identity).items():
        if payload.get(key) != value:
            raise ValueError("Metric checkpoint provenance mismatch: " + key)
    if payload.get("immutable_checks") != checks:
        raise ValueError("Metric checkpoint immutable checks mismatch")
    if not isinstance(payload.get("dim"), int) or payload["dim"] != bundle["bank_cpu"]["tokens"].shape[-1]:
        raise ValueError("Metric dimension differs from original bank")
    with torch.random.fork_rng(devices=[]):
        projector = RetrievalProjector(payload["dim"]).to(bundle["device"])
    projector.load_state_dict(payload["model"], strict=True)
    return projector.eval().requires_grad_(False)


if __name__ == "__main__":
    run(parser().parse_args())
