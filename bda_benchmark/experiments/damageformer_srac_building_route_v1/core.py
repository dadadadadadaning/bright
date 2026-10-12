"""Provenance and matched-head contracts for the isolated building-route test."""
from __future__ import annotations

import hashlib
from pathlib import Path


FORMAT = "SRAC_BUILDING_ROUTE_V1"
ARMS = ("unsplit", "capacity_control", "hard", "soft")
FORMAL_STEPS = 2000
FORMAL_BUDGETS = (2000, 5000)
SMOKE_STEPS = 2


def source_hash() -> str:
    root = Path(__file__).parent
    experiments = root.parent
    h = hashlib.sha256()
    for path in sorted(root.iterdir()):
        if path.is_file() and (path.suffix == ".py" or path.name == "run.sh") and not path.name.startswith("test_"):
            h.update(path.name.encode("utf-8"))
            h.update(path.read_bytes())
    # Direct helper imports outside the immutable E0 implementation tree.
    for package, name in (
        ("damageformer_srac_q_factorial_v1", "evaluate_q.py"),
        ("damageformer_srac_q_factorial_v1", "q_variants.py"),
        ("damageformer_srac_region_head_adapt_v1", "core.py"),
        ("damageformer_srac_region_head_adapt_v1", "native.py"),
        ("damageformer_srac_region_head_adapt_v1", "train.py"),
        ("damageformer_srac_stage_diagnostics_v1", "shared.py"),
    ):
        path = experiments / package / name
        h.update((package + "/" + name).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def validate_budget(steps: int, smoke: bool) -> int:
    allowed = (SMOKE_STEPS,) if smoke else FORMAL_BUDGETS
    if type(steps) is not int or steps not in allowed:
        raise ValueError(f"Matched {'smoke' if smoke else 'formal'} route experiment requires one of {allowed} steps")
    return steps


def load_route_source(bundle, q_checkpoint=None, smoke=False):
    """Accept only E0 Q, or a completed immutable Q-factorial final checkpoint."""
    import torch
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest

    if not q_checkpoint:
        return bundle["front"], dict(kind="e0_original_q", checkpoint=None, sha256=None)
    path = Path(q_checkpoint).resolve()
    if path.name != "last.pth" or not path.is_file():
        raise ValueError("Q source must be an existing completed last.pth")
    from bda_benchmark.experiments.damageformer_srac_q_factorial_v1.q_variants import load_q_checkpoint
    from bda_benchmark.experiments.damageformer_srac_q_factorial_v1.evaluate_q import _checked_checkpoint, _same_base
    model, payload = load_q_checkpoint(path, bundle["device"],
                                       a1_override=bundle["args"].a1_checkpoint,
                                       allow_smoke=smoke)
    _checked_checkpoint(payload, bundle, digest(path), smoke)
    if payload.get("smoke") is not False and not smoke:
        raise ValueError("Smoke Q checkpoint cannot enter formal route training")
    if payload.get("arm") != "building_reconstruction":
        raise ValueError("Stage-4 Q comparison accepts only completed building_reconstruction Q")
    for name in ("base", "adapter", "embedding", "region_classifier"):
        if not _same_base(getattr(bundle["front"], name), getattr(model, name)):
            raise ValueError("Q checkpoint changed an A1 or downstream frozen component: " + name)
    model.requires_grad_(False).eval()
    bundle["input_paths"]["q_checkpoint"] = path
    bundle["inputs_before"]["q_checkpoint"] = digest(path)
    return model, dict(kind="completed_building_reconstruction_q", checkpoint=str(path),
                       sha256=digest(path), steps=payload["completed_steps"],
                       train_source_hash=payload["source_hash"])


def make_heads(front, device, threshold=0.5, soft_floor=0.0):
    import torch
    from .model import BuildingRouteHead
    heads = torch.nn.ModuleDict({arm: BuildingRouteHead(
        front, mode=arm, threshold=threshold, soft_floor=soft_floor) for arm in ARMS}).to(device)
    for head in heads.values():
        head.front.requires_grad_(False).eval()
    return heads


def head_state(head):
    """Never put the large, frozen front into a trainable-head checkpoint."""
    return dict(embedding=head.embedding.state_dict(),
                region_classifier=head.region_classifier.state_dict())


def load_head_state(head, state):
    if set(state) != {"embedding", "region_classifier"}:
        raise ValueError("Route checkpoint has incomplete head state")
    head.embedding.load_state_dict(state["embedding"], strict=True)
    head.region_classifier.load_state_dict(state["region_classifier"], strict=True)


def trainable_parameters(head):
    return tuple(head.embedding.parameters()) + tuple(head.region_classifier.parameters())


def equal_initial_heads(heads):
    import torch
    states = {arm: head_state(heads[arm]) for arm in ARMS}
    reference = states[ARMS[0]]
    return all(set(states[arm][section]) == set(reference[section]) and all(
        torch.equal(states[arm][section][key], value)
        for key, value in reference[section].items())
        for arm in ARMS[1:] for section in reference)


def assert_independent_heads(heads, front):
    if set(heads) != set(ARMS) or not equal_initial_heads(heads):
        raise ValueError("Route heads must start from identical independent E0 weights")
    seen = {id(p) for p in front.parameters()}
    for arm in ARMS:
        parameters = trainable_parameters(heads[arm])
        ids = {id(p) for p in parameters}
        if not parameters or len(ids) != len(parameters) or ids & seen:
            raise ValueError("Route heads share parameters or overlap the frozen front")
        if any(not p.requires_grad for p in parameters):
            raise ValueError("A route readout is unexpectedly frozen")
        seen.update(ids)
