"""Synthetic CPU/CUDA compatibility checks against an existing original package.

No data, checkpoints, training jobs, global monkeypatches or original file writes.
This tests synthetic optimizer updates, NOT full-data CUDA runtime reliability.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


def load_original(directory):
    directory = Path(directory).resolve()
    for name in ("model.py", "slots.py"):
        if not (directory / name).is_file():
            raise FileNotFoundError(str(directory / name))
    tag = "_bright_prior_reference_" + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    module = types.ModuleType(tag)
    module.__path__ = [str(directory)]
    sys.modules[tag] = module
    spec = importlib.util.spec_from_file_location(tag + ".model", directory / "model.py")
    model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)
    slots = sys.modules[tag + ".slots"]
    return model.JointRegionMemoryClassifier, slots.prepare_slots


def fixture(torch, prepare, device):
    generator = torch.Generator(device=device).manual_seed(19)
    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device)
    height, width, regions, members = 2, 4, 3, 5
    pixel = randn(1, 256, height, width)
    q = randn(1, 9, height * width).softmax(1)
    ids = torch.randint(regions, (9, height * width), generator=generator, device=device)
    building = randn(1, 1, height, width).sigmoid() * .8 + .1
    slots = prepare(pixel, q, ids, regions, (height, width), building, "soft")
    fraction = randn(members, 2).softmax(-1)
    return dict(pixel_features=pixel, region_f=randn(1, regions, 256), region_t=randn(1, regions, 128),
                q=q, ids=ids, base_probability_full=randn(1, 4, 4, 8).softmax(1),
                **{k: slots[k] for k in ("slot_f", "slot_fraction", "slot_valid", "route_q", "route_ids")},
                support_ids=torch.tensor([[0, 2, 4], [1, 2, -1], [3, 4, 0]], device=device),
                support_f=randn(members, 2, 256), support_t=randn(members, 128),
                support_labels=randn(members, 2, 4).softmax(-1), support_fraction=fraction,
                support_valid=torch.ones(members, 2, dtype=torch.bool, device=device))


def run_checks(original_dir, device="cpu"):
    import random
    import torch
    from reader_prior import classifier_with_prior, shuffled_valid_prior
    torch.set_num_threads(1)
    torch.manual_seed(781)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    Base, prepare = load_original(original_dir)
    Prior = classifier_with_prior(Base)
    inputs = fixture(torch, prepare, device)
    snapshots = {k: v.clone() for k, v in inputs.items()}
    base = Base(hidden=16, heads=4, chunk_size=2).to(device)
    # A zero residual head masks Reader differences, so activate it for meaningful tests.
    with torch.no_grad():
        base.joint_head[-1].weight.normal_(0, .1)
    prior = Prior(hidden=16, heads=4, chunk_size=2, prior_beta=0).to(device)
    prior.load_state_dict(base.state_dict(), strict=True)
    assert list(base.state_dict()) == list(prior.state_dict())
    opt0 = torch.optim.AdamW(base.parameters(), lr=1e-4, weight_decay=.005)
    opt1 = torch.optim.AdamW(prior.parameters(), lr=1e-4, weight_decay=.005)
    for _ in range(2):
        opt0.zero_grad(set_to_none=True); opt1.zero_grad(set_to_none=True)
        a, b = base(**inputs), prior(**inputs)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        a["logits"].square().mean().backward()
        b["logits"].square().mean().backward()
        for x, y in zip(base.parameters(), prior.parameters()):
            assert (x.grad is None) == (y.grad is None)
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
        opt0.step(); opt1.step()
        for x, y in zip(base.parameters(), prior.parameters()):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    checks = ["beta0_exact_outputs_gradients_two_AdamW_updates", "original_parameter_ABI"]
    prior.prior_beta = 2
    original_context = base(**inputs)["memory_context"]
    assert not torch.equal(original_context, prior(**inputs)["memory_context"])
    checks.append("nonzero_prior_changes_Reader_with_activated_head")
    for mode in ("cosine", "shuffled"):
        prior.prior_mode = mode
        before_rng = torch.random.get_rng_state().clone()
        before_py_rng = random.getstate()
        before_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
        for chunk in (1, 100):
            prior.zero_grad(set_to_none=True)
            result = prior(**inputs, chunk_size=chunk, prior_key="synthetic_image")
            result["logits"].square().mean().backward()
            grads = {n: p.grad.clone() for n, p in prior.named_parameters() if p.grad is not None}
            if chunk == 1:
                first, first_grads = result, grads
            else:
                torch.testing.assert_close(first["logits"], result["logits"], atol=2e-6, rtol=1e-5)
                for n in grads:
                    torch.testing.assert_close(first_grads[n], grads[n], atol=3e-6, rtol=1e-4)
        assert torch.equal(before_rng, torch.random.get_rng_state()) and before_py_rng == random.getstate()
        if before_cuda_rng is not None:
            assert torch.equal(before_cuda_rng, torch.cuda.get_rng_state(device))
        checks.append(mode + "_chunk_output_gradient_equivalence_and_no_RNG_consumption")
    # Invalid slots may contain NaNs. They must never affect a valid prior or output.
    bad = copy.deepcopy(inputs)
    bad["support_ids"].fill_(-1)
    bad["support_valid"].zero_()
    for k in ("support_f", "support_t", "support_labels", "support_fraction"):
        bad[k].fill_(float("nan"))
    prior.zero_grad(set_to_none=True)
    out = prior(**bad, prior_key="empty_candidates")
    assert out["memory_context"].abs().sum().item() == 0
    assert bool(torch.isfinite(out["probabilities"]).all())
    out["logits"].square().mean().backward()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in prior.parameters())
    checks.append("all_invalid_support_zero_context_finite_backward")
    # Support masks with missing slots and NaN padding, mixed with valid candidates.
    bad = copy.deepcopy(inputs)
    bad["support_valid"][2, 1] = False
    for k in ("support_f", "support_labels", "support_fraction"):
        bad[k][2, 1] = float("nan")
    out = prior(**bad, prior_key="partial_mask")
    assert bool(torch.isfinite(out["probabilities"]).all())
    checks.append("partially_invalid_NaN_support_is_masked")
    for k in inputs:
        torch.testing.assert_close(inputs[k], snapshots[k], atol=0, rtol=0)
    try:
        prior(**inputs)  # shuffled requires a stable image key
    except ValueError:
        pass
    else:
        raise AssertionError("Missing C2 key accepted")
    assert prior._raw_prior_inputs is None
    checks.append("inputs_unmodified_and_context_cleared")
    values = torch.tensor([[[1., 2., 999., 3.], [4., 5., 999., 6.]]], device=device)
    mask = torch.tensor([[True, True, False, True]], device=device)
    shuffled = shuffled_valid_prior(values, mask, image_key="x", start_region=0, seed=3407)
    for role in range(2):
        assert sorted(values[0, role, mask[0]].tolist()) == sorted(shuffled[0, role, mask[0]].tolist())
        assert shuffled[0, role, 2].item() == 999
    checks.append("shuffled_control_preserves_valid_score_multiset_and_padding")
    # Independent analytic oracle: two candidates with cosine +1/-1, zero
    # learned logits and values +1/-1 must yield tanh(beta), in every head.
    analytic = Prior(hidden=16, heads=4, prior_beta=2).to(device)
    with torch.no_grad():
        analytic.query_projection.weight.zero_()
        analytic.query_projection.bias.zero_()
        analytic.context_projection.weight.copy_(torch.eye(16, device=device))
    raw_q = torch.zeros(1, 1, 2, 256, device=device)
    raw_q[..., 0] = 1
    raw_s = torch.zeros(1, 2, 256, device=device)
    raw_s[0, 0, 0], raw_s[0, 1, 0] = 1, -1
    analytic._raw_prior_inputs = (raw_q, raw_s, "analytic")
    values = torch.stack((torch.ones(16, device=device), -torch.ones(16, device=device)))[None]
    context = analytic._read(torch.zeros(1, 2, 16, device=device), torch.zeros(1, 1, dtype=torch.long, device=device),
                             torch.zeros(1, 2, 16, device=device), values,
                             torch.ones(1, 2, dtype=torch.bool, device=device),
                             torch.ones(1, 2, dtype=torch.bool, device=device), 1)
    torch.testing.assert_close(context, torch.full_like(context, torch.tensor(2.).tanh().item()), atol=1e-6, rtol=1e-6)
    analytic._raw_prior_inputs = None
    checks.append("analytic_two_candidate_prior_direction_and_scale")
    return dict(status="synthetic_checks_passed", device=device, torch=torch.__version__, checks=checks,
                prior_config=prior.prior_config(),
                model_file_sha256=hashlib.sha256((Path(original_dir) / "model.py").read_bytes()).hexdigest(),
                real_data_forward_tested=False, full_training_tested=False,
                note="Synthetic checks do not resolve historical route/NLL failures or certify a training run.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--original-dir")
    source.add_argument("--run", help="Discover original code from run/train/manifest.json arguments.tool_dir")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    directory = args.original_dir
    if args.run:
        manifest = Path(args.run) / "train/manifest.json"
        d = json.loads(manifest.read_text(encoding="utf-8"))
        directory = d["arguments"]["tool_dir"]
    print(json.dumps(run_checks(directory, args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
