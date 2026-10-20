"""CPU checks of support ablations against the original classifier and pooling."""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import unittest
import torch

from supports import ARM_SPECS, SupportFactory


def _original():
    candidates = [
        Path(os.environ.get("BRIGHT_ABLATION_REFERENCE", "")),
        Path(__file__).resolve().parent.parent / "joint_anchor_fresh_chain_v2",
    ]
    for root in candidates:
        original = root / "deps" / "original"
        if (original / "model.py").is_file():
            name = "_ablation_supports_original"
            if name not in sys.modules:
                package = type(sys)(name)
                package.__path__ = [str(original)]
                sys.modules[name] = package
            modules = []
            for stem in ("slots", "model"):
                fullname = name + "." + stem
                if fullname not in sys.modules:
                    spec = importlib.util.spec_from_file_location(fullname, original / (stem + ".py"))
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[fullname] = module
                    spec.loader.exec_module(module)
                modules.append(sys.modules[fullname])
            return modules
    raise RuntimeError("Set BRIGHT_ABLATION_REFERENCE to joint_anchor_fresh_chain_v2 for CPU tests")


slots, original_model = _original()


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return type(value)((key, _clone_tree(item)) for key, item in value.items())
    return copy.deepcopy(value)


def _assert_tree_equal(actual, expected):
    assert type(actual) is type(expected)
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_tree_equal(actual[key], expected[key])
    else:
        assert actual == expected


def _direct_parent_pool(value, q, ids, regions):
    """Deliberately simple independent all-pixel/all-candidate weighted pool."""
    channels = value.shape[1]
    sums = torch.zeros(regions, channels, dtype=torch.float64)
    mass = torch.zeros(regions, dtype=torch.float64)
    for candidate in range(q.shape[1]):
        for pixel in range(q.shape[-1]):
            region = int(ids[0, candidate, pixel])
            weight = float(q[0, candidate, pixel])
            mass[region] += weight
            sums[region] += weight * value.flatten(2)[0, :, pixel].double()
    return (sums / torch.where(mass > 0, mass, 1)[:, None]).float(), mass


def make_fixture():
    generator = torch.Generator().manual_seed(317)
    regions, height, width, members, topk = 4, 4, 5, 10, 8
    feature = torch.randn(1, 256, height, width, generator=generator)
    q = torch.rand(1, 9, height * width, generator=generator)
    q = q / q.sum(1, keepdim=True)
    ids = torch.randint(0, 3, q.shape, generator=generator)
    building = torch.rand(1, 1, height, width, generator=generator)
    probability = torch.randn(1, 4, height, width, generator=generator).softmax(1)
    region_f, _ = _direct_parent_pool(feature, q, ids, regions)
    region_t = torch.randn(regions, 128, generator=generator)
    local_ids = torch.tensor([[0, 1, 2, -1, -1, -1, -1, -1]] * 3 + [[-1] * topk])
    memory_ids = torch.tensor([[3, 4, 5, 6, 9, -1, -1, -1]] * 3 + [[-1] * topk])
    masses = 1 + 3 * torch.rand(members, 2, generator=generator)
    masses[0] = torch.tensor([9., 1.])
    masses[5:7, 1] = 0
    masses[8] = 0
    label_mass = masses * torch.rand(members, 2, generator=generator)
    label_mass[0] = torch.tensor([0., 1.])
    valid = (masses > 0) & (label_mass > 0)
    labels = torch.randn(members, 2, 4, generator=generator).softmax(-1)
    labels = torch.where(valid[..., None], labels, 0)
    train_f = torch.randn(members, 2, 256, generator=generator)
    train_f = torch.where(masses[..., None] > 0, train_f, 0)
    fraction = masses / torch.where(masses.sum(-1, keepdim=True) > 0, masses.sum(-1, keepdim=True), 1)
    soft_table = dict(slot_f=train_f, slot_mass=masses, slot_fraction=fraction,
                      slot_labels=labels, label_mass=label_mass, slot_valid=valid)
    bank = dict(
        t128=torch.randn(members, 128, generator=generator),
        eligible=torch.tensor([True] * 7 + [False, False, True]),
        images=torch.arange(members), events=torch.tensor([0, 0, 0, 1, 2, 2, 3, 3, 4, 1]),
    )
    runtime = SimpleNamespace(bank=bank, retrieval_bank=dict(bank),
                              values=dict(rules=dict(soft=soft_table)))
    supports = {}
    for rule, local_arm, memory_arm in (("capacity_control", "a", "c"), ("soft", "b", "d")):
        slot = slots.prepare_slots(feature, q, ids, regions, (height, width), building, rule)
        local_labels, _ = slots.slot_pool(probability, slot["route_q"], slot["route_ids"], 2 * regions)
        query = {key: slot[key] for key in ("slot_f", "slot_fraction", "slot_valid", "route_q", "route_ids")}
        supports[local_arm] = dict(
            **query, support_ids=local_ids, support_f=slot["slot_f"][0], support_t=region_t,
            support_labels=local_labels[0], support_fraction=slot["slot_fraction"][0],
            support_valid=slot["slot_valid"][0],
        )
        supports[memory_arm] = dict(
            **query, support_ids=memory_ids, support_f=train_f, support_t=bank["t128"],
            support_labels=labels, support_fraction=fraction, support_valid=valid,
        )
    example = SimpleNamespace(
        image="event0/query.png", event="event0", target=torch.zeros(1, height, width, dtype=torch.long),
        common=dict(pixel_features=feature, region_f=region_f[None], region_t=region_t[None],
                    q=q, ids=ids, base_probability_full=probability),
        supports=supports, diagnostics=dict(event_id=0, image_id=2),
    )
    return runtime, example


def _model():
    with torch.random.fork_rng():
        torch.manual_seed(25)
        model = original_model.JointRegionMemoryClassifier(hidden=16, heads=4, chunk_size=2)
        # The shipped head is zero initialized; make output changes observable.
        torch.nn.init.normal_(model.joint_head[-1].weight, std=0.02)
    return model


class SupportTransformationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = make_fixture()

    def test_single_query_and_local_equal_direct_q_pool_batched_ids(self):
        self._check_single_query_and_local_equal_direct_original_q_pool(False)

    def test_single_query_and_local_equal_direct_q_pool_unbatched_ids(self):
        self._check_single_query_and_local_equal_direct_original_q_pool(True)

    def _check_single_query_and_local_equal_direct_original_q_pool(self, unbatched_ids):
        runtime, example = self.fixture
        if unbatched_ids:
            example.common["ids"] = example.common["ids"][0]
        prepared = SupportFactory(runtime, 91).prepare(example, ("e", "f"))
        ids = example.common["ids"]
        if ids.ndim == 2:
            ids = ids[None]
        expected_f, mass = _direct_parent_pool(example.common["pixel_features"], example.common["q"], ids, 4)
        expected_labels, _ = _direct_parent_pool(example.common["base_probability_full"], example.common["q"], ids, 4)
        for arm, support in prepared.items():
            torch.testing.assert_close(support["slot_f"][0, :, 0], expected_f, atol=2e-6, rtol=2e-5)
            assert torch.count_nonzero(support["slot_f"][:, :, 1]) == 0
            assert torch.equal(support["slot_valid"][0, :, 0], mass > 0)
            assert not support["slot_valid"][:, :, 1].any()
            assert torch.equal(support["route_q"][:, :9], example.common["q"])
            assert torch.count_nonzero(support["route_q"][:, 9:]) == 0
            assert torch.equal(support["route_ids"][:, :9], 2 * ids)
            assert torch.equal(support["route_ids"][:, 9:], 2 * ids + 1)
        torch.testing.assert_close(prepared["e"]["support_f"][:, 0], expected_f, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(prepared["e"]["support_labels"][:, 0], expected_labels, atol=2e-6, rtol=2e-5)


    def test_single_train_feature_and_label_mass_are_separate_and_cached(self):
        runtime, example = self.fixture
        factory = SupportFactory(runtime, 17)
        table = runtime.values["rules"]["soft"]
        support = factory.prepare(example, ("f",))["f"]
        # A feature-rich slot with no valid GT must still affect the feature pool.
        expected_feature = (9 * table["slot_f"][0, 0] + table["slot_f"][0, 1]) / 10
        torch.testing.assert_close(support["support_f"][0, 0], expected_feature)
        torch.testing.assert_close(support["support_labels"][0, 0], table["slot_labels"][0, 1])
        assert support["support_valid"][0].tolist() == [True, False]
        assert support["support_valid"][8].tolist() == [False, False]
        assert torch.count_nonzero(support["support_f"][:, 1]) == 0
        assert torch.count_nonzero(support["support_labels"][:, 1]) == 0
        assert factory.prepare(example, ("f",))["f"]["support_f"] is support["support_f"]
        assert support["support_t"] is runtime.bank["t128"]


    def test_prepare_never_reads_query_target_or_mutates_inputs(self):
        runtime, example = self.fixture
        class TargetTrap:
            @property
            def target(self):
                raise AssertionError("Query ground truth was accessed")
        trapped = TargetTrap()
        trapped.__dict__.update({key: value for key, value in vars(example).items() if key != "target"})
        before = _clone_tree(dict(common=example.common, supports=example.supports, bank=runtime.bank, values=runtime.values))
        factory = SupportFactory(runtime, 53)
        first = factory.prepare(trapped, tuple(ARM_SPECS))
        example.target.fill_(255)
        second = factory.prepare(example, reversed(tuple(ARM_SPECS)))
        for arm in ARM_SPECS:
            _assert_tree_equal(first[arm], second[arm])
        _assert_tree_equal(dict(common=example.common, supports=example.supports, bank=runtime.bank, values=runtime.values), before)


    def test_random_supports_are_legal_deterministic_padded_and_rng_independent(self):
        runtime, example = self.fixture
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        factory = SupportFactory(runtime, 39)
        ids = factory.prepare(example, ("r",))["r"]["support_ids"]
        assert torch.equal(torch.random.get_rng_state(), torch_state)
        assert random.getstate() == python_state
        expected = {3, 4, 5, 6, 9}
        for row in ids[:3]:
            chosen = row[row >= 0].tolist()
            assert set(chosen) == expected and len(chosen) == len(set(chosen))
            assert (row[5:] == -1).all()
        assert (ids[3] == -1).all()
        other = copy.copy(example)
        other.image = "unrelated/query.png"
        factory.prepare(other, ("l", "r", "n"))
        repeated = factory.prepare(example, ("e", "r", "d"))["r"]["support_ids"]
        assert torch.equal(ids, repeated)
        assert torch.equal(ids, SupportFactory(runtime, 39).prepare(example, ("r",))["r"]["support_ids"])
        assert not torch.equal(ids, SupportFactory(runtime, 40).prepare(example, ("r",))["r"]["support_ids"])
        assert factory.random_choices.keys() == {example.image}
        assert len(factory.random_records) == 2
        assert factory.last_diagnostics[example.image]["r"]["selection_legal"]
        # Parent eligibility can be stricter than positive soft slot mass.
        # Preserve D's per-query budget, including an otherwise nonempty row
        # with no retrieved supports, rather than infer it from slot mass.
        example.supports["d"] = dict(example.supports["d"])
        reference = example.supports["d"]["support_ids"].clone()
        reference[0, 3:] = -1
        reference[1] = -1
        example.supports["d"]["support_ids"] = reference
        assert example.supports["d"]["slot_valid"][0, 1].any()
        matched = factory.prepare(example, ("r",))["r"]["support_ids"]
        assert torch.equal((matched >= 0).sum(-1), (reference >= 0).sum(-1))
        assert (matched[1] == -1).all()


    def test_random_without_any_legal_parent_returns_padding(self):
        runtime, example = self.fixture
        runtime.retrieval_bank["eligible"] = torch.zeros_like(runtime.bank["eligible"])
        factory = SupportFactory(runtime, 45)
        result = factory.prepare(example, ("r",))["r"]
        assert (result["support_ids"] == -1).all()
        assert factory.random_records[example.image]["legal_parent_count"] == 0


    def test_label_permutation_keeps_paired_roles_and_validity_patterns(self):
        runtime, example = self.fixture
        factory = SupportFactory(runtime, 19)
        source = example.supports["d"]
        result = factory.prepare(example, ("l", "d"))["l"]
        mapping = factory.permutation
        valid = source["support_valid"]
        eligible = runtime.retrieval_bank["eligible"]
        events = runtime.retrieval_bank["events"]
        assert torch.equal(valid[mapping], valid)
        assert torch.equal(events[mapping], events)
        assert torch.equal(torch.sort(mapping[eligible]).values, torch.where(eligible)[0])
        assert torch.equal(result["support_labels"], source["support_labels"][mapping])
        assert torch.equal(result["support_labels"][~valid], torch.zeros_like(result["support_labels"][~valid]))
        torch.testing.assert_close(result["support_labels"][valid].sum(-1), torch.ones(int(valid.sum())))
        for key in source:
            if key != "support_labels":
                assert result[key] is source[key]
        assert mapping[0] == 0  # The sole [False, True] parent cannot be deranged.
        assert (mapping[torch.tensor([1, 2, 3, 9])] != torch.tensor([1, 2, 3, 9])).all()
        assert torch.equal(mapping[torch.tensor([4, 5, 6])], torch.tensor([4, 5, 6]))
        chosen = result["support_ids"][result["support_ids"] >= 0]
        donors = mapping[chosen]
        assert (events[donors] != example.diagnostics["event_id"]).all()
        assert (runtime.retrieval_bank["images"][donors] != example.diagnostics["image_id"]).all()
        provenance = factory.provenance()["label_permutation"]
        assert provenance["changed_parent_rows"] == 4
        assert provenance["actual_label_changed_rows"] == 4
        assert provenance["changed_parent_fraction"] == 0.5
        assert provenance["actual_label_changed_fraction"] == 0.5
        assert provenance["event_strata_count"] == 4
        assert provenance["recipient_to_donor"] == mapping.tolist()
        assert len(provenance["mapping_sha256"]) == 64
        assert any(record["fixed_points"] == 1 for record in provenance["strata"])
        assert torch.equal(mapping, SupportFactory(runtime, 19).permutation)


    def test_memory_arm_legality_is_checked_even_when_reusing_retrieved_ids(self):
        runtime, example = self.fixture
        example.supports["d"] = dict(example.supports["d"])
        ids = example.supports["d"]["support_ids"].clone()
        ids[0, 0] = 2
        example.supports["d"]["support_ids"] = ids
        example.diagnostics["event_id"] = -1  # Check image exclusion independently of event exclusion.
        for arm in ("d", "f", "l", "n"):
            with self.assertRaisesRegex(ValueError, "same-query images"):
                SupportFactory(runtime, 1).prepare(example, (arm,))


    def test_no_label_hook_matches_zero_embedding_math_and_restores_model(self):
        runtime, example = self.fixture
        factory = SupportFactory(runtime, 19)
        support = factory.prepare(example, ("d", "n"))
        model = _model()
        state = _clone_tree(model.state_dict())
        n = factory.forward(model, example, "n", support["n"])
        assert not model.label_embedding._forward_hooks
        zero_model = copy.deepcopy(model)
        with torch.no_grad():
            zero_model.label_embedding.weight.zero_()
        expected = zero_model(**example.common, **support["d"])
        torch.testing.assert_close(n["memory_context"], expected["memory_context"])
        torch.testing.assert_close(n["probabilities"], expected["probabilities"])
        d = factory.forward(model, example, "d", support["d"])
        assert not torch.allclose(d["memory_context"], n["memory_context"])
        assert not torch.allclose(d["probabilities"], n["probabilities"])
        n["logits"].sum().backward()
        assert model.label_embedding.weight.grad is None
        assert model.value_projection.weight.grad is not None
        _assert_tree_equal(model.state_dict(), state)
        broken = dict(support["n"], slot_valid=support["n"]["slot_valid"].long())
        with self.assertRaises(ValueError):
            factory.forward(model, example, "n", broken)
        assert not model.label_embedding._forward_hooks
        recovered = factory.forward(model, example, "d", support["d"])
        torch.testing.assert_close(recovered["memory_context"], d["memory_context"])


    def test_all_arms_run_same_original_architecture_and_single_slot_is_inactive(self):
        runtime, example = self.fixture
        factory = SupportFactory(runtime, 19)
        support = factory.prepare(example, tuple(ARM_SPECS))
        model = _model()
        count = sum(parameter.numel() for parameter in model.parameters())
        for arm in ARM_SPECS:
            output = factory.forward(model, example, arm, support[arm])
            assert torch.isfinite(output["probabilities"]).all()
            torch.testing.assert_close(output["probabilities"].sum(1), torch.ones(1, 4, 5))
            assert count == sum(parameter.numel() for parameter in model.parameters())
            if arm in ("e", "f"):
                assert torch.count_nonzero(output["slot_encoding"][:, :, 1]) == 0
                assert torch.count_nonzero(output["memory_context"][:, :, 1]) == 0


if __name__ == "__main__":
    unittest.main()
