"""Synthetic CPU checks for label memory, calibration, and retrieval isolation.

Run with ``python -m unittest
bda_benchmark.experiments.damageformer_memory_label_v1.test_core -v``.
These checks establish implementation contracts, not real-data benefit.
"""

import inspect
import math
import unittest

import torch
from torch.nn import functional as F

from .core import (
    LabelMixer,
    mix_fixed,
    select_neighbors,
    shuffled_distribution,
    weighted_labels,
)


def make_bank():
    return {
        "tokens": torch.tensor([
            [3., 0.], [2., 1.], [1., 2.],
            [0., 4.], [-2., 1.], [-1., -2.],
        ]),
        "distribution": torch.tensor([
            [1., 0., 0., 0.], [0., 1., 0., 0.],
            [0., .8, .2, 0.], [0., .1, .9, 0.],
            [0., 0., .3, .7], [0., 0., 0., 1.],
        ]),
        "events": torch.tensor([0, 0, 1, 1, 2, 2]),
        "images": torch.tensor([10, 11, 12, 13, 14, 15]),
    }


def base_probabilities():
    return torch.tensor([[.1, .6, .2, .1], [.4, .1, .2, .3]])


def support_data():
    scores = torch.tensor([[.9, .2], [.1, .7]])
    labels = torch.tensor([
        [[0., .2, .8, 0.], [0., 0., .1, .9]],
        [[.2, .3, .5, 0.], [0., .1, .2, .7]],
    ])
    return scores, labels, torch.ones((2, 2), dtype=torch.bool)


class LabelMixerTests(unittest.TestCase):
    def test_both_controls_have_exactly_two_zero_initialized_scalars(self):
        for mode in ("memory", "nomemory"):
            with self.subTest(mode=mode):
                mixer = LabelMixer(mode)
                parameters = dict(mixer.named_parameters())
                self.assertEqual(set(parameters), {"alpha", "log_temperature"})
                self.assertEqual(sum(p.numel() for p in parameters.values()), 2)
                self.assertTrue(all(p.requires_grad for p in parameters.values()))
                self.assertTrue(all(torch.equal(p, torch.zeros_like(p))
                                    for p in parameters.values()))
                self.assertEqual(mixer.values(), {"alpha": 0., "temperature": 1.})

    def test_memory_zero_alpha_preserves_base_bit_for_bit(self):
        mixer = LabelMixer("memory")
        base = base_probabilities()
        scores, labels, valid = support_data()
        self.assertTrue(torch.equal(mixer(base, scores, labels, valid), base))

    def test_nomemory_initialization_preserves_base_with_roundoff_only(self):
        base = base_probabilities()
        torch.testing.assert_close(LabelMixer("nomemory")(base), base,
                                   rtol=0., atol=1e-7)

    def test_memory_convex_mixture_matches_analytical_result(self):
        mixer = LabelMixer("memory")
        with torch.no_grad():
            mixer.alpha.fill_(.35)
            mixer.log_temperature.fill_(math.log(.6))
        base = base_probabilities()
        scores, labels, valid = support_data()
        evidence = (torch.softmax(scores / .6, dim=-1)[..., None] * labels).sum(1)
        expected = .65 * base + .35 * evidence
        actual = mixer(base, scores, labels, valid)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(bool((actual >= 0).all()))
        torch.testing.assert_close(actual.sum(-1), torch.ones(len(base)))

    def test_unsupported_rows_are_exact_base_despite_masked_nans(self):
        mixer = LabelMixer("memory")
        with torch.no_grad():
            mixer.alpha.fill_(.5)
        base = base_probabilities()
        scores, labels, valid = support_data()
        valid[1] = False
        scores[1] = float("nan")
        labels[1] = float("nan")
        actual = mixer(base, scores, labels, valid)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(torch.equal(actual[1], base[1]))
        self.assertFalse(torch.equal(actual[0], base[0]))

    def test_memory_negative_alpha_is_clamped_to_identity(self):
        mixer = LabelMixer("memory")
        with torch.no_grad():
            mixer.alpha.fill_(-.4)
        base = base_probabilities()
        self.assertTrue(torch.equal(mixer(base, *support_data()), base))
        mixer.project_()
        self.assertEqual(float(mixer.alpha.detach()), 0.)

    def test_projection_enforces_mode_specific_alpha_and_temperature_bounds(self):
        for mode, minimum_alpha in (("memory", 0.), ("nomemory", -.5)):
            mixer = LabelMixer(mode)
            with torch.no_grad():
                mixer.alpha.fill_(-100.)
                mixer.log_temperature.fill_(-100.)
            mixer.project_()
            self.assertAlmostEqual(float(mixer.alpha.detach()), minimum_alpha)
            self.assertAlmostEqual(mixer.values()["temperature"], .02, places=6)
            with torch.no_grad():
                mixer.alpha.fill_(100.)
                mixer.log_temperature.fill_(100.)
            mixer.project_()
            self.assertAlmostEqual(float(mixer.alpha.detach()), .5)
            self.assertAlmostEqual(mixer.values()["temperature"], 4., places=6)

    def test_nomemory_is_independent_of_all_support_inputs(self):
        mixer = LabelMixer("nomemory")
        with torch.no_grad():
            mixer.alpha.fill_(.2)
            mixer.log_temperature.fill_(math.log(.7))
        base = base_probabilities()
        expected = torch.softmax(base.log() / .7 + torch.tensor([0., .2, .2, .2]), -1)
        actual = mixer(base)
        torch.testing.assert_close(actual, expected)
        poisoned = mixer(base, scores=torch.full((2, 3), float("nan")),
                         labels=torch.full((2, 3, 4), float("nan")), valid=None)
        self.assertTrue(torch.equal(actual, poisoned))

    def test_nomemory_has_temperature_and_bias_gradients_at_initialization(self):
        mixer = LabelMixer("nomemory")
        base = base_probabilities()
        loss = -mixer(base)[0, 0].log()
        loss.backward()
        for name, parameter in mixer.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad)))
                self.assertGreater(abs(float(parameter.grad)), 1e-6)

    def test_only_mixer_receives_gradients_with_frozen_inputs(self):
        query = torch.tensor([[1., .1], [.2, 1.]], requires_grad=True)
        base = base_probabilities().requires_grad_()
        bank = make_bank()
        bank["tokens"].requires_grad_()
        bank["distribution"].requires_grad_()
        snapshot = {key: value.detach().clone() for key, value in bank.items()}
        result = select_neighbors(query.detach(), bank, event_id=0, image_id=10, topk=3)
        self.assertFalse(result["scores"].requires_grad)
        self.assertFalse(result["labels"].requires_grad)
        mixer = LabelMixer("memory")
        with torch.no_grad():
            mixer.alpha.fill_(.25)
        prediction = mixer(base.detach(), result["scores"], result["labels"], result["valid"])
        (-prediction[:, 3].log().mean()).backward()
        self.assertIsNone(base.grad)
        self.assertIsNone(query.grad)
        self.assertIsNone(bank["tokens"].grad)
        self.assertIsNone(bank["distribution"].grad)
        for parameter in mixer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad)))
            self.assertGreater(abs(float(parameter.grad)), 1e-6)
        for key in bank:
            self.assertTrue(torch.equal(bank[key].detach(), snapshot[key]))

    def test_zero_alpha_can_remain_a_valid_optimization_result(self):
        mixer = LabelMixer("memory")
        base = torch.tensor([[.01, .97, .01, .01]])
        scores = torch.tensor([[1.]])
        labels = torch.tensor([[[0., 0., 0., 1.]]])
        valid = torch.ones((1, 1), dtype=torch.bool)
        optimizer = torch.optim.SGD(mixer.parameters(), lr=.1)
        loss = -mixer(base, scores, labels, valid)[0, 1].log()
        loss.backward()
        self.assertGreater(float(mixer.alpha.grad), 0.)
        optimizer.step()
        mixer.project_()
        self.assertEqual(mixer.values()["alpha"], 0.)
        self.assertTrue(torch.equal(mixer(base, scores, labels, valid), base))


class WeightedLabelsTests(unittest.TestCase):
    def test_soft_labels_are_not_replaced_with_majority_classes(self):
        scores = torch.zeros((1, 2))
        labels = torch.tensor([[[.49, .51, 0., 0.], [0., .01, .99, 0.]]])
        valid = torch.ones((1, 2), dtype=torch.bool)
        probability, supported = weighted_labels(scores, labels, valid, 1.)
        torch.testing.assert_close(probability, torch.tensor([[.245, .26, .495, 0.]]))
        majority_vote = F.one_hot(labels.argmax(-1), 4).float().mean(1)
        self.assertFalse(torch.allclose(probability, majority_vote))
        self.assertEqual(supported.tolist(), [True])

    def test_masked_nan_values_do_not_affect_valid_evidence(self):
        scores = torch.tensor([[.5, float("nan")], [float("nan"), float("nan")]])
        labels = torch.full((2, 2, 4), float("nan"))
        labels[0, 0] = torch.tensor([.1, .2, .3, .4])
        valid = torch.tensor([[True, False], [False, False]])
        probability, supported = weighted_labels(scores, labels, valid, .02)
        self.assertTrue(bool(torch.isfinite(probability).all()))
        torch.testing.assert_close(probability[0], labels[0, 0])
        self.assertEqual(supported.tolist(), [True, False])

    def test_zero_mass_invalid_support_is_ignored(self):
        labels = torch.tensor([[[.1, .2, .3, .4], [0., 0., 0., 0.]]])
        probability, supported = weighted_labels(
            torch.tensor([[0., 100.]]), labels, torch.tensor([[True, False]]), 1.)
        torch.testing.assert_close(probability, labels[:, 0])
        self.assertEqual(supported.tolist(), [True])

    def test_zero_mass_valid_support_is_rejected(self):
        with self.assertRaises(ValueError):
            weighted_labels(torch.zeros((1, 1)), torch.zeros((1, 1, 4)),
                            torch.ones((1, 1), dtype=torch.bool), 1.)

    def test_invalid_temperatures_and_valid_nan_labels_are_rejected(self):
        scores, labels, valid = support_data()
        for temperature in (0., -1., float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                weighted_labels(scores, labels, valid, temperature)
        labels[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            weighted_labels(scores, labels, valid, 1.)


class NeighborSelectionTests(unittest.TestCase):
    def test_same_image_and_same_event_are_excluded_independently(self):
        bank = make_bank()
        query = torch.tensor([[1., 0.], [0., 1.]])
        for intervention in ("normal", "random", "shuffle_labels"):
            with self.subTest(intervention=intervention):
                result = select_neighbors(query, bank, event_id=1, image_id=10,
                                          topk=8, intervention=intervention,
                                          generator=torch.Generator().manual_seed(7))
                selected = result["indices"][result["valid"]]
                self.assertTrue(bool((bank["events"][selected] != 1).all()))
                self.assertTrue(bool((bank["images"][selected] != 10).all()))
                self.assertEqual(set(selected.tolist()), {1, 4, 5})
                self.assertEqual(result["valid"].sum(-1).tolist(), [3, 3])

    def test_normal_neighbors_rank_by_cosine_with_real_scores(self):
        bank = make_bank()
        query = torch.tensor([[1., .1], [.2, 1.]])
        result = select_neighbors(query, bank, event_id=0, image_id=10, topk=2)
        cosine = F.normalize(query, dim=-1) @ F.normalize(bank["tokens"], dim=-1).T
        legal_scores = cosine.masked_fill(bank["events"][None] == 0, -torch.inf)
        expected_scores, expected_indices = legal_scores.topk(2, dim=-1)
        self.assertTrue(torch.equal(result["indices"], expected_indices))
        torch.testing.assert_close(result["scores"], expected_scores)
        torch.testing.assert_close(result["labels"], bank["distribution"][expected_indices])

    def test_random_neighbors_are_legal_and_still_use_cosine_weights(self):
        bank = make_bank()
        query = torch.tensor([[1., .1]]).expand(24, -1).clone()
        result = select_neighbors(query, bank, event_id=0, image_id=10, topk=2,
                                  intervention="random",
                                  generator=torch.Generator().manual_seed(19))
        self.assertTrue(bool(result["valid"].all()))
        selected = result["indices"]
        self.assertTrue(bool((bank["events"][selected] != 0).all()))
        self.assertTrue(bool((bank["images"][selected] != 10).all()))
        cosine = F.normalize(query, dim=-1) @ F.normalize(bank["tokens"], dim=-1).T
        torch.testing.assert_close(result["scores"], cosine.gather(1, selected))
        self.assertGreater(len({tuple(row) for row in selected.tolist()}), 1)

    def test_random_intervention_is_reproducible_from_seed(self):
        bank = make_bank()
        query = torch.tensor([[1., .1], [.2, 1.]])
        results = [select_neighbors(query, bank, event_id=0, image_id=10, topk=3,
                                    intervention="random",
                                    generator=torch.Generator().manual_seed(29))
                   for _ in range(2)]
        for key in results[0]:
            self.assertTrue(torch.equal(results[0][key], results[1][key]))

    def test_label_shuffle_preserves_neighbor_indices_scores_and_eligible_labels(self):
        bank = make_bank()
        query = torch.tensor([[1., .1], [.2, 1.]])
        normal = select_neighbors(query, bank, event_id=1, image_id=10, topk=8)
        shuffled = select_neighbors(query, bank, event_id=1, image_id=10, topk=8,
                                    intervention="shuffle_labels",
                                    generator=torch.Generator().manual_seed(31))
        for key in ("indices", "scores", "valid"):
            self.assertTrue(torch.equal(normal[key], shuffled[key]))
        eligible_labels = bank["distribution"][[1, 4, 5]]
        expected = sorted(map(tuple, eligible_labels.tolist()))
        for labels, valid in zip(shuffled["labels"], shuffled["valid"]):
            self.assertEqual(sorted(map(tuple, labels[valid].tolist())), expected)

    def test_precomputed_shuffle_is_eligible_only_and_chunk_invariant(self):
        bank = make_bank()
        snapshot = bank["distribution"].clone()
        distribution = shuffled_distribution(
            bank, event_id=1, image_id=10,
            generator=torch.Generator().manual_seed(37))
        excluded = (bank["events"] == 1) | (bank["images"] == 10)
        self.assertTrue(torch.equal(distribution[excluded], snapshot[excluded]))
        self.assertEqual(sorted(map(tuple, distribution[~excluded].tolist())),
                         sorted(map(tuple, snapshot[~excluded].tolist())))
        query = torch.tensor([[1., .1], [.2, 1.], [-1., .4], [-.4, -1.]])
        whole = select_neighbors(query, bank, 1, 10, topk=3, distribution=distribution)
        chunks = [select_neighbors(part, bank, 1, 10, topk=3, distribution=distribution)
                  for part in query.split(2)]
        for key in whole:
            self.assertTrue(torch.equal(whole[key], torch.cat([part[key] for part in chunks])))
        self.assertTrue(torch.equal(bank["distribution"], snapshot))

    def test_no_eligible_support_falls_back_without_nan(self):
        bank = make_bank()
        bank["events"].fill_(0)
        base = base_probabilities()
        result = select_neighbors(torch.tensor([[1., 0.], [0., 1.]]), bank,
                                  event_id=0, image_id=10, topk=8)
        self.assertFalse(bool(result["valid"].any()))
        self.assertTrue(bool(torch.isfinite(result["scores"]).all()))
        self.assertTrue(bool(torch.isfinite(result["labels"]).all()))
        mixer = LabelMixer("memory")
        with torch.no_grad():
            mixer.alpha.fill_(.5)
        actual = mixer(base, result["scores"], result["labels"], result["valid"])
        self.assertTrue(torch.equal(actual, base))

    def test_query_ground_truth_is_not_an_accepted_retrieval_or_mixer_argument(self):
        forbidden = {"gt", "query_gt", "target", "targets", "query_labels"}
        self.assertFalse(forbidden.intersection(inspect.signature(select_neighbors).parameters))
        self.assertFalse(forbidden.intersection(inspect.signature(LabelMixer.forward).parameters))
        query = torch.tensor([[1., 0.]])
        for query_gt in (torch.tensor([0]), torch.tensor([3])):
            with self.subTest(query_gt=int(query_gt[0])), self.assertRaises(TypeError):
                select_neighbors(query, make_bank(), 0, 10, query_gt=query_gt)


class FixedMixtureTests(unittest.TestCase):
    def test_zero_and_one_mixture_have_expected_endpoints_and_fallback(self):
        base = base_probabilities()
        evidence = torch.tensor([[0., 0., 0., 1.], [0., 1., 0., 0.]])
        supported = torch.tensor([True, False])
        self.assertTrue(torch.equal(mix_fixed(base, evidence, supported, 0.), base))
        actual = mix_fixed(base, evidence, supported, 1.)
        torch.testing.assert_close(actual[0], evidence[0], rtol=0., atol=1e-7)
        self.assertTrue(torch.equal(actual[1], base[1]))

    def test_invalid_fixed_mixture_weights_are_rejected(self):
        base = base_probabilities()
        for alpha in (-.01, 1.01, float("nan"), float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                mix_fixed(base, base, torch.ones(2, dtype=torch.bool), alpha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
