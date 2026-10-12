"""CPU contracts for evidence isolation, masking, and strict J1 fallback."""
import inspect
import unittest

import torch

from .reader import GatedMemoryCorrection, drop_support, retrieve_evidence


class MemoryCorrectionContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(2)
        self.query = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        self.bank = dict(
            tokens=torch.eye(4), distribution=torch.eye(4),
            events=torch.tensor([1, 2, 3, 3]), images=torch.tensor([10, 11, 12, 13]),
        )

    def evidence(self, **kwargs):
        return retrieve_evidence(self.query, self.bank, -1, -1, **kwargs)

    def test_selection_excludes_both_provenances_and_has_no_gt_argument(self):
        evidence = retrieve_evidence(self.query, self.bank, 1, 11, topk=5)
        self.assertEqual(evidence["selected"].shape, (2, 5))
        for row in evidence["selected"].tolist():
            self.assertNotIn(0, row)
            self.assertNotIn(1, row)
            self.assertEqual(row.count(-1), 3)
        for name in ("gt", "target", "label", "class_id"):
            self.assertNotIn(name, inspect.signature(retrieve_evidence).parameters)
        torch.testing.assert_close(evidence["quality"][:, 2], torch.full((2,), 0.4))
        torch.testing.assert_close(evidence["quality"][:, 5], torch.full((2,), 0.5))

    def test_weighted_labels_are_correct_and_not_used_for_selection(self):
        evidence = self.evidence(topk=4, temperature=0.2)
        expected = torch.softmax(self.query / 0.2, dim=-1)
        torch.testing.assert_close(evidence["posterior"], expected)
        permuted = dict(self.bank, distribution=self.bank["distribution"].roll(1, dims=1))
        changed = retrieve_evidence(self.query, permuted, -1, -1, topk=4, temperature=0.2)
        self.assertTrue(torch.equal(evidence["selected"], changed["selected"]))
        torch.testing.assert_close(changed["posterior"], expected.roll(1, dims=1))
        self.assertFalse(torch.equal(evidence["posterior"], changed["posterior"]))

    def test_soft_label_values_preserved_and_invalid_label_rejected(self):
        self.bank["distribution"][0] = torch.tensor([0.2, 0.8, 0., 0.])
        evidence = self.evidence(topk=1)
        torch.testing.assert_close(evidence["posterior"][0], self.bank["distribution"][0])
        self.bank["distribution"][0, 0] = -1
        with self.assertRaises(ValueError):
            self.evidence()

    def test_empty_and_fully_excluded_bank_are_finite_and_zero(self):
        empty = {key: value[:0] for key, value in self.bank.items()}
        for bank in (empty, dict(self.bank, events=torch.zeros(4, dtype=torch.long))):
            evidence = retrieve_evidence(self.query, bank, 0, -1, topk=8)
            self.assertFalse(evidence["supported"].any())
            self.assertTrue((evidence["selected"] == -1).all())
            for key in ("scores", "weights", "posterior", "quality", "support_distribution"):
                self.assertTrue(torch.isfinite(evidence[key]).all())
                self.assertEqual(torch.count_nonzero(evidence[key]).item(), 0)

    def test_dropout_recomputes_labels_quality_and_does_not_mutate(self):
        evidence = self.evidence(topk=4)
        original = evidence["valid"].clone()
        dropped = drop_support(evidence, 0.5, torch.Generator().manual_seed(6))
        self.assertTrue(torch.equal(evidence["valid"], original))
        self.assertEqual(dropped["valid"].shape, original.shape)
        for row in range(len(self.query)):
            valid = dropped["valid"][row]
            if valid.any():
                expected_weights = dropped["scores"][row, valid].div(0.1).softmax(-1)
                expected = (expected_weights[:, None] * dropped["support_distribution"][row, valid]).sum(0)
                torch.testing.assert_close(dropped["posterior"][row], expected)
                events = dropped["support_events"][row, valid]
                self.assertAlmostEqual(dropped["quality"][row, 5].item(), len(events.unique()) / len(events))
        gone = drop_support(evidence, 1.0)
        for key in ("scores", "weights", "posterior", "quality", "support_distribution"):
            self.assertEqual(torch.count_nonzero(gone[key]).item(), 0)
        same = drop_support(evidence, 0.0)
        for key in ("scores", "weights", "posterior", "quality", "valid"):
            torch.testing.assert_close(same[key], evidence[key], rtol=0, atol=0)
        no_rows = drop_support(evidence, 0, row_probability=1)
        self.assertFalse(no_rows["supported"].any())
        self.assertEqual(torch.count_nonzero(no_rows["quality"]).item(), 0)

    def test_initial_identity_bounds_ungated_and_strict_fallback(self):
        evidence = self.evidence()
        model = GatedMemoryCorrection(dim=4, hidden=8, bound=2)
        base = self.query.softmax(-1)
        delta, gate, _ = model(self.query, base, evidence)
        self.assertEqual(torch.count_nonzero(delta).item(), 0)
        torch.testing.assert_close(gate, torch.full((2,), 0.5))
        with torch.no_grad():
            model.delta_head.bias.fill_(10)
        delta, gate, _ = model(self.query, base, evidence, mode="ungated")
        self.assertTrue((delta.abs() <= 2).all())
        self.assertTrue((gate == 1).all())
        for value, force_off in ((evidence, True), (drop_support(evidence, 1), False)):
            delta, gate, _ = model(self.query, base, value, force_off=force_off)
            self.assertEqual(torch.count_nonzero(delta).item(), 0)
            self.assertEqual(torch.count_nonzero(gate).item(), 0)
            baseline = torch.randn(2, 4)
            self.assertTrue(torch.equal(baseline + delta, baseline))

    def test_rejected_nonfinite_inputs_do_not_create_nan(self):
        evidence = drop_support(self.evidence(), 1)
        evidence["posterior"].fill_(float("nan"))
        evidence["quality"].fill_(float("inf"))
        model = GatedMemoryCorrection(dim=4, hidden=8)
        delta, gate, diagnostics = model(self.query * float("nan"), self.query * float("inf"), evidence)
        self.assertTrue(torch.isfinite(delta).all())
        self.assertEqual(torch.count_nonzero(delta).item(), 0)
        self.assertEqual(torch.count_nonzero(gate).item(), 0)
        self.assertTrue(torch.isfinite(diagnostics["raw_delta"]).all())

    def test_delta_gradient_then_gate_gradient_without_frozen_input_gradient(self):
        evidence = self.evidence()
        model = GatedMemoryCorrection(dim=4, hidden=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        query = self.query.clone().requires_grad_()
        base = self.query.softmax(-1).requires_grad_()
        delta, _, _ = model(query, base, evidence)
        (delta[:, 0] - 1).square().mean().backward()
        self.assertGreater(model.delta_head.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(query.grad)
        self.assertIsNone(base.grad)
        optimizer.step()
        optimizer.zero_grad()
        delta, _, _ = model(query, base, evidence)
        (delta[:, 0] - 1).square().mean().backward()
        self.assertGreater(model.gate_head.weight.grad.abs().sum().item(), 0)

    def test_quality_can_affect_learned_rejection(self):
        evidence = self.evidence()
        model = GatedMemoryCorrection(dim=4, hidden=1)
        # Construct a deterministic quality-only gate to verify the quality
        # features reach it; this does not claim dropout calibrates reliability.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.encoder[0].weight[0, 4 + 4 + 2 + 4 + 2] = 1
            model.encoder[2].weight[0, 0] = 1
            model.gate_head.weight[0, 0] = 4
            model.delta_head.bias.fill_(0.3)
        low = dict(evidence, quality=evidence["quality"].clone())
        low["quality"][:, 2] = 0.1
        _, high_gate, _ = model(self.query, self.query.softmax(-1), evidence)
        _, low_gate, _ = model(self.query, self.query.softmax(-1), low)
        self.assertTrue((high_gate > low_gate).all())


if __name__ == "__main__":
    unittest.main()
