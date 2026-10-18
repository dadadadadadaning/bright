"""CPU contracts for the D-only loss fork; no BRIGHT/CUDA validation."""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace
import unittest

import torch

import continuation as subject


def original_loss(probability, target):
    """Independent NLL + original present-class Lovasz reference."""
    labels = target.reshape(-1)
    probability = probability.permute(0, 2, 3, 1).reshape(-1, 4)
    valid = labels != 255
    labels, probability = labels[valid], probability[valid]
    if labels.numel() == 0:
        return probability.sum() * 0
    nll = torch.nn.functional.nll_loss(probability.clamp_min(1e-8).log(), labels)
    losses = []
    for category in range(4):
        foreground = (labels == category).to(probability.dtype)
        if foreground.sum() == 0:
            continue
        errors, order = (foreground - probability[:, category]).abs().sort(descending=True)
        sorted_foreground = foreground[order]
        total = sorted_foreground.sum()
        intersection = total - sorted_foreground.cumsum(0)
        union = total + (1 - sorted_foreground).cumsum(0)
        gradient = 1 - intersection / union
        if labels.numel() > 1:
            gradient[1:] = gradient[1:] - gradient[:-1]
        losses.append(torch.dot(errors, gradient))
    return nll + sum(losses) / len(losses)


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_projection = torch.nn.Conv2d(3, 4, 1)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, *, features, offset):
        logits = self.query_projection(self.dropout(features)) + offset
        return {"probabilities": logits.softmax(dim=1)}


class TinyRuntime:
    loss = staticmethod(original_loss)


def example():
    return SimpleNamespace(
        common={"features": torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3) / 17},
        supports={"d": {"offset": 0.1}},
        target=torch.tensor([[[0, 1, 2], [3, 0, 2]]]),
    )


def trained_checkpoint():
    """Real 500 AdamW updates; no fabricated optimizer step counters."""
    torch.manual_seed(3407)
    head = TinyHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.005)
    fixture = example()
    for step in range(1, 501):
        for group in optimizer.param_groups:
            group["lr"] = subject.lr_at_step(step)
        optimizer.zero_grad(set_to_none=True)
        result = head(**fixture.common, **fixture.supports["d"])
        original_loss(result["probabilities"], fixture.target).backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
    return {
        "step": 500, "mode": "train", "smoke": False, "training_complete": False,
        "heads": {"d": copy.deepcopy(head.state_dict())},
        "optimizers": {"d": copy.deepcopy(optimizer.state_dict())},
        "updates_per_arm": {"d": 500}, "rng": subject.capture_rng(),
        "config": {"steps": 5000, "eval_every": 500, "lr": 1e-4, "weight_decay": 0.005},
        "protocol_info": {"synthetic": True}, "source_hash": "synthetic",
        "input_file_sha256": {},
    }


class ContinuationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.saved = trained_checkpoint()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fork(self):
        return subject.fork_heads_and_optimizers(
            self.saved, SimpleNamespace(chunk_size=2), "cpu", model_factory=TinyHead)

    def test_only_two_d_forks_with_unmodified_schedule(self):
        self.assertEqual(subject.BINDINGS,
                         {"d_control": ("d", 1.0), "d_no_bg_lovasz": ("d", 1.0)})
        self.assertTrue(subject.validate_checkpoint_payload(self.saved))
        self.assertNotIn("b", self.saved["heads"])
        self.assertEqual(subject.lr_at_step(500), 1e-4)
        self.assertEqual(subject.lr_at_step(1500),
                         1e-4 * 0.5 * (1 + math.cos(math.pi * 1000 / 4500)))
        self.assertGreater(subject.lr_at_step(1500), 0)
        self.assertEqual(subject.lr_at_step(5000), 0)

    def test_invalid_checkpoint_or_missing_d_moments_rejected(self):
        mutations = (
            lambda s: s.update(step=1500),
            lambda s: s.update(smoke=True),
            lambda s: s.update(training_complete=True),
            lambda s: s.pop("rng"),
            lambda s: s["config"].update(steps=1500),
            lambda s: s["optimizers"].pop("d"),
            lambda s: s["optimizers"]["d"].update(state={}),
            lambda s: s["optimizers"]["d"]["param_groups"][0].update(lr=2e-5),
            lambda s: s["updates_per_arm"].update(d=499),
        )
        for mutate in mutations:
            bad = copy.deepcopy(self.saved)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                subject.validate_checkpoint_payload(bad)

    def test_bad_adam_step_or_nonfinite_moments_rejected(self):
        for fault in ("step", "missing", "nonfinite"):
            bad = copy.deepcopy(self.saved)
            state = next(iter(bad["optimizers"]["d"]["state"].values()))
            if fault == "step":
                state["step"].fill_(499)
            elif fault == "missing":
                state.pop("exp_avg_sq")
            else:
                state["exp_avg"].flatten()[0] = float("nan")
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                subject.validate_checkpoint_payload(bad)

    def test_independent_exact_model_and_adam_restoration(self):
        source_hash = subject.state_fingerprint(self.saved)
        heads, optimizers, records = self.fork()
        parameter_addresses, moment_addresses = [], []
        for name in subject.BINDINGS:
            self.assertEqual(subject.state_fingerprint(heads[name].state_dict()),
                             subject.state_fingerprint(self.saved["heads"]["d"]))
            self.assertEqual(subject.state_fingerprint(optimizers[name].state_dict()),
                             subject.state_fingerprint(self.saved["optimizers"]["d"]))
            self.assertEqual(records[name]["post500_lr_multiplier"], 1.0)
            for parameter in heads[name].parameters():
                parameter_addresses.append(parameter.data_ptr())
                moment_addresses.extend(optimizers[name].state[parameter][key].data_ptr()
                                        for key in ("exp_avg", "exp_avg_sq"))
        self.assertEqual(len(parameter_addresses), len(set(parameter_addresses)))
        self.assertEqual(len(moment_addresses), len(set(moment_addresses)))
        self.assertEqual(source_hash, subject.state_fingerprint(self.saved))

    def test_control_exactly_matches_original_loss_optimizer_and_rng(self):
        heads, optimizers, _ = self.fork()
        expected_heads, expected_optimizers, _ = self.fork()
        fixture = example()
        before_rng = subject.capture_rng()
        head, optimizer = expected_heads["d_control"], expected_optimizers["d_control"]
        for group in optimizer.param_groups:
            group["lr"] = subject.lr_at_step(501)
        optimizer.zero_grad(set_to_none=True)
        result = head(**fixture.common, **fixture.supports["d"])
        original_loss(result["probabilities"], fixture.target).backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        expected_rng = subject.state_fingerprint(subject.capture_rng())
        subject.restore_rng(before_rng)
        subject.train_matched_step(heads, optimizers, fixture, TinyRuntime(), 501, 1e-4)
        self.assertEqual(subject.state_fingerprint(heads["d_control"].state_dict()),
                         subject.state_fingerprint(head.state_dict()))
        self.assertEqual(subject.state_fingerprint(optimizers["d_control"].state_dict()),
                         subject.state_fingerprint(optimizer.state_dict()))
        self.assertEqual(subject.state_fingerprint(subject.capture_rng()), expected_rng)

    def test_first_step_common_loss_equal_but_candidate_update_differs(self):
        heads, optimizers, _ = self.fork()
        fixture = example()
        before = subject.state_fingerprint((fixture.common, fixture.supports, fixture.target))
        for head in heads.values():
            head.eval()
        rows = subject.train_matched_step(heads, optimizers, fixture, TinyRuntime(), 501, 1e-4)
        control, candidate = rows["d_control"], rows["d_no_bg_lovasz"]
        self.assertEqual(control["lr"], candidate["lr"])
        self.assertEqual(control["loss"], candidate["loss"])
        self.assertEqual(control["candidate_loss"], candidate["candidate_loss"])
        self.assertAlmostEqual(control["loss"] - candidate["training_loss"],
                               candidate["background_lovasz_contribution"], places=6)
        self.assertGreater(control["training_loss"], candidate["training_loss"])
        self.assertGreater(candidate["reader_grad_abs_sum"], 0)
        self.assertNotEqual(subject.state_fingerprint(heads["d_control"].state_dict()),
                            subject.state_fingerprint(heads["d_no_bg_lovasz"].state_dict()))
        self.assertTrue(all(head.training for head in heads.values()))
        self.assertEqual(before, subject.state_fingerprint((fixture.common, fixture.supports, fixture.target)))

    def test_no_background_target_makes_two_objectives_and_updates_match(self):
        heads, optimizers, _ = self.fork()
        fixture = example()
        fixture.target[fixture.target == 0] = 1
        rows = subject.train_matched_step(heads, optimizers, fixture, TinyRuntime(), 501, 1e-4)
        self.assertEqual(rows["d_control"]["background_lovasz_contribution"], 0)
        self.assertAlmostEqual(rows["d_control"]["training_loss"],
                               rows["d_no_bg_lovasz"]["training_loss"], places=6)
        for control, candidate in zip(heads["d_control"].parameters(),
                                      heads["d_no_bg_lovasz"].parameters()):
            torch.testing.assert_close(control, candidate, rtol=0, atol=1e-8)

    def test_actual_parameter_update_and_optimizer_count_logged(self):
        heads, optimizers, _ = self.fork()
        previous = [p.detach().clone() for p in heads["d_control"].parameters()]
        rows = subject.train_matched_step(heads, optimizers, example(), TinyRuntime(), 501, 1e-4)
        distance = math.sqrt(sum(float((p.detach().double() - q.double()).square().sum())
                                 for p, q in zip(heads["d_control"].parameters(), previous)))
        magnitude = math.sqrt(sum(float(p.double().square().sum()) for p in previous))
        row = rows["d_control"]
        self.assertEqual(row["parameter_update_l2"], distance)
        self.assertEqual(row["parameter_update_relative_l2"], distance / magnitude)
        self.assertEqual(row["optimizer_step"], 501)
        self.assertTrue(row["all_optimizer_steps_verified"])
        self.assertTrue(row["grad_norm_is_preclip"])
        self.assertEqual(row["gradient_clipped"], row["grad_norm"] > 1.0)

    def test_reference_checked_at_501_520_and_explicit_request(self):
        heads, optimizers, _ = self.fork()
        for step in range(501, 521):
            rows = subject.train_matched_step(heads, optimizers, example(), TinyRuntime(),
                                             step, 1e-4, verify_reference=(step == 507))
            for row in rows.values():
                self.assertEqual(row["reference_verified"], step in (501, 507, 520))
                self.assertEqual(row["optimizer_step"], step)

    def test_wrong_reference_or_nonfinite_loss_stops_before_control_update(self):
        class WrongRuntime:
            @staticmethod
            def loss(probability, target):
                return original_loss(probability, target) + 0.1

        class NonFiniteRuntime:
            @staticmethod
            def loss(probability, target):
                return probability.sum() * float("nan")

        for runtime in (WrongRuntime(), NonFiniteRuntime()):
            heads, optimizers, _ = self.fork()
            before = {k: subject.state_fingerprint(v.state_dict()) for k, v in heads.items()}
            with self.subTest(runtime=runtime), self.assertRaises((ValueError, RuntimeError)):
                subject.train_matched_step(heads, optimizers, example(), runtime, 501, 1e-4)
            self.assertEqual(before, {k: subject.state_fingerprint(v.state_dict())
                                      for k, v in heads.items()})

    def test_rejects_skipped_updates_and_partial_forks(self):
        for step in (500, 503, 1501):
            heads, optimizers, _ = self.fork()
            with self.subTest(step=step), self.assertRaises(ValueError):
                subject.train_matched_step(heads, optimizers, example(), TinyRuntime(), step, 1e-4)
        heads, optimizers, _ = self.fork()
        optimizers.pop("d_no_bg_lovasz")
        with self.assertRaises(ValueError):
            subject.train_matched_step(heads, optimizers, example(), TinyRuntime(), 501, 1e-4)


if __name__ == "__main__":
    unittest.main()
