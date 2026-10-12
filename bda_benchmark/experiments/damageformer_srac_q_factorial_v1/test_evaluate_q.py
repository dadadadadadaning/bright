"""CPU-only validation of formal evaluation gates."""

from __future__ import annotations

import unittest

from .evaluate_q import _checked_checkpoint, _select_images


def payload():
    return {
        "format": "SRAC_Q_FACTORIAL_V1", "run_complete": True,
        "test_accessed": False, "a1_unchanged": True,
        "downstream_unchanged": True,
        "immutable": {"e0_files": True, "protocol": True, "q_code": True,
                      "runtime": True, "frozen_downstream": True},
        "protocol_sha256": "a" * 64, "a1_sha256": "b" * 64,
        "front_sha256": "c" * 64, "smoke": False,
        "completed_steps": 2000, "step": 2000,
        "arguments": {"steps": 2000}, "arm": "semantic_reconstruction",
    }


def bundle():
    return {
        "protocol_sha256": "a" * 64,
        "hashes": {"a1_sha256": "b" * 64, "front_sha256": "c" * 64},
    }


class EvaluationGateTests(unittest.TestCase):
    def test_only_full_equal_budget_immutable_train_can_be_formal(self):
        _checked_checkpoint(payload(), bundle(), "d" * 64, smoke=False)
        for change in (
            {"completed_steps": 1999, "step": 1999, "arguments": {"steps": 1999}},
            {"run_complete": False}, {"a1_unchanged": False},
            {"downstream_unchanged": False},
            {"smoke": True}, {"test_accessed": True},
            {"immutable": {"runtime": False}},
        ):
            row = payload()
            row.update(change)
            with self.assertRaises(ValueError):
                _checked_checkpoint(row, bundle(), "d" * 64, smoke=False)

    def test_prefix_is_always_debug_not_full(self):
        names = [f"image_{i:03d}" for i in range(450)]
        full, smoke = _select_images(names, 0)
        self.assertEqual(len(full), 450)
        self.assertFalse(smoke)
        short, smoke = _select_images(names, 2)
        self.assertEqual(short, names[:2])
        self.assertTrue(smoke)
        with self.assertRaises(ValueError):
            _select_images(names, 450)


if __name__ == "__main__":
    unittest.main()
