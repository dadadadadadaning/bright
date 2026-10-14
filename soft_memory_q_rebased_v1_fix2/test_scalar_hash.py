"""Regression checks for scalar buffers in PyTorch 2.2 state dictionaries."""
from __future__ import annotations

import hashlib
import json
import unittest

import torch
from torch import nn

from common import module_hash
from server_runtime import tree_hash
from base_runtime import _bank_fingerprint
from fresh_parent_bank import _tensor_sha


class ScalarHashTests(unittest.TestCase):
    def test_module_hash_accepts_zero_dimensional_long_buffer(self):
        module = nn.Module()
        module.register_buffer("num_batches_tracked", torch.tensor(17, dtype=torch.long))
        value = module.num_batches_tracked
        expected = hashlib.sha256(
            b"num_batches_tracked" + str(value.dtype).encode() + b"()"
            + value.numpy().tobytes()
        ).hexdigest()
        self.assertEqual(module_hash(module), expected)
        module.num_batches_tracked.add_(1)
        self.assertNotEqual(module_hash(module), expected)

    def test_module_hash_preserves_vector_hash(self):
        module = nn.Module()
        module.register_buffer("values", torch.tensor([17, 18], dtype=torch.long))
        value = module.values
        legacy = hashlib.sha256(
            b"values" + str(value.dtype).encode() + b"(2,)"
            + value.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        self.assertEqual(module_hash(module), legacy)

    def test_tree_and_bank_hashes_accept_scalar_and_detect_changes(self):
        scalar = torch.tensor(17, dtype=torch.long)
        vector = torch.tensor([3, 5], dtype=torch.long)
        vector_bytes = vector.view(torch.uint8).numpy().tobytes()
        legacy_tree = hashlib.sha256(
            b"tensor" + str(vector.dtype).encode() + b"(2,)" + vector_bytes
        ).hexdigest()
        legacy_bank = hashlib.sha256(
            b"values" + str(vector.dtype).encode() + b"(2,)" + vector_bytes
        ).hexdigest()
        self.assertEqual(tree_hash(vector), legacy_tree)
        self.assertEqual(_bank_fingerprint({"values": vector}), legacy_bank)
        tree = {"nested": {"count": scalar, "values": vector}}
        bank = {"count": scalar, "values": vector}
        tree_before, bank_before = tree_hash(tree), _bank_fingerprint(bank)
        scalar.add_(1)
        self.assertNotEqual(tree_hash(tree), tree_before)
        self.assertNotEqual(_bank_fingerprint(bank), bank_before)

    def test_tensor_sha_preserves_vector_bytes_and_accepts_scalar(self):
        for value in (torch.tensor(17, dtype=torch.long),
                      torch.tensor([17, 18], dtype=torch.long)):
            header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                                sort_keys=True, separators=(",", ":")).encode()
            expected = hashlib.sha256(header + value.numpy().tobytes()).hexdigest()
            self.assertEqual(_tensor_sha(value), expected)


if __name__ == "__main__":
    unittest.main()
