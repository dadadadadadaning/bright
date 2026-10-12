# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.cmr.local_verifier import LOCAL_CORRELATION_NAMES, token_correlation_statistics
from model.DamageFormer_S8_MICB_DSMIL import MICBDSMILHead
from model.cmr.pooling import build_cmr_feature_maps, extract_cmr_local_tokens


class CMRV0LocalVerifierTests(unittest.TestCase):
    def test_identical_tokens_have_stronger_correlation_than_orthogonal(self):
        query = torch.tensor([[[1., 0.], [0., 1.]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        identical = token_correlation_statistics(query, mask, query, mask)
        different = token_correlation_statistics(
            query, mask, torch.tensor([[[-1., 0.], [0., -1.]]]), mask
        )
        self.assertEqual(identical.shape[1], len(LOCAL_CORRELATION_NAMES))
        self.assertGreater(float(identical[0, 0]), float(different[0, 0]))

    def test_padding_never_changes_statistics(self):
        query = torch.tensor([[[1., 0.], [0., 1.], [99., 99.]]])
        support = torch.tensor([[[1., 0.], [0., 1.], [-99., -99.]]])
        mask = torch.tensor([[True, True, False]])
        padded = token_correlation_statistics(query, mask, support, mask)
        compact = token_correlation_statistics(query[:, :2], mask[:, :2], support[:, :2], mask[:, :2])
        torch.testing.assert_close(padded, compact)

    def test_pooling_exposes_exact_local_token_contract(self):
        torch.manual_seed(3)
        head = MICBDSMILHead(in_channels=4, token_dim=8, topk=2, max_tokens=5).eval()
        maps = build_cmr_feature_maps(
            head, torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)
        )
        mask = torch.ones(8, 8)
        tokens = extract_cmr_local_tokens(head, maps, mask)
        self.assertEqual(tuple(tokens.shape), (5, 8))
        torch.testing.assert_close(tokens.norm(dim=1), torch.ones(5), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

