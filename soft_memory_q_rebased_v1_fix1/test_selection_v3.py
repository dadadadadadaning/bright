"""Regression against the actual V3 cosine ranker used by the server bridge."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

from server_runtime import searchable_parent_view, validate_selected
from test_build_values import parent_fixture


def real_v3_select_neighbors():
    candidates = (
        Path('/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3/core.py'),
        Path(__file__).resolve().parents[1] / 'references_v3/core.py',
    )
    source = None
    for path in candidates:
        try:
            if path.is_file():
                source = path
                break
        except PermissionError:
            continue
    if source is None:
        raise RuntimeError('Actual V3 core.py is required for the selection regression')
    spec = importlib.util.spec_from_file_location('actual_v3_core_for_selection_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_neighbors


class V3SelectionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.select = staticmethod(real_v3_select_neighbors())

    def select_ids(self, query, bank, key):
        return self.select(query, bank, key_name=key, event_id=0,
                           image_id=0, topk=3, policy='global', chunk_size=2)

    def test_partial_child_bank_excludes_unwritten_rows_before_topk(self):
        parent = parent_fixture()
        parent['eligible'] = torch.tensor([True, True, False, False])
        original_keys = {key: parent[key].clone() for key in ('t128', 'f256')}
        view = searchable_parent_view(parent)
        for key in ('t128', 'f256'):
            query = parent[key][0:1]
            raw_ids, _ = self.select_ids(query, parent, key)
            self.assertTrue((raw_ids >= 2).any(), 'V3 ignores the added eligible field')
            ids, scores = self.select_ids(query, view, key)
            self.assertEqual(ids.tolist(), [[1, -1, -1]])
            self.assertTrue(torch.isfinite(scores[0, 0]))
            self.assertTrue(torch.isneginf(scores[0, 1:]).all())
            validate_selected(ids, parent, event_id=0, image_id=0)
            self.assertTrue(torch.equal(parent[key], original_keys[key]))
            self.assertTrue(torch.equal(view[key][2:], torch.zeros_like(view[key][2:])))
        for field in ('member_ids', 'images', 'events', 'regions'):
            self.assertIs(view[field], parent[field], 'Parent row identity must not be rewritten')

    def test_no_legal_cross_event_parent_returns_padding(self):
        parent = parent_fixture()
        parent['eligible'] = torch.tensor([True, False, False, False])
        view = searchable_parent_view(parent)
        for key in ('t128', 'f256'):
            ids, scores = self.select_ids(parent[key][0:1], view, key)
            self.assertEqual(ids.tolist(), [[-1, -1, -1]])
            self.assertTrue(torch.isneginf(scores).all())
            validate_selected(ids, parent, event_id=0, image_id=0)

    def test_full_child_bank_matches_original_v3_selection_exactly(self):
        parent = parent_fixture()
        view = searchable_parent_view(parent)
        for key in ('t128', 'f256'):
            query = parent[key][0:1]
            old_ids, old_scores = self.select_ids(query, parent, key)
            new_ids, new_scores = self.select_ids(query, view, key)
            self.assertTrue(torch.equal(new_ids, old_ids))
            self.assertTrue(torch.equal(new_scores, old_scores))
            self.assertTrue(torch.equal(view[key], parent[key]))
            validate_selected(new_ids, parent, event_id=0, image_id=0)


if __name__ == '__main__':
    unittest.main()
