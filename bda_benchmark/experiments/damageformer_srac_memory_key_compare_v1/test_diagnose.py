"""Evidence gates, no-support semantics, and persisted diagnostic integrity."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn

from .diagnose import (
    _record_failure, _verify_saved_artifacts, _vote_probability,
    _weighted_evidence_diagnostic, formal_gate, run)
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import (
    NativeA1LogitsCapture, native_grid_probq)

def _saved_fixture(root: Path):
    arrays = dict(target=np.zeros((1, 2, 2), dtype=np.int16),
                  prediction_grid_probq=np.zeros((1, 2, 2), dtype=np.uint8),
                  support_t128_global=np.array([[0, -1]], dtype=np.int32),
                  scores_t128_global=np.array([[.8, -np.inf]], dtype=np.float32),
                  support_t128_global_64=np.array([[0, -1]], dtype=np.int32),
                  scores_t128_global_64=np.array([[.8, -np.inf]], dtype=np.float32),
                  softlabels_t128_global=np.array([[[0., 1., 0., 0.], [0., 0., 0., 0.]]], dtype=np.float32),
                  evidence_t128_global=np.array([[0., 1., 0., 0.]], dtype=np.float32),
                  quality_t128_global=np.ones((1, 6), dtype=np.float32),
                  supported_mass_t128_global=np.ones((1, 2, 2), dtype=np.float32))
    artifact = root / "0000.npz"
    np.savez_compressed(artifact, **arrays)
    row = dict(index=0, image="image_a", prediction_file=artifact.name,
               prediction_sha256=digest(artifact))
    (root / "images.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return artifact, row, arrays

class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def patch(self, target, name, value):
        patcher = patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)


    def test_formal_gate_requires_both_reproductions_full450_and_immutable_boolean(self):
        args = [450, 450, {"passed": True}, {"passed": True}, {"files": True}, True, True]
        assert formal_gate(*args)
        for position, replacement in ((0, 2), (1, 2), (2, {"passed": None}),
                                       (3, {"passed": False}), (4, {}),
                                       (4, {"files": False}), (4, {"files": 1}),
                                       (5, False), (6, False)):
            changed = args.copy()
            changed[position] = replacement
            assert not formal_gate(*changed)

    def test_vote_is_standalone_and_unsupported_pixels_fall_back_exactly(self):
        baseline = torch.tensor([[[[.1, .2]], [[.2, .3]], [[.3, .4]], [[.4, .1]]]])
        result = dict(q=torch.ones(1, 1, 2), ids=torch.tensor([[0, 1]]), shape=(1, 2))
        evidence = dict(posterior=torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.]]),
                        supported=torch.tensor([True, False]))
        probability, coverage = _vote_probability(evidence, result, (1, 2), baseline)
        assert torch.equal(probability[0, :, 0, 0], evidence["posterior"][0])
        assert torch.equal(probability[0, :, 0, 1], baseline[0, :, 0, 1])
        assert torch.equal(coverage, torch.tensor([[[1., 0.]]]))
        assert torch.equal(probability.sum(1), torch.ones(1, 1, 2))

    def test_same_pass_native_hook_captures_without_second_classifier_call(self):
        class Front(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Module()
                self.base.clf_2 = nn.Conv2d(2, 4, 1)

            def forward(self, feature):
                logits = self.base.clf_2(feature)
                return dict(old_pixel=logits, q=torch.ones(1, 1, 2),
                            ids=torch.tensor([[0, 1]]), count=2, shape=(1, 2))

        torch.manual_seed(1)
        front, feature = Front(), torch.randn(1, 2, 1, 2)
        observed_calls = []
        independent_hook = front.base.clf_2.register_forward_hook(
            lambda _m, _i, _o: observed_calls.append(1))
        capture = NativeA1LogitsCapture(front)
        try:
            capture.clear()
            result = front(feature)
            low_logits = capture.take()
            probability, replay = native_grid_probq(result, feature, low_logits, (1, 2))
            assert replay["passed"] is True
            assert observed_calls == [1]
            assert torch.allclose(probability, result["old_pixel"].softmax(1), atol=1e-6)
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                capture.take()
            capture.clear()
            front(feature)
            front(feature)
            with self.assertRaisesRegex(RuntimeError, "got 2"):
                capture.take()
        finally:
            capture.close()
            independent_hook.remove()

    def test_posthoc_evidence_reports_mixed_region_and_intact_misleading_vote(self):
        evidence = dict(posterior=torch.tensor([[0., 0., 1., 0.], [0., .5, .5, 0.],
                                               [0., 0., 0., 0.]]),
                        supported=torch.tensor([True, True, False]))
        query = torch.tensor([[0., 1., 0., 0.], [0., .5, .5, 0.], [0., 0., 1., 0.]])
        report = _weighted_evidence_diagnostic(evidence, query, torch.tensor([2., 4., 1.]))
        assert report["all"]["supported_queries"] == 2
        assert report["all"]["intact_vote_damaged"] == 1
        assert report["foreground_mixed"]["queries"] == 1
        self.assertAlmostEqual(report["foreground_mixed"]["soft_compatibility_sum"], .5)
        self.assertAlmostEqual(report["all"]["supported_valid_mass"], 6.)
        self.assertAlmostEqual(report["all"]["mass_weighted_compatibility_sum"], 2.)

    def test_saved_journal_verifies_checksum_and_order(self):
        tmp_path = self.tmp_path
        artifact, row, _ = _saved_fixture(tmp_path)
        rows = _verify_saved_artifacts(tmp_path, ["image_a"], ["grid_probq"], ["t128_global"], 1)
        assert rows == [row]
        with self.assertRaisesRegex(ValueError, "order"):
            _verify_saved_artifacts(tmp_path, ["image_b"], ["grid_probq"], ["t128_global"], 1)
        artifact.write_bytes(artifact.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA"):
            _verify_saved_artifacts(tmp_path, ["image_a"], ["grid_probq"], ["t128_global"], 1)

    def _reject_invalid_artifact(self, bad_field, bad_value, match):
        tmp_path = self.tmp_path
        artifact, row, arrays = _saved_fixture(tmp_path)
        arrays[bad_field] = bad_value
        np.savez_compressed(artifact, **arrays)
        row["prediction_sha256"] = digest(artifact)
        (tmp_path / "images.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, match):
            _verify_saved_artifacts(tmp_path, ["image_a"], ["grid_probq"], ["t128_global"], 1)

    def test_failure_preserves_partial_count_phase_and_traceback(self):
        tmp_path = self.tmp_path
        summary = dict(image_count=17, complete=True, execution_complete=True)
        try:
            raise IndexError("failed pool")
        except IndexError as error:
            _record_failure(tmp_path, summary, "front_same_pass", "image_18", error)
        saved = json.loads((tmp_path / "summary.json").read_text())
        assert saved["image_count"] == 17
        assert saved["phase"] == "front_same_pass"
        assert saved["image"] == "image_18"
        assert saved["complete"] is saved["full_validation_evidence"] is False
        assert saved["execution_complete"] is False
        trace = (tmp_path / "traceback.txt").read_text()
        assert "Traceback" in trace and "IndexError: failed pool" in trace

    def test_synthetic_smoke_runs_one_front_and_freezes_neighbors_before_gt(self):
        tmp_path = self.tmp_path
        """Exercise real cosine/Q/vote/journal code with mocked external artifacts."""
        from . import core, shared
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1 import memory_audit
        from bda_benchmark.experiments.damageformer_srac_transformer_v1 import common

        calls = []
        torch.manual_seed(12)
        feature = torch.randn(1, 256, 1, 3)
        token = torch.randn(1, 3, 128)
        token[:, 2] = 0

        class Front(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Module()
                self.base.clf_2 = nn.Conv2d(256, 4, 1)

            def forward(self, _pre, _post):
                calls.append("front")
                logits = self.base.clf_2(feature)
                return dict(tokens=token, old_pixel=logits, prob=logits.softmax(1),
                            q=torch.full((1, 9, 3), 1 / 9), ids=torch.tensor([[0, 1, 1]]).expand(9, -1),
                            count=3, shape=(1, 3), valid_regions=torch.tensor([[True, True, False]]))

        paired = dict(t128=torch.randn(4, 128), f256=torch.randn(4, 256),
                      distribution=torch.eye(4), events=torch.ones(4, dtype=torch.long),
                      images=torch.arange(4), regions=torch.zeros(4, dtype=torch.long),
                      event_names=["val", "support"])
        original = dict(paired, tokens=paired["t128"], geometry=torch.zeros(4, 5),
                        disasters=torch.zeros(4, dtype=torch.long))
        names = [f"val_image_{i}" for i in range(450)]
        bundle = dict(args=SimpleNamespace(dataset="fixture", workers=0, seed=3407, pool_mode="native"),
                      protocol={"selected_images": names, "metrics": {}},
                      topk=8, val_names=names, device=torch.device("cpu"),
                      front=Front(), bank=original, metric=object(),
                      decoders={arm: object() for arm in ("j1", "j2", "j3")},
                      reference={}, protocol_sha256="fixture", hashes={"bank_sha256": "fixture"})
        key_root = tmp_path / "keys"
        key_root.mkdir()
        for name in ("paired_bank.pt", "manifest.json", "summary.json"):
            (key_root / name).write_bytes(b"external fixture")
        target = torch.tensor([[[1, 2, 255]]])
        batch = (torch.empty(1, 3, 1, 3), torch.empty(1, 3, 1, 3), None, target, [names[0]])
        self.patch(shared, "load_bundle", lambda _protocol, _output: bundle)
        self.patch(shared, "load_paired_bank", lambda *_a, **_k: (paired, {"smoke": True}))
        self.patch(shared, "verify_all", lambda *_a, **_k: {"fixture": True})
        self.patch(memory_audit, "full_image_loader", lambda *_a, **_k: [batch])
        self.patch(memory_audit, "parse_batch", lambda batch, _device: batch)
        self.patch(memory_audit, "event", lambda _name: "val")

        def historical_select(query, _bank, _event, _image, _topk, projector=None):
            calls.append("j3_ids" if projector is not None else "j2_ids")
            return torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]]).expand(len(query), -1).clone()

        def historical_decode(_decoder, arm, _feature, result, _j2, _j3):
            calls.append(arm)
            return {"logits": result["old_pixel"]}

        self.patch(memory_audit, "select_ids", historical_select)
        self.patch(memory_audit, "decoder_forward", historical_decode)
        real_select = core.select_neighbors
        real_region_targets = common.region_targets

        def checked_select(*args, **kwargs):
            assert "gt" not in calls
            calls.append("new_ids")
            return real_select(*args, **kwargs)

        def checked_gt(*args, **kwargs):
            assert calls.count("new_ids") == 8  # four TopK and four Top64
            calls.append("gt")
            return real_region_targets(*args, **kwargs)

        self.patch(core, "select_neighbors", checked_select)
        self.patch(common, "region_targets", checked_gt)
        output = tmp_path / "diagnostic"
        report = run(SimpleNamespace(protocol="fixture", keys=str(key_root), output=str(output),
                                     limit=1, temperature=.1, topk=8, policy="both"))
        assert calls[:6] == ["front", "j2_ids", "j3_ids", "j1", "j2", "j3"]
        assert calls.count("front") == 1
        assert report["execution_complete"] is True
        assert report["complete"] is report["full_validation_evidence"] is False
        assert report["smoke"] is True
        assert report["image_count"] == 1
        assert report["query_eligibility"]["inactive_both_keys"] == 1
        assert report["query_eligibility"]["unlabeled_posthoc_excluded"] == 1
        assert report["native_a1_replay"]["max_abs"] == 0
        assert report["pool_guard"]["restored"] is True
        with np.load(output / "predictions" / "0000.npz", allow_pickle=False) as saved:
            for key in ("t128", "f256"):
                for policy in ("global", "per_class"):
                    assert (saved[f"support_{key}_{policy}"][2] == -1).all()
                    assert (saved[f"evidence_{key}_{policy}"][2] == 0).all()

    def test_valid_checksum_rejects_illegal_prediction(self):
        self._reject_invalid_artifact('prediction_grid_probq', np.full((1, 2, 2), 4, dtype=np.uint8), 'class mismatch')

    def test_valid_checksum_rejects_illegal_support(self):
        self._reject_invalid_artifact('support_t128_global', np.array([[1, -1]], dtype=np.int32), 'IDs/scores invalid')

    def test_valid_checksum_rejects_missing_score_sentinel(self):
        self._reject_invalid_artifact('scores_t128_global', np.array([[.8, 0.]], dtype=np.float32), 'IDs/scores invalid')

    def test_valid_checksum_rejects_illegal_soft_label(self):
        self._reject_invalid_artifact('softlabels_t128_global', np.ones((1, 2, 4), dtype=np.float32), 'soft labels invalid')

    def test_valid_checksum_rejects_zero_supported_evidence(self):
        self._reject_invalid_artifact('evidence_t128_global', np.zeros((1, 4), dtype=np.float32), 'evidence values invalid')

    def test_valid_checksum_rejects_nonfinite_coverage(self):
        self._reject_invalid_artifact('supported_mass_t128_global', np.full((1, 2, 2), np.nan), 'evidence values invalid')


if __name__ == '__main__':
    unittest.main()
