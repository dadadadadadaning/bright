"""Artifact provenance and single-image scope; no BRIGHT images are read."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
from .recheck_probability import checked_failure, run


class ProbabilityRecheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.failed = self.root / "fit" / "initial_val"
        self.failed.mkdir(parents=True)
        self.names = ["event_" + str(i) for i in range(450)]
        self.protocol = self.root / "protocol.json"
        self.e0 = dict(
            format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True,
            full_validation_evidence=True, image_count=450,
            scope="all", profile="legacy", pool_mode="native", smoke=False,
            reproduction=dict(passed=True), input_files_unchanged=True,
            code_unchanged=True, frozen_evidence_unchanged=True, numerical_unchanged=True,
            selected_images=self.names)
        self.save(self.protocol, self.e0)
        self.manifest = dict(format="SRAC_REGION_HEAD_ADAPT_V1",
                             source_sha256="a" * 64, protocol_sha256=digest(self.protocol))
        self.manifest_path = self.failed.parent / "manifest.json"
        self.save(self.manifest_path, self.manifest)
        self.summary = dict(format="SRAC_REGION_HEAD_ADAPT_EVAL_V1", complete=False,
                            full_validation_evidence=False, initial=True,
                            image=self.names[72], image_count=72,
                            error="ValueError: Probabilities are not normalized")
        self.summary_path = self.failed / "summary.json"
        self.save(self.summary_path, self.summary)
        self.output = self.root / "recheck"

    @staticmethod
    def save(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def context(self, **kwargs):
        return checked_failure(kwargs.get("failed", self.failed),
                               kwargs.get("protocol", self.protocol),
                               kwargs.get("output", self.output))

    def test_accepts_only_failed_initial_probability_artifact_and_exact_image(self):
        origin = self.context()
        self.assertEqual(origin["image"], self.names[72])
        self.assertEqual(origin["summary_sha256"], digest(self.summary_path))
        self.assertEqual(self.context(failed=self.summary_path), origin)
        for key, wrong in (("complete", True), ("initial", False),
                           ("full_validation_evidence", True), ("image", ""),
                           ("format", "another"), ("error", "CUDA out of memory")):
            with self.subTest(key=key):
                self.save(self.summary_path, {**self.summary, key: wrong})
                with self.assertRaises(ValueError):
                    self.context()
        self.save(self.summary_path, {**self.summary,
                                     "error": "DiagnosticProbabilityError: invalid sum"})
        self.assertEqual(self.context()["image"], self.names[72])

    def test_rejects_wrong_protocol_hash_unpassed_e0_and_unknown_image(self):
        self.save(self.manifest_path, {**self.manifest, "protocol_sha256": "b" * 64})
        with self.assertRaisesRegex(ValueError, "E0 SHA256"):
            self.context()
        self.save(self.manifest_path, self.manifest)
        self.save(self.summary_path, {**self.summary, "image": "unknown"})
        with self.assertRaisesRegex(ValueError, "not in"):
            self.context()
        self.save(self.summary_path, self.summary)
        self.save(self.protocol, {**self.e0, "profile": "strict"})
        with self.assertRaisesRegex(ValueError, "passed legacy/native"):
            self.context()

    def test_rejects_wrong_path_missing_manifest_or_output_inside_old_fit(self):
        with self.assertRaisesRegex(ValueError, "initial_val"):
            self.context(failed=self.failed.parent)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.context(output=self.failed.parent / "new_diagnostic")
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.context()
        self.manifest_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.context()

    def mocked_run(self, effect=None):
        package = "bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1"
        bundle = dict(protocol_sha256=digest(self.protocol), val_names=self.names,
                      front=object(), device="cpu", input_paths={}, inputs_before={})
        heads = mock.Mock()
        heads.to.return_value = heads

        def evaluate(actual_bundle, actual_heads, **kwargs):
            self.assertIs(actual_bundle, bundle)
            self.assertIs(actual_heads, heads)
            self.assertEqual(kwargs, dict(output=self.output.resolve(), limit=0,
                                          image=self.names[72], initial=True))
            kwargs["output"].mkdir()
            for key, path in bundle["input_paths"].items():
                self.assertEqual(digest(path), bundle["inputs_before"][key])
            if effect:
                return effect()
            return dict(complete=True, smoke=True, full_validation_evidence=False, image_count=1)

        with ExitStack() as stack:
            load = stack.enter_context(mock.patch(package + ".core.load_source", return_value=bundle))
            make = stack.enter_context(mock.patch(package + ".core.make_heads", return_value=heads))
            stack.enter_context(mock.patch(package + ".core.source_hash", return_value="c" * 64))
            ev = stack.enter_context(mock.patch(package + ".evaluate.evaluate_heads", side_effect=evaluate))
            result = run(argparse.Namespace(failed=str(self.failed), protocol=str(self.protocol),
                                            output=str(self.output)))
            load.assert_called_once_with(self.protocol.resolve(), self.output.resolve())
            make.assert_called_once_with(bundle["front"])
            ev.assert_called_once()
        return result, bundle

    def test_single_image_call_binds_evidence_and_records_changed_code_readonly(self):
        before = {p: p.read_bytes() for p in (self.summary_path, self.manifest_path, self.protocol)}
        result, bundle = self.mocked_run()
        self.assertFalse(result["full_validation_evidence"])
        metadata = json.loads((self.output / "recheck_origin.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["diagnostic_source_changed"])
        self.assertFalse(metadata["training"])
        self.assertEqual(metadata["old_evidence_unchanged"], dict(summary=True, manifest=True))
        self.assertEqual(metadata["old_source_sha256"], "a" * 64)
        self.assertEqual(metadata["current_source_sha256"], "c" * 64)
        self.assertEqual(len(bundle["input_paths"]), 2)
        self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))

    def test_failure_preserves_origin_and_failed_record_without_training(self):
        before = self.summary_path.read_bytes()
        def fails():
            raise ValueError("Probabilities are not normalized")
        with self.assertRaisesRegex(ValueError, "not normalized"):
            self.mocked_run(fails)
        self.assertEqual(self.summary_path.read_bytes(), before)
        metadata = json.loads((self.output / "recheck_origin.json").read_text(encoding="utf-8"))
        self.assertTrue(all(metadata["old_evidence_unchanged"].values()))

    def test_rejects_formal_claim_or_mutated_old_evidence(self):
        with self.assertRaisesRegex(ValueError, "nonformal"):
            self.mocked_run(lambda: dict(full_validation_evidence=True, smoke=False, image_count=450))
        # A separate fixture is used because fresh outputs cannot be reused.
        self.output = self.root / "second_recheck"
        def tamper():
            self.save(self.summary_path, {**self.summary, "changed": True})
            return dict(full_validation_evidence=False, smoke=True, image_count=1)
        with self.assertRaisesRegex(RuntimeError, "Old failed evidence changed"):
            self.mocked_run(tamper)


if __name__ == "__main__":
    unittest.main()
