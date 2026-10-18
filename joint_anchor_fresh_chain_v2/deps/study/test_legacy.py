"""CPU/file contracts for dependency discovery; no dataset, CUDA, or training."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

try:
    from . import legacy
except ImportError:
    import legacy


class LegacyContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return path

    def config(self):
        return {key: str(self.root / key) for key in legacy._FRESH_REQUIRED}

    def test_missing_manifest_rejected_before_import(self):
        with self.assertRaises(legacy.MissingInputsError) as exc:
            legacy.load_legacy(self.root)
        self.assertIn(str(self.root / "SHA256SUMS.json"), exc.exception.missing)

    def test_unpinned_manifest_rejected_before_module_execution(self):
        self.write("study.py", "raise AssertionError('must not execute')")
        self.write("SHA256SUMS.json", {"study.py": legacy._digest(self.root / "study.py")})
        with self.assertRaisesRegex(ValueError, "Pinned dependency manifest mismatch"):
            legacy.load_legacy(self.root)

    def test_manifest_rejects_escape_and_unbound_python(self):
        self.write("SHA256SUMS.json", {"../escape.py": "0" * 64})
        with self.assertRaisesRegex(ValueError, "Invalid or missing package"):
            legacy._package_files(self.root)
        self.write("known.py", "value=1")
        self.write("SHA256SUMS.json", {"known.py": legacy._digest(self.root / "known.py")})
        self.write("unexpected.py", "value=2")
        with self.assertRaisesRegex(ValueError, "Unbound Python"):
            legacy._package_files(self.root)

    def test_manifest_detects_modified_dependency(self):
        self.write("known.py", "value=1")
        self.write("SHA256SUMS.json", {"known.py": legacy._digest(self.root / "known.py")})
        files = legacy._package_files(self.root)
        self.write("known.py", "value=2")
        with self.assertRaisesRegex(ValueError, "Pinned helper changed"):
            legacy.verify_dependencies(SimpleNamespace(files=files))

    def test_private_import_does_not_install_bare_modules(self):
        path = self.write("continuation.py", "value=7")
        before = sys.modules.get("continuation")
        name = "_bright_anchor_test_" + legacy._digest(path)[:8]
        try:
            loaded = legacy._load_named(path, name)
            self.assertEqual(loaded.value, 7)
            self.assertIs(sys.modules.get("continuation"), before)
        finally:
            sys.modules.pop(name, None)

    def test_fresh_config_has_explicit_no_resume_semantics(self):
        path = self.write("config.json", self.config())
        config = legacy._fresh_config(path)
        self.assertEqual(config["steps"], 5000)
        self.assertEqual(config["eval_every"], 500)
        self.assertEqual(config["lr"], 1e-4)
        self.assertEqual(config["key_name"], "t128")
        self.assertFalse(config["allow_smoke_values"])
        self.assertIsNone(config["baseline_root"])
        with self.assertRaisesRegex(ValueError, "requires an explicit"):
            legacy._fresh_config(None)

    def test_fresh_config_rejects_hidden_extra_changes(self):
        for change in ({"steps": 1500}, {"key_name": "f256"}, {"topk": 128},
                       {"lr": 2e-5}, {"seed": True}, {"teacher": "test"},
                       {"allow_smoke_values": True}, {"device": "cpu"}):
            with self.subTest(change=change):
                config = dict(self.config(), **change)
                path = self.write("config.json", config)
                with self.assertRaises(ValueError):
                    legacy._fresh_config(path)

    def test_missing_source_and_both_banks_aggregate(self):
        config = self.config()
        source = dict(q_checkpoint=str(self.root / "formal/train_baseline/last.pth"),
                      a1_checkpoint=str(self.root / "a1.pth"),
                      e0_metadata_protocol=str(self.root / "e0.json"),
                      evidence_level="training_final_val450")
        Path(config["protocol"]).write_text(json.dumps(source))
        missing = legacy.missing_inputs(config)
        for expected in (source["q_checkpoint"], source["a1_checkpoint"], source["e0_metadata_protocol"],
                         str(self.root / "formal/train_baseline/summary.json"),
                         str(self.root / "formal/train_baseline/manifest.json"),
                         str(self.root / "formal/train_baseline/validation.jsonl"),
                         str(self.root / "bank_root/paired_bank.pt"),
                         str(self.root / "values/slot_bank.pt")):
            self.assertIn(expected, missing)
        self.assertEqual(missing, sorted(set(missing)))

    def test_independent_q_evidence_is_discovered(self):
        config = self.config()
        source = dict(q_checkpoint=str(self.root / "formal/train_baseline/last.pth"),
                      a1_checkpoint=str(self.root / "a1.pth"),
                      e0_metadata_protocol=str(self.root / "e0.json"),
                      evidence_level="independent_val450")
        Path(config["protocol"]).write_text(json.dumps(source))
        missing = legacy.missing_inputs(config)
        self.assertIn(str(self.root / "formal/val_baseline/result.json"), missing)
        self.assertNotIn(str(self.root / "formal/train_baseline/validation.jsonl"), missing)

    def test_metadata_does_not_require_deleted_e0_weights(self):
        config = self.config()
        source = dict(q_checkpoint=str(self.root / "formal/train_baseline/last.pth"),
                      a1_checkpoint=str(self.root / "a1.pth"),
                      e0_metadata_protocol=str(self.root / "e0.json"),
                      evidence_level="training_final_val450")
        Path(config["protocol"]).write_text(json.dumps(source))
        self.write("e0.json", {"arguments": {"dataset": str(self.root / "dataset")},
                              "input_paths": {"train_list": str(self.root / "train.txt"),
                                              "val_list": str(self.root / "val.txt"),
                                              "front_snapshot": str(self.root / "DELETED_E0.pth")}})
        missing = legacy.missing_inputs(config)
        self.assertIn(str(self.root / "train.txt"), missing)
        self.assertIn(str(self.root / "val.txt"), missing)
        self.assertIn(str(self.root / "dataset"), missing)
        self.assertNotIn(str(self.root / "DELETED_E0.pth"), missing)

    def test_child_and_parent_ledgers_discovered(self):
        config = self.config()
        self.write("bank_root/manifest.json", {"source_image_fingerprints": {
            str(self.root / "source_image.tif"): "0" * 64},
            "archived_membership_root": str(self.root / "archive"),
            "archived_files_sha256": {"old.pt": "0" * 64}})
        self.write("values/manifest.json", {"file_sha256": {"extra.pt": "0" * 64}})
        self.write("values/input_fingerprints.json", {str(self.root / "child_source.tif"): "0" * 64})
        missing = legacy.missing_inputs(config)
        for path in ("source_image.tif", "archive/old.pt", "values/extra.pt", "child_source.tif"):
            self.assertIn(str(self.root / path), missing)

    def test_child_ledger_escape_rejected(self):
        config = self.config()
        self.write("values/manifest.json", {"file_sha256": {"../escape.pt": "0" * 64}})
        with self.assertRaisesRegex(ValueError, "Unsafe artifact"):
            legacy.missing_inputs(config)

    def test_prepare_checkpoint_does_not_fall_back_to_fresh(self):
        args = SimpleNamespace(start="checkpoint500", tool_dir=str(self.root / "tool"),
                               train_root=str(self.root / "old"))
        deps = SimpleNamespace(files={})
        with self.assertRaises(legacy.MissingInputsError) as exc:
            legacy.prepare_source(args, deps)
        self.assertIn(str(self.root / "old/train/checkpoints/checkpoint_000500.pt"), exc.exception.missing)
        self.assertIn(str(self.root / "old/train/summary.json"), exc.exception.missing)

    def test_prepare_checkpoint_collects_runtime_missing_with_reports(self):
        config = self.config()
        self.write("old/train/manifest.json", {"arguments": config})
        args = SimpleNamespace(start="checkpoint500", tool_dir=str(self.root / "tool"),
                               train_root=str(self.root / "old"))
        with self.assertRaises(legacy.MissingInputsError) as exc:
            legacy.prepare_source(args, SimpleNamespace(files={}))
        self.assertIn(str(self.root / "old/train/summary.json"), exc.exception.missing)
        self.assertIn(str(self.root / "values/slot_bank.pt"), exc.exception.missing)

    def test_prepare_fresh_needs_no_historical_summary(self):
        path = self.write("config.json", self.config())
        args = SimpleNamespace(start="fresh", tool_dir=str(self.root / "tool"), config=str(path))
        with self.assertRaises(legacy.MissingInputsError) as exc:
            legacy.prepare_source(args, SimpleNamespace(files={}))
        self.assertFalse(any("checkpoint_000500" in x or "/old/" in x for x in exc.exception.missing))
        self.assertIn(str(self.root / "values/slot_bank.pt"), exc.exception.missing)

    def test_fresh_never_loads_old_checkpoint(self):
        class NoLoad:
            def load(self, *args, **kwargs):
                raise AssertionError("fresh must not load old weights")
        self.assertIsNone(legacy.load_checkpoint({"start": "fresh"}, None, NoLoad()))


if __name__ == "__main__":
    unittest.main()
