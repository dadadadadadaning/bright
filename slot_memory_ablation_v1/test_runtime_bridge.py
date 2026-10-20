"""Stdlib-only source binding tests; no CUDA or model weight loading."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

import runtime_bridge as bridge
import ablation_runner as runner


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    package = root / "extended"
    original = package / "deps/original"
    for rel in (
        "fresh_chain.py", "runtime_adapter.py", "io_utils.py", "deps/study/legacy.py",
        "deps/original/model.py", "deps/original/train.py", "deps/original/slots.py",
    ):
        p = package / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# fixture only\n")
    for rel in ("deps/study/SHA256SUMS.json", "deps/helpers/SHA256SUMS.json"):
        dump(package / rel, {})
    dump(package / "SHA256SUMS.json", {
        str(p.relative_to(package)): bridge.digest(p)
        for p in package.rglob("*") if p.is_file()
    })
    run = root / "source_run"
    repo = root / "repo"
    for name in ("core.py", "paired.py", "shared.py"):
        p = repo / "bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# fixture\n")
    dataset = root / "dataset"
    dataset.mkdir()
    checkpoint = root / "source.pth"
    checkpoint.write_bytes(b"not deserialized")
    config = dict(
        repo=str(repo), protocol=str(run / "source.json"), membership_root=str(run / "members"),
        bank_root=str(run / "members"), values=str(run / "values"), device="cuda:0",
        workers=1, seed=3407, key_name="t128", topk=64, chunk_size=32, lr=1e-4,
        weight_decay=.005, steps=5000, eval_every=500,
    )
    for folder, names in (
        ("members", ("paired_bank.pt", "manifest.json", "summary.json", "protocol.json", "images.jsonl")),
        ("values", ("slot_bank.pt", "manifest.json", "summary.json", "images.jsonl", "input_fingerprints.json")),
    ):
        for name in names:
            p = run / folder / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        dump(run / folder / "summary.json", dict(complete=True, smoke=False))
    dump(run / "source.json", dict(
        format="BRIGHT_SURVIVING_Q_SOURCE_V2", release_sha256=bridge.digest(package / "SHA256SUMS.json"),
        input_paths={"q_checkpoint": str(checkpoint)}, dataset=str(dataset),
    ))
    dump(run / "train/manifest.json", dict(
        arguments={"tool_dir": str(original)}, runtime_config=config, source_sha256="fixture",
        release_sha256=bridge.digest(package / "deps/study/SHA256SUMS.json"),
    ))
    return run, package


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.package = fixture(self.root)

    def test_distinct_outer_and_study_scopes_are_valid(self):
        result = bridge.inspect_source(self.run)
        self.assertNotEqual(result["outer_release_sha256"], result["study_release_sha256"])
        self.assertFalse(result["weights_loaded"])
        self.assertFalse(result["cuda_forward_verified"])
        self.assertEqual(result["package"], str(self.package))
        bridge.verify_ledgers(result)

    def test_changed_outer_code_fails(self):
        (self.package / "fresh_chain.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            bridge.inspect_source(self.run)

    def test_unbound_python_fails(self):
        (self.package / "extra.py").write_text("pass")
        with self.assertRaisesRegex(ValueError, "Unbound"):
            bridge.inspect_source(self.run)

    def test_changed_source_scope_fails(self):
        source = bridge.read_json(self.run / "source.json")
        source["release_sha256"] = bridge.digest(self.package / "deps/study/SHA256SUMS.json")
        dump(self.run / "source.json", source)
        with self.assertRaisesRegex(ValueError, "actual outer package"):
            bridge.inspect_source(self.run)

    def test_source_requires_formal_bank_even_for_budget_smoke(self):
        dump(self.run / "members/summary.json", dict(complete=True, smoke=True))
        with self.assertRaisesRegex(ValueError, "FORMAL"):
            bridge.inspect_source(self.run)

    def test_required_files_collected(self):
        (self.run / "members/paired_bank.pt").unlink()
        (self.run / "values/slot_bank.pt").unlink()
        with self.assertRaises(FileNotFoundError) as caught:
            bridge.inspect_source(self.run)
        self.assertIn("paired_bank.pt", str(caught.exception))
        self.assertIn("slot_bank.pt", str(caught.exception))

    def test_recreated_package_cannot_replace_recorded_code(self):
        with self.assertRaisesRegex(ValueError, "recorded|record|manifest"):
            bridge.inspect_source(self.run, self.root / "another")

    def test_ledger_changes_after_plan_fail(self):
        result = bridge.inspect_source(self.run)
        dump(self.run / "members/manifest.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "changed after plan"):
            bridge.verify_ledgers(result)

    def test_declared_steps_and_groups(self):
        args = argparse.Namespace(
            arms=None, group="main", steps=2500, post500_lr=.2,
            seeds="3407,3408,3409", fixed_train=450, smoke_steps=3,
        )
        arms, seeds = runner.options(args)
        self.assertEqual(arms, ("e", "f", "b", "d", "r", "l"))
        self.assertEqual(seeds, (3407, 3408, 3409))
        self.assertEqual(runner.schedule_steps(2500), [500, 1000, 1500, 2000, 2500])
        args.seeds = "3407,3407"
        with self.assertRaises(ValueError):
            runner.options(args)

    def test_finite_train_loader_repeats_with_seed_plus_epoch(self):
        calls = []

        class Runtime:
            def batches(self, names, train, seed):
                calls.append((tuple(names), train, seed))
                return iter([(seed, "first"), (seed, "second")])

        iterator = runner.training_batches(Runtime(), ["train_a", "train_b"], 3407)
        observed = [next(iterator) for _ in range(5)]
        self.assertEqual(observed, [
            (0, (3407, "first")), (0, (3407, "second")),
            (1, (3408, "first")), (1, (3408, "second")),
            (2, (3409, "first")),
        ])
        self.assertEqual(calls, [(("train_a", "train_b"), True, s) for s in (3407, 3408, 3409)])

    def test_empty_training_epoch_fails(self):
        class Runtime:
            def batches(self, names, train, seed):
                return iter([])

        with self.assertRaisesRegex(RuntimeError, "no samples"):
            next(runner.training_batches(Runtime(), ["train_a"], 3407))


if __name__ == "__main__":
    unittest.main()
