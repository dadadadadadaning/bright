"""CPU contracts for sample identity and checkpoint provenance, no GPU/data."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from probe import candidate_from_journal, validate_checkpoint, validate_live_protocol


FORMAT = "BRIGHT_SLOT_MEMORY_ABLATION_V1"
ARMS = ("e", "f", "b", "d", "r", "l")
SEED = 3407


def fingerprint(state):
    parts = {
        name: {"dtype": str(t.dtype), "shape": list(t.shape), "values": t.tolist()}
        for name, t in state.items()
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


class CandidateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "train.jsonl"
        self.names = ["synthetic_train_%06d" % i for i in range(2918)]
        self.plan = {"format": FORMAT, "arms": list(ARMS), "seeds": [SEED]}
        self.manifest = {"seed": SEED, "train_names": self.names}
        loader = torch.utils.data.DataLoader(
            self.names, batch_size=1, shuffle=True, num_workers=0,
            drop_last=False, generator=torch.Generator().manual_seed(SEED),
        )
        self.order = [batch[0] for batch in loader]

    def row(self, step):
        return {
            "step": step, "image": self.order[step - 1],
            "event": "synthetic_train", "epoch": 0, "epoch_seed": SEED,
            "arms": {a: {"optimizer_step": step} for a in ARMS},
        }

    def journal(self, rows, suffix=""):
        self.path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows) + suffix,
            encoding="utf-8",
        )
        return self.path

    def candidate(self):
        return candidate_from_journal(torch, self.manifest, self.plan, self.path)

    def test_every_recorded_image_matches_before_selecting_next_sample(self):
        self.journal([self.row(i) for i in range(1, 1397)])
        before = self.path.read_bytes()
        result = self.candidate()
        self.assertEqual(result["last_complete_step"], 1396)
        self.assertEqual(result["candidate_step"], 1397)
        self.assertEqual(result["verified_prefix_count"], 1396)
        self.assertEqual(result["image"], self.order[1396])
        self.assertEqual(result["epoch"], 0)
        self.assertEqual(result["epoch_seed"], SEED)
        self.assertEqual(self.path.read_bytes(), before)

    def test_wrong_earlier_image_is_rejected_even_when_last_record_matches(self):
        rows = [self.row(1), self.row(2), self.row(3)]
        rows[0]["image"] = "another_image"
        self.journal(rows)
        with self.assertRaises(ValueError):
            self.candidate()

    def test_sampler_requires_dataloader_base_seed_consumption(self):
        # Bare randperm starts from a different RNG state than DataLoader.
        plain = torch.randperm(
            len(self.names), generator=torch.Generator().manual_seed(SEED)
        ).tolist()
        self.assertNotEqual(self.order[:3], [self.names[i] for i in plain[:3]])
        wrong = self.row(1)
        wrong["image"] = self.names[plain[0]]
        self.journal([wrong])
        with self.assertRaises(ValueError):
            self.candidate()

    def test_truncated_or_unterminated_last_record_is_rejected(self):
        for suffix in ('{"step": 3', json.dumps(self.row(3))):
            with self.subTest(suffix=suffix[:20]):
                self.journal([self.row(1), self.row(2)], suffix)
                with self.assertRaises(ValueError):
                    self.candidate()

    def test_malformed_terminated_line_is_rejected(self):
        self.journal([self.row(1)], "not JSON\n")
        with self.assertRaises(ValueError):
            self.candidate()

    def test_gap_duplicate_or_incomplete_arm_update_is_rejected(self):
        cases = [
            [self.row(1), self.row(3)],
            [self.row(1), self.row(1)],
            [self.row(1), self.row(2)],
        ]
        cases[-1][-1]["arms"].pop("l")
        for rows in cases:
            with self.subTest(rows=[row["step"] for row in rows]):
                self.journal(rows)
                with self.assertRaises(ValueError):
                    self.candidate()

    def test_nonzero_epoch_and_wrong_epoch_seed_are_rejected(self):
        for field, value in (("epoch", 1), ("epoch_seed", SEED + 1)):
            with self.subTest(field=field):
                rows = [self.row(1), self.row(2)]
                rows[0][field] = value
                self.journal(rows)
                with self.assertRaises(ValueError):
                    self.candidate()


class CheckpointTests(unittest.TestCase):
    def fixture(self):
        plan = {
            "format": FORMAT, "arms": list(ARMS), "seeds": [SEED],
            "source": {"source_run": "/synthetic/source"},
        }
        manifest = {
            "seed": SEED,
            "runtime_config": {"device": "cuda:0", "seed": SEED, "chunk_size": 32},
            "protocol_info": {"source_protocol_sha256": "a" * 64},
        }
        heads = {a: {"weight": torch.tensor([float(i), 2.0])} for i, a in enumerate(ARMS)}
        saved = {
            "format": FORMAT, "step": 1000, "study_plan": copy.deepcopy(plan),
            "heads": heads,
            "runtime_config": copy.deepcopy(manifest["runtime_config"]),
            "protocol_info": copy.deepcopy(manifest["protocol_info"]),
        }
        sidecar = {
            "step": 1000, "roundtrip_verified": True,
            "head_hashes": {a: fingerprint(heads[a]) for a in ARMS},
        }
        return saved, sidecar, plan, manifest

    def test_matching_checkpoint_has_all_expected_provenance(self):
        saved, sidecar, plan, manifest = self.fixture()
        before = {a: fingerprint(s) for a, s in saved["heads"].items()}
        self.assertIsNone(validate_checkpoint(saved, sidecar, plan, manifest, fingerprint))
        self.assertEqual(before, {a: fingerprint(s) for a, s in saved["heads"].items()})

    def test_json_roundtrip_list_tuple_equivalence_preserves_identity(self):
        saved, sidecar, plan, manifest = self.fixture()
        saved["study_plan"]["arms"] = tuple(ARMS)
        saved["protocol_info"]["shape"] = (1, 4, 1024, 1024)
        manifest["protocol_info"]["shape"] = [1, 4, 1024, 1024]
        self.assertIsNone(validate_checkpoint(saved, sidecar, plan, manifest, fingerprint))

    def test_wrong_format_step_plan_runtime_or_protocol_is_rejected(self):
        for field in ("format", "step", "study_plan", "runtime_config", "protocol_info"):
            with self.subTest(field=field):
                saved, sidecar, plan, manifest = self.fixture()
                if field == "format":
                    saved[field] = "BRIGHT_JOINT_ANCHOR_EXTENDED_V2"
                elif field == "step":
                    saved[field] = 500
                elif field == "study_plan":
                    saved[field]["source"]["source_run"] = "/different/source"
                elif field == "runtime_config":
                    saved[field]["seed"] = SEED + 1
                else:
                    saved[field]["source_protocol_sha256"] = "b" * 64
                with self.assertRaises(ValueError):
                    validate_checkpoint(saved, sidecar, plan, manifest, fingerprint)

    def test_missing_or_extra_head_is_rejected(self):
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                saved, sidecar, plan, manifest = self.fixture()
                if mutation == "missing":
                    saved["heads"].pop("l")
                else:
                    saved["heads"]["a"] = {"weight": torch.ones(2)}
                with self.assertRaises(ValueError):
                    validate_checkpoint(saved, sidecar, plan, manifest, fingerprint)

    def test_modified_head_or_sidecar_hash_is_rejected(self):
        for mutation in ("tensor", "hash"):
            with self.subTest(mutation=mutation):
                saved, sidecar, plan, manifest = self.fixture()
                if mutation == "tensor":
                    saved["heads"]["d"]["weight"][0] += 1
                else:
                    sidecar["head_hashes"]["d"] = "0" * 64
                with self.assertRaises(ValueError):
                    validate_checkpoint(saved, sidecar, plan, manifest, fingerprint)


class LiveProtocolTests(unittest.TestCase):
    def fixture(self):
        recorded = {
            "source_protocol_sha256": "a" * 64,
            "execution_settings": {
                "torch_version": "2.2.1+cu121", "cuda_version": "12.1",
                "cuda_launch_blocking": None, "matmul_allow_tf32": False,
                "deterministic_algorithms": True,
            },
        }
        current = copy.deepcopy(recorded)
        current["execution_settings"]["cuda_launch_blocking"] = "1"
        return current, recorded

    def test_only_intended_blocking_change_is_allowed_without_mutating_evidence(self):
        current, recorded = self.fixture()
        before = copy.deepcopy((current, recorded))
        change = validate_live_protocol(current, recorded)
        self.assertEqual(change, {
            "setting": "CUDA_LAUNCH_BLOCKING", "recorded": None, "diagnostic": "1",
        })
        self.assertEqual((current, recorded), before)

    def test_other_policy_source_changes_or_disabled_blocking_are_rejected(self):
        for mutation in ("tf32", "source", "blocking"):
            with self.subTest(mutation=mutation):
                current, recorded = self.fixture()
                if mutation == "tf32":
                    current["execution_settings"]["matmul_allow_tf32"] = True
                elif mutation == "source":
                    current["source_protocol_sha256"] = "b" * 64
                else:
                    current["execution_settings"]["cuda_launch_blocking"] = None
                with self.assertRaises(ValueError):
                    validate_live_protocol(current, recorded)


if __name__ == "__main__":
    unittest.main()
