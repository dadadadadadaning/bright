"""Synthetic archived-evidence contracts; no real BRIGHT data/model/CUDA.

Each fixture uses the formal 450 journal entries, but maps are only 2x3.
Tamper tests reseal ordinary hashes where necessary so semantic gates, rather
than a coincidental stale checksum, must reject the incorrect evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import compare_routes as compare


ARMS = ("unsplit", "capacity_control", "hard", "soft")
HISTORICAL = ("a1_pixel", "direct_region", "j1", "j2", "j3", "q_a1_pixel")
FORMAT = "SRAC_SAR_SOFT_ROUTE_V1"
PROTOCOL = "1" * 64
SOURCE = "2" * 64


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, sort_keys=True, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def confusion(gt, prediction):
    valid = gt != 255
    return np.bincount(4 * gt[valid].astype(np.int64) + prediction[valid], minlength=16).reshape(4, 4)


def metric(cm):
    cm = np.asarray(cm, dtype=np.int64)
    true_positive = np.diag(cm).astype(np.float64)
    row, col = cm.sum(1), cm.sum(0)
    union = row + col - true_positive
    def ratios(denominator):
        return [float(n/d) if d else None for n, d in zip(true_positive, denominator)]
    p, r = ratios(col), ratios(row)
    present = [value for value in ratios(union)[1:] if value is not None]
    return {"count": int(cm.sum()), "confusion": cm.tolist(), "iou": ratios(union),
            "precision": p, "recall": r,
            "f1": [float(2*n/d) if d else None for n, d in zip(true_positive, row+col)],
            "support": row.tolist(), "background_to_building": int(cm[0, 1:].sum()),
            "building_to_background": int(cm[1:, 0].sum()),
            "present_foreground_miou": sum(present)/len(present) if present else None,
            "valid_miou": float((true_positive / np.maximum(union, 1))[1:].mean())}


def fragment(result):
    return {key: copy.deepcopy(result[key]) for key in ("complete", "full_validation_evidence",
            "reproduction", "e0_metric_comparison", "q_a1_reproduction", "metrics")}


class EvidenceFixture:
    def __init__(self, root, *, absent_damage=False, different_sar_targets=False):
        self.root = Path(root)
        names = [("event-a" if index < 225 else "event-b") + f"_{index:08d}" for index in range(450)]
        write_json(self.root / "protocol.json", {"selected_images": names, "arguments": {"seed": 3407}})
        self.protocol_sha = digest(self.root / "protocol.json")
        self.certificate = self.root / "sar_certificate.json"
        write_json(self.certificate, {"format": "BRIGHT_EXTERNAL_SAR_VALIDATION_V1", "complete": True,
                                      "training": False, "test_accessed": False})
        self.fits, self.evals = {}, {}
        self.different_sar_targets = different_sar_targets
        self.compare_count = 0
        for name, source_arm in (("baseline", "baseline"), ("sar", "sar_cue")):
            fit, evaluation = self.root / (name + "_fit"), self.root / (name + "_eval")
            fit.mkdir()
            self.fits[name], self.evals[name] = fit, evaluation
            q_checkpoint = self.root / (name + "_q") / "last.pth"
            q_checkpoint.parent.mkdir()
            q_checkpoint.write_bytes(("synthetic Q " + name).encode())
            q_source = {"kind": "completed_" + source_arm + "_q", "arm": source_arm,
                        "checkpoint": str(q_checkpoint.resolve()), "sha256": digest(q_checkpoint),
                        "steps": 2000, "protocol_sha256": self.protocol_sha,
                        "train_source_hash": "3" * 64, "inherited_source_hash": "4" * 64,
                        "cue_gain": .1, "route_seed": 3438,
                        "initial_readout_fingerprints": {"embedding": "5" * 64, "region_classifier": "6" * 64},
                        "training_hyperparameters": {"steps": 2000, "eval_every": 2000, "batch": 2,
                            "accum": 8, "workers": 4, "lr": 1e-4, "aux_weight": .1,
                            "cue_gain": .1, "amp": "bf16"}}
            (fit / "heads_last.pth").write_bytes(("synthetic trained readouts " + name).encode())
            stream = hashlib.sha256()
            rows = []
            for step in range(1, 2001):
                image = "train-event_" + str(step % 11)
                stream.update((str(step) + "\t" + image + "\n").encode())
                rows.append({"step": step, "image": image, "event": "train-event", "epoch": step // 11,
                             "lr": 1e-4, "heads": {arm: {"loss": .5, "grad_norm": .1} for arm in ARMS}})
            (fit / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            parameters = {arm: 1000 for arm in ARMS}
            manifest = {"format": FORMAT, "arguments": {"protocol": str(self.root / "protocol.json"),
                        "q_checkpoint": str(q_checkpoint), "output": str(fit), "steps": 2000,
                        "eval_every": 2000, "lr": 1e-4, "threshold": .5, "soft_floor": .05,
                        "smoke": False}, "q_source": q_source, "source_sha256": SOURCE,
                        "protocol_sha256": self.protocol_sha, "original_front_sha256": "7" * 64,
                        "original_a1_sha256": "8" * 64, "train_list_sha256": "9" * 64,
                        "val_list_sha256": "a" * 64, "trainable_parameters": parameters,
                        "source_model_state_sha256": digest(q_checkpoint), "test_accessed": False, "smoke": False}
            write_json(fit / "manifest.json", manifest)
            initial = self.make_evaluation(fit / "initial_val", name, q_source, initial=True,
                                           absent_damage=absent_damage)
            final = self.make_evaluation(fit / "final_val", name, q_source, initial=False,
                                         absent_damage=absent_damage)
            summary = {"format": FORMAT, "complete": True, "full_validation_evidence": True,
                       "test_accessed": False, "training": False, "smoke": False,
                       "steps": 2000, "updates_per_arm": {arm: 2000 for arm in ARMS},
                       "threshold": .5, "soft_floor": .05, "source_sha256": SOURCE,
                       "protocol_sha256": self.protocol_sha, "q_source": q_source,
                       "checkpoint": "heads_last.pth", "checkpoint_sha256": digest(fit / "heads_last.pth"),
                       "training_journal_sha256": digest(fit / "training.jsonl"),
                       "train_stream_sha256": stream.hexdigest(), "trainable_parameters": parameters,
                       "immutable": {"files": True, "runtime": True, "route_code": True,
                                     "route_q_source": True, "manifest": True},
                       "initial": fragment(initial), "final": fragment(final), "phase": "complete"}
            write_json(fit / "summary.json", summary)
            self.make_evaluation(evaluation, name, q_source, initial=False, absent_damage=absent_damage)
            self.seal_worker(name)
        write_json(self.certificate, {"format": "SAR_SOFT_SOURCE_CERTIFICATE_V1", "complete": True,
            "full_validation_evidence": True, "smoke": False, "test_accessed": False,
            "protocol_sha256": self.protocol_sha,
            **{name + "_checkpoint": str((self.root / (name + "_q") / "last.pth").resolve())
               for name in ("baseline", "sar")},
            **{name + "_checkpoint_sha256": digest(self.root / (name + "_q") / "last.pth")
               for name in ("baseline", "sar")}})
        for name in ("baseline", "sar"):
            self.seal_worker(name)

    def make_evaluation(self, root, q_name, q_source, *, initial, absent_damage=False):
        root.mkdir(parents=True)
        (root / "predictions").mkdir()
        totals = {arm: np.zeros((4, 4), dtype=np.int64) for arm in (*HISTORICAL, *ARMS)}
        events = {}
        journal = []
        for index in range(450):
            event = "event-a" if index < 225 else "event-b"
            gt = np.array([[0, 1, 2], [3, 1, 255]], dtype=np.int16)
            if absent_damage:
                gt[gt == 2] = 1
                gt[gt == 3] = 0
            if self.different_sar_targets and q_name == "sar":
                gt[0, 0] = 1
            arrays = {"target": gt[None], "building_grid": np.array([[[[.2, .9]]]], dtype=np.float32)}
            arrays["support_j2"] = np.arange(16, dtype=np.int32).reshape(2, 8)
            arrays["support_j3"] = np.arange(16, dtype=np.int32).reshape(2, 8)
            cms = {}
            for arm in (*HISTORICAL, *ARMS):
                pred = np.where(gt == 255, index % 4, gt).astype(np.uint8)
                if not absent_damage and arm in ARMS:
                    if arm == "unsplit" and index % 2 == 0:
                        pred[0, 2] = 1
                    if arm == "capacity_control" and index % 3 == 0:
                        pred[0, 2] = 1
                    if arm == "hard" and index % 4 == 0:
                        pred[0, 2] = 1
                    if arm == "soft" and q_name == "baseline":
                        pred[0, 2] = 1
                    if arm == "soft" and q_name == "sar" and index % 5 == 0:
                        pred[1, 1] = 0
                arrays["prediction_" + arm] = pred[None]
                cm = confusion(gt, pred)
                cms[arm] = cm.tolist()
                totals[arm] += cm
                events.setdefault(event, {}).setdefault(arm, np.zeros((4, 4), dtype=np.int64))
                events[event][arm] += cm
            archive = root / "predictions" / f"{index:04d}.npz"
            np.savez_compressed(archive, **arrays)
            journal.append({"index": index, "image": f"{event}_{index:08d}", "event": event,
                            "confusion": cms, "native_a1_replay": {"original": {"passed": True},
                                                                     "q_source": {"passed": True}},
                            "prediction_file": archive.relative_to(root).as_posix(),
                            "prediction_sha256": digest(archive)})
        (root / "images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in journal))
        immutable = {"files": True, "runtime": True, "building_route_code": True,
                     "route_q_source": True, "readouts": True}
        result = {"format": FORMAT + "_EVAL", "complete": True, "full_validation_evidence": True,
                  "image_count": 450, "expected_image_count": 450, "smoke": False, "initial": initial,
                  "metrics": {arm: metric(cm) for arm, cm in totals.items()},
                  "per_event": {event: {arm: metric(cm) for arm, cm in rows.items()} for event, rows in events.items()},
                  "reproduction": {"passed": True, "checks": []},
                  "e0_metric_comparison": {"passed": True, "checks": []},
                  "q_a1_reproduction": {"passed": True, "checks": []},
                  "immutable": immutable, "native_a1_replay_ok": True, "native_a1_replay_max_abs": 0,
                  "initial_unsplit_max_abs": 0 if initial else None,
                  "pool_guard": {"backend": "native", "restored": True, "failures": [], "restore_conflicts": []},
                  "q_source": q_source, "protocol_sha256": self.protocol_sha, "source_sha256": SOURCE,
                  "test_accessed": False, "training": False}
        write_json(root / "result.json", result)
        summary = {"format": FORMAT + "_EVAL", "complete": True, "full_validation_evidence": True,
                   "image_count": 450, "expected_image_count": 450, "smoke": False, "initial": initial,
                   "test_accessed": False, "training": False, "reproduction_passed": True,
                   "e0_protocol_comparison_passed": True, "q_a1_reproduction_passed": True,
                   "native_a1_replay_ok": True, "immutable": immutable,
                   "result_file": "result.json", "result_sha256": digest(root / "result.json"), "phase": "complete"}
        write_json(root / "summary.json", summary)
        return result

    def seal_worker(self, name):
        fit, evaluation = self.fits[name], self.evals[name]
        write_json(evaluation / "worker_record.json", {"format": "SAR_SOFT_WORKER_V1", "complete": True,
                   "action": "evaluate", "protocol_sha256": self.protocol_sha, "adapter_source_sha256": SOURCE,
                   "q_arm": "baseline" if name == "baseline" else "sar_cue", "smoke": False,
                   "test_accessed": False,
                   "fit_root": str(fit.resolve()), "fit_summary_sha256": digest(fit / "summary.json"),
                   "fit_manifest_sha256": digest(fit / "manifest.json"),
                   "fit_checkpoint_sha256": digest(fit / "heads_last.pth"),
                   "training_journal_sha256": digest(fit / "training.jsonl"),
                   "result_sha256": digest(evaluation / "result.json"),
                   "summary_sha256": digest(evaluation / "summary.json"),
                   "source_certificate": str(self.certificate.resolve()),
                   "source_certificate_sha256": digest(self.certificate),
                   "environment": {"CUDA_LAUNCH_BLOCKING": "1"}})

    def reseal_eval(self, name):
        evaluation = self.evals[name]
        summary = read_json(evaluation / "summary.json")
        summary["result_sha256"] = digest(evaluation / "result.json")
        write_json(evaluation / "summary.json", summary)
        self.seal_worker(name)

    def compare(self, output=None):
        self.compare_count += 1
        output = output or (self.root / ("comparison_" + str(self.compare_count)))
        return compare.run(self.fits["baseline"], self.evals["baseline"],
                           self.fits["sar"], self.evals["sar"], output)


class CompareRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = EvidenceFixture(Path(self.temp.name))

    def assert_rejected(self):
        with self.assertRaises((ValueError, RuntimeError, KeyError)):
            self.fixture.compare()

    def test_formal_eight_cells_and_interaction_no_positive_effect_gate(self):
        result = self.fixture.compare()
        self.assertTrue(result["complete"])
        self.assertFalse(result["test_accessed"])
        self.assertFalse(result["training"])
        self.assertEqual(set(result["cells"]), {"baseline", "sar"})
        for name in ("baseline", "sar"):
            self.assertEqual(set(result["cells"][name]), set(ARMS))
            expected = read_json(self.fixture.evals[name] / "result.json")["metrics"]
            for arm in ARMS:
                self.assertAlmostEqual(result["cells"][name][arm]["foreground_miou_pct"],
                                       100 * expected[arm]["valid_miou"])
        cells = result["cells"]
        means = lambda name, arm: cells[name][arm]["foreground_miou_pct"]
        interaction = (means("sar", "soft") - means("sar", "unsplit")) - (
            means("baseline", "soft") - means("baseline", "unsplit"))
        self.assertAlmostEqual(result["effects"]["interaction_pp"]["foreground_miou_pp"], interaction)
        self.assertLess(result["effects"]["soft_minus_unsplit_at_baseline_q"]["foreground_miou_pp"], 0)
        self.assertEqual(set(result["per_event"]), {"event-a", "event-b"})
        transition = result["paired_transitions"]["baseline_soft__to__sar_soft"]["per_class"]
        self.assertEqual(transition["Damaged"]["rescue"], 450)
        self.assertEqual(transition["Intact"]["harm"], 90)

    def test_wrong_q_source_kind_rejected(self):
        for name in ("summary.json", "manifest.json"):
            p = self.fixture.fits["baseline"] / name
            row = read_json(p)
            row["q_source"]["kind"] = "e0_original_q"
            write_json(p, row)
        self.fixture.seal_worker("baseline")
        self.assert_rejected()

    def test_wrong_q_source_arm_rejected(self):
        for name in ("summary.json", "manifest.json"):
            p = self.fixture.fits["sar"] / name
            row = read_json(p)
            row["q_source"]["arm"] = "rgb_capacity"
            write_json(p, row)
        self.fixture.seal_worker("sar")
        self.assert_rejected()

    def test_source_mismatch_even_if_checkpoint_same(self):
        p = self.fixture.evals["sar"] / "result.json"
        row = read_json(p)
        row["source_sha256"] = "f" * 64
        write_json(p, row)
        self.fixture.reseal_eval("sar")
        self.assert_rejected()

    def test_worker_bound_to_other_fit_rejected(self):
        p = self.fixture.evals["sar"] / "worker_record.json"
        row = read_json(p)
        row["fit_root"] = str(self.fixture.fits["baseline"])
        write_json(p, row)
        self.assert_rejected()

    def test_different_train_stream_rejected(self):
        p = self.fixture.fits["sar"] / "training.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        rows[10]["image"] = "another-event_00000001"
        p.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary_path = self.fixture.fits["sar"] / "summary.json"
        summary = read_json(summary_path)
        summary["training_journal_sha256"] = digest(p)
        summary["train_stream_sha256"] = hashlib.sha256("".join(
            str(row["step"]) + "\t" + row["image"] + "\n" for row in rows).encode()).hexdigest()
        write_json(summary_path, summary)
        self.fixture.seal_worker("sar")
        self.assert_rejected()

    def test_prediction_hash_change_rejected(self):
        p = self.fixture.evals["sar"] / "predictions/0000.npz"
        p.write_bytes(p.read_bytes() + b"changed")
        self.assert_rejected()

    def test_actual_map_confusion_mismatch_rejected_after_rehash(self):
        root = self.fixture.evals["sar"]
        p = root / "predictions/0000.npz"
        with np.load(p, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["prediction_soft"][0, 0, 0] = 3
        np.savez_compressed(p, **arrays)
        journal = root / "images.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        rows[0]["prediction_sha256"] = digest(p)
        journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assert_rejected()

    def test_per_event_confusion_mismatch_rejected(self):
        p = self.fixture.evals["baseline"] / "result.json"
        row = read_json(p)
        row["per_event"]["event-a"]["soft"]["confusion"][0][0] += 1
        row["per_event"]["event-a"]["soft"] = metric(row["per_event"]["event-a"]["soft"]["confusion"])
        write_json(p, row)
        self.fixture.reseal_eval("baseline")
        self.assert_rejected()

    def test_order_changed_rejected(self):
        p = self.fixture.evals["sar"] / "images.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        rows[0]["image"], rows[1]["image"] = rows[1]["image"], rows[0]["image"]
        p.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assert_rejected()

    def test_smoke_cannot_be_formal_even_with_450_rows(self):
        p = self.fixture.evals["baseline"] / "summary.json"
        row = read_json(p)
        row["smoke"] = True
        write_json(p, row)
        self.fixture.seal_worker("baseline")
        self.assert_rejected()

    def test_comparison_does_not_modify_evidence(self):
        before = {str(p): digest(p) for p in Path(self.temp.name).rglob("*") if p.is_file()}
        self.fixture.compare()
        after = {path: digest(path) for path in before}
        self.assertEqual(before, after)

    def test_different_targets_rejected_even_when_local_hashes_and_confusions_match(self):
        self.fixture = EvidenceFixture(Path(self.temp.name) / "different_targets", different_sar_targets=True)
        self.assert_rejected()

    def test_ignore_target_255_does_not_enter_confusion(self):
        result = self.fixture.compare()
        self.assertTrue(result["complete"])
        for name in ("baseline", "sar"):
            stored = read_json(self.fixture.evals[name] / "result.json")
            for arm in (*HISTORICAL, *ARMS):
                self.assertEqual(int(np.asarray(stored["metrics"][arm]["confusion"]).sum()), 450 * 5)

    def test_missing_damage_classes_still_use_fixed_foreground_three_mean(self):
        self.fixture = EvidenceFixture(Path(self.temp.name) / "absent_classes", absent_damage=True)
        result = self.fixture.compare()
        self.assertTrue(result["complete"])
        stored = read_json(self.fixture.evals["sar"] / "result.json")
        self.assertEqual(stored["metrics"]["soft"]["iou"], [1.0, 1.0, None, None])
        self.assertAlmostEqual(stored["metrics"]["soft"]["valid_miou"], 1 / 3)
        self.assertAlmostEqual(result["cells"]["sar"]["soft"]["foreground_miou_pct"], 100 / 3)

    def test_reject_stored_miou_using_only_present_classes(self):
        self.fixture = EvidenceFixture(Path(self.temp.name) / "absent_bad_mean", absent_damage=True)
        p = self.fixture.evals["sar"] / "result.json"
        row = read_json(p)
        row["metrics"]["soft"]["valid_miou"] = 1.0
        write_json(p, row)
        self.fixture.reseal_eval("sar")
        self.assert_rejected()

    def test_change_after_prediction_read_is_detected_before_complete_report(self):
        original_load = np.load
        victim = self.fixture.evals["baseline"] / "predictions/0000.npz"
        trigger = self.fixture.evals["sar"] / "predictions/0449.npz"
        changed = []
        def changing_load(path, *args, **kwargs):
            if Path(path).resolve() == trigger.resolve() and not changed:
                victim.write_bytes(victim.read_bytes() + b"mutation after original read")
                changed.append(True)
            return original_load(path, *args, **kwargs)
        with patch.object(compare.np, "load", side_effect=changing_load):
            with self.assertRaisesRegex(ValueError, "changed during comparison"):
                self.fixture.compare()
        self.assertTrue(changed)
        report = read_json(self.fixture.root / "comparison_1/report.json")
        self.assertFalse(report["complete"])
        self.assertFalse(report["full_validation_evidence"])

    def test_certificate_checkpoint_binding_cannot_be_substituted(self):
        row = read_json(self.fixture.certificate)
        row["sar_checkpoint_sha256"] = row["baseline_checkpoint_sha256"]
        write_json(self.fixture.certificate, row)
        for name in ("baseline", "sar"):
            self.fixture.seal_worker(name)
        with self.assertRaisesRegex(ValueError, "certified checkpoint"):
            self.fixture.compare()

    def test_certificate_wrong_protocol_rejected(self):
        row = read_json(self.fixture.certificate)
        row["protocol_sha256"] = "b" * 64
        write_json(self.fixture.certificate, row)
        for name in ("baseline", "sar"):
            self.fixture.seal_worker(name)
        with self.assertRaisesRegex(ValueError, "four-arm source certificate"):
            self.fixture.compare()

    def test_certificate_incomplete_rejected(self):
        original = read_json(self.fixture.certificate)
        for key in ("complete", "full_validation_evidence"):
            with self.subTest(field=key):
                row = copy.deepcopy(original)
                row[key] = False
                write_json(self.fixture.certificate, row)
                for name in ("baseline", "sar"):
                    self.fixture.seal_worker(name)
                with self.assertRaisesRegex(ValueError, "four-arm source certificate"):
                    self.fixture.compare()

    def test_eval_not_bound_to_current_checkpoint_is_rejected(self):
        p = self.fixture.evals["sar"] / "worker_record.json"
        row = read_json(p)
        row["fit_checkpoint_sha256"] = digest(self.fixture.fits["baseline"] / "heads_last.pth")
        write_json(p, row)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.fixture.compare()


if __name__ == "__main__":
    unittest.main()
