"""CPU contracts for fixed-weight Test evaluation; no Test images are loaded."""
from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .core import ARMS, PAIRS, Accumulator, evaluate_image, read_test_names, sample_paths


def fixture():
    generator = torch.Generator().manual_seed(731)
    feature = torch.randn(1, 256, 2, 3, generator=generator)
    head = nn.Conv2d(256, 4, 1)
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=generator) * 0.16)
        head.bias.copy_(torch.tensor([-0.4, 0.6, 0.1, -0.2]))
    # The duplicate ID at pixel 0 must add, and region 3 has zero mass.
    ids = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 2, 2, 0, 1]])
    q = torch.tensor([[[0.7, 0.2, 0.8, 0.4, 0.6, 0.1],
                       [0.3, 0.8, 0.2, 0.6, 0.4, 0.9]]])
    size = (4, 5)
    logits = F.interpolate(head(feature), size=size, mode="bilinear", align_corners=False)
    region = torch.randn(1, 4, *size, generator=generator).softmax(1)
    result = dict(q=q, ids=ids, count=4, shape=(2, 3), old_pixel=logits, prob=region)
    return SimpleNamespace(base=SimpleNamespace(clf_2=head)), result, feature, size


def dense_reference(front, result, feature, size):
    """Independent linear-algebra reference: W[N,K], not grouped scatter code."""
    q, ids, count = result["q"], result["ids"], result["count"]
    route = torch.zeros(feature.shape[-2] * feature.shape[-1], count)
    for position in range(route.shape[0]):
        for slot in range(ids.shape[0]):
            route[position, int(ids[slot, position])] += q[0, slot, position]
    mass = route.sum(0)
    normalized = route.T / torch.where(mass > 0, mass, torch.ones_like(mass))[:, None]
    pixel = feature[0].flatten(1).T
    pooled = normalized @ pixel
    weight, bias = front.base.clf_2.weight[:, :, 0, 0], front.base.clf_2.bias
    direct_logits = pixel @ weight.T + bias
    region_logits = pooled @ weight.T + bias

    def resize(values):
        probability = values.T.reshape(1, 4, *feature.shape[-2:])
        probability = F.interpolate(probability, size, mode="bilinear", align_corners=False)
        return probability / probability.sum(1, keepdim=True)

    return {
        "a1_pixel": result["old_pixel"].softmax(1),
        "pixel_grid": resize(direct_logits.softmax(-1)),
        "feature_mean": resize(((route @ pooled) @ weight.T + bias).softmax(-1)),
        "region_mean": resize(route @ region_logits.softmax(-1)),
        "grid_probq": resize(route @ (normalized @ direct_logits.softmax(-1))),
        "region": result["prob"],
    }


class TestSplitContracts(unittest.TestCase):
    def test_canonical_split_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_set.txt"
            path.write_text("event_b_2\n\nevent_a-1\n", encoding="utf-8")
            self.assertEqual(read_test_names(path, ["train_a"], ["val_a"], 2),
                             ["event_b_2", "event_a-1"])

    def test_rejects_empty_duplicate_overlap_unsafe_and_wrong_count(self):
        cases = [("", [], [], 1), ("a\na", [], [], 2),
                 ("a", ["a"], [], 1), ("a", [], ["a"], 1),
                 ("../a", [], [], 1), ("dir/a", [], [], 1),
                 ("dir\\a", [], [], 1), ("a.tif", [], [], 1),
                 ("a b", [], [], 1), ("a", [], [], 2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_set.txt"
            for content, train, val, count in cases:
                with self.subTest(content=content, train=train, val=val, count=count):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_test_names(path, train, val, count)

    def test_noncanonical_filename_and_default_count_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val_set.txt"
            path.write_text("event_1", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_test_names(path, [], [], 1)
            path = path.rename(path.with_name("test_set.txt"))
            with self.assertRaises(ValueError):
                read_test_names(path, [], [])

    def test_dataset_suffixes_and_path_safety(self):
        root = Path("dataset")
        self.assertEqual(sample_paths(root, "event_42"), {
            "pre": root / "pre-event/event_42_pre_disaster.tif",
            "post": root / "post-event/event_42_post_disaster.tif",
            "label": root / "target/event_42_building_damage.tif",
        })
        for bad in ("../a", "C:\\a", "a/b", "", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sample_paths(root, bad)


class TestPredictionContracts(unittest.TestCase):
    def test_six_paths_match_independent_dense_math(self):
        front, result, feature, size = fixture()
        expected = dense_reference(front, result, feature, size)
        original = {key: value.clone() for key, value in result.items() if torch.is_tensor(value)}
        actual, checks = evaluate_image(front, result, feature, size)
        self.assertEqual(set(actual), set(ARMS))
        for arm in ARMS:
            with self.subTest(arm=arm):
                torch.testing.assert_close(actual[arm], expected[arm], atol=2e-6, rtol=2e-5)
                self.assertEqual(tuple(actual[arm].shape), (1, 4, *size))
        self.assertTrue(checks["a1_native_replay"]["passed"])
        self.assertTrue(checks["q_normalization"]["passed"])
        self.assertTrue(checks["constant_reconstruction"]["passed"])
        self.assertEqual(checks["empty_regions"], 1)
        for key, value in original.items():
            self.assertTrue(torch.equal(result[key], value))
        # Native A1 interpolates logits; pixel_grid interpolates probabilities.
        self.assertGreater(float((actual["a1_pixel"] - actual["pixel_grid"]).abs().max()), 1e-3)

    def test_a1_replay_reports_changed_native_output(self):
        front, result, feature, size = fixture()
        result["old_pixel"] = result["old_pixel"].clone()
        result["old_pixel"][:, 2] += 0.5
        _, checks = evaluate_image(front, result, feature, size)
        self.assertFalse(checks["a1_native_replay"]["passed"])

    def test_labels_only_change_accounting_not_predictions(self):
        self.assertEqual(tuple(inspect.signature(evaluate_image).parameters),
                         ("front", "result", "feature", "output_size"))
        front, result, feature, size = fixture()
        first, _ = evaluate_image(front, result, feature, size)
        predictions = {arm: p.argmax(1) for arm, p in first.items()}
        a, b = Accumulator(), Accumulator()
        a.add("a", "event", torch.zeros(1, *size, dtype=torch.long), predictions)
        b.add("a", "event", torch.ones(1, *size, dtype=torch.long), predictions)
        second, _ = evaluate_image(front, result, feature, size)
        self.assertNotEqual(a.finish()["metrics"], b.finish()["metrics"])
        for arm in ARMS:
            self.assertTrue(torch.equal(first[arm], second[arm]))

    def test_rejects_bad_region_probabilities(self):
        for kind in ("nan", "negative", "unnormalized", "wrong_size"):
            with self.subTest(kind=kind):
                front, result, feature, size = fixture()
                p = result["prob"].clone()
                if kind == "nan":
                    p[0, 0, 0, 0] = float("nan")
                elif kind == "negative":
                    p[0, 0, 0, 0] = -0.1
                elif kind == "unnormalized":
                    p *= 0.5
                else:
                    p = p[:, :, :-1]
                result["prob"] = p
                with self.assertRaises(ValueError):
                    evaluate_image(front, result, feature, size)

    def test_rejects_invalid_routes_and_grid_mismatch(self):
        for kind in ("q", "shape", "missing"):
            with self.subTest(kind=kind):
                front, result, feature, size = fixture()
                if kind == "q":
                    result["q"] *= 0.5
                elif kind == "shape":
                    result["shape"] = (3, 2)
                else:
                    del result["old_pixel"]
                with self.assertRaises(ValueError):
                    evaluate_image(front, result, feature, size)


class TestAccountingContracts(unittest.TestCase):
    def test_ignore_labels_events_and_true_intact_damage_transitions(self):
        target = torch.tensor([[[1, 1, 2, 2, 3, 0, 255]]])
        before = torch.tensor([[[1, 2, 1, 2, 3, 0, 3]]])
        after = torch.tensor([[[2, 1, 2, 1, 3, 0, 0]]])
        predictions = {arm: before.clone() for arm in ARMS}
        predictions["grid_probq"] = after
        acc = Accumulator()
        rows = acc.add("event_a_1", "event_a", target, predictions)
        self.assertEqual(rows["pixel_grid"], [[1, 0, 0, 0], [0, 1, 1, 0],
                                               [0, 1, 1, 0], [0, 0, 0, 1]])
        second_y = torch.tensor([[[1]]])
        acc.add("event_b_1", "event_b", second_y,
                {arm: second_y.clone() for arm in ARMS})
        report = acc.finish()
        self.assertEqual(report["image_count"], 2)
        self.assertEqual(report["metrics"]["pixel_grid"]["count"], 7)
        self.assertEqual(report["per_event"]["event_a"]["pixel_grid"]["count"], 6)
        self.assertEqual(report["per_event"]["event_b"]["pixel_grid"]["count"], 1)
        self.assertEqual(set(report["transitions"]), set(PAIRS))
        transition = report["transitions"]["pixel_grid__grid_probq"]
        joint = transition["joint_gt_reference_candidate"]
        self.assertEqual(joint[1][1][2], 1)  # Correct Intact turned into Damaged.
        self.assertEqual(joint[1][2][1], 1)  # Old Intact false positive removed.
        self.assertEqual(sum(sum(sum(row) for row in plane) for plane in joint), 7)
        damaged = transition["per_class"]["Damaged"]
        self.assertEqual((damaged["rescue"], damaged["harm"], damaged["delta_fp"]), (1, 1, 0))

    def test_dynamic_existing_memory_arms(self):
        arms = ("j1", "j2", "j3")
        acc = Accumulator(arms=arms, pairs=("j1__j3",))
        y = torch.tensor([[[1, 2]]])
        acc.add("a", "event", y, {"j1": torch.tensor([[[1, 1]]]),
                                    "j2": y.clone(), "j3": y.clone()})
        report = acc.finish()
        self.assertEqual(set(report["metrics"]), set(arms))
        self.assertEqual(report["transitions"]["j1__j3"]["per_class"]["Damaged"]["rescue"], 1)

    def test_rejects_bad_prediction_contracts(self):
        y = torch.tensor([[[0, 1]]])
        for kind in ("float", "class", "shape", "missing"):
            with self.subTest(kind=kind):
                predictions = {arm: y.clone() for arm in ARMS}
                if kind == "float":
                    predictions["a1_pixel"] = y.float()
                elif kind == "class":
                    predictions["a1_pixel"][0, 0, 0] = 4
                elif kind == "shape":
                    predictions["a1_pixel"] = y[0]
                else:
                    del predictions["a1_pixel"]
                with self.assertRaises(ValueError):
                    Accumulator().add("a", "event", y, predictions)

    def test_rejects_ambiguous_arm_and_pair_plans(self):
        for arms, pairs in (((), ()), (("j1", "j1"), ()),
                            (("j1",), ("j1__j3",)),
                            (("j1",), ("j1__j1", "j1__j1"))):
            with self.subTest(arms=arms, pairs=pairs), self.assertRaises(ValueError):
                Accumulator(arms, pairs)

    def test_rejected_late_arm_does_not_partially_commit(self):
        y = torch.tensor([[[0, 1, 2, 3]]])
        acc = Accumulator()
        valid = {arm: y.clone() for arm in ARMS}
        acc.add("first", "existing_event", y, valid)
        before = acc.finish()
        invalid = {arm: y.clone() for arm in ARMS}
        invalid[ARMS[-1]][0, 0, 0] = 4
        with self.assertRaises(ValueError):
            acc.add("second", "new_event", y, invalid)
        self.assertEqual(acc.finish(), before)


if __name__ == "__main__":
    unittest.main()
