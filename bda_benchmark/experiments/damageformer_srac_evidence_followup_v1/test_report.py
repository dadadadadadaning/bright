"""Synthetic contracts; no BRIGHT data, torch, CUDA, training, or source mutations."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from .report import (aggregate_memory, build_report, confusion_metrics,
                     difficult_retrieval_rates, gate_status, inventory_report,
                     transition_metrics)


def fixture(name="image_1", scale=1, no_damaged=False):
    joint = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    joint[0][0][0] = 5 * scale
    joint[1][1][2] = scale
    joint[3][3][3] = scale
    if not no_damaged:
        joint[2][1][2] = 2 * scale
        joint[2][2][1] = scale
    old = [[sum(joint[g][o]) for o in range(4)] for g in range(4)]
    new = [[sum(joint[g][o][n] for o in range(4)) for n in range(4)] for g in range(4)]
    identity = [[[old[g][o] if n == o else 0 for n in range(4)] for o in range(4)] for g in range(4)]
    cohort = {}
    for label, n, final in (("damaged_j1_wrong", 0 if no_damaged else 2*scale, 0 if no_damaged else 2*scale),
                             ("damaged_j1_correct", 0 if no_damaged else scale, 0),
                             ("intact_j1_correct", scale, scale), ("destroyed_j1_correct", scale, 0)):
        cohort[label] = dict(count=n, any_supported_pixels=n, support_mass_sum=n*.5,
                            unsupported_mass_sum=n*.5, evidence_damaged=n,
                            evidence_damaged_and_final_damaged=final, final_damaged=final,
                            conditional_damaged_probability_sum=n*.8)
    retrieval = {}
    for c in range(4):
        q = 0 if c == 2 and no_damaged else scale
        retrieval[str(c)] = dict(queries=q, supported_queries=q, hit_at_k_count=q,
                                valid_slots=8*q, same_class_slots=4*q)
    retrieval["all_damaged_label_mass"] = dict(total=0. if no_damaged else 3.*scale,
        with_any_support=0. if no_damaged else 3.*scale, with_damaged_hit=0. if no_damaged else 3.*scale,
        in_non_damaged_dominant_queries=0.)
    return dict(image=name, event="event", metrics={"j1": dict(confusion=old), "j2": dict(confusion=new),
        "j3": dict(confusion=new)}, transitions={"j1_to_j2": dict(gt_old_new=joint),
        "j1_to_j3": dict(gt_old_new=joint), "j1_to_j1": dict(gt_old_new=identity)},
        difficult_cohorts={"j2": copy.deepcopy(cohort), "j3": copy.deepcopy(cohort)},
        retrieval={"j2": copy.deepcopy(retrieval), "j3": copy.deepcopy(retrieval)})


def add_difficult(row):
    row["difficult_retrieval"] = {}
    for arm, groups in row["difficult_cohorts"].items():
        row["difficult_retrieval"][arm] = {
            name: dict(count=value["count"], support_mass_sum=value["support_mass_sum"],
                       damaged_hit_mass_sum=value["support_mass_sum"]*.75,
                       damaged_neighbor_fraction_mass_sum=value["support_mass_sum"]*.25,
                       intact_neighbor_fraction_mass_sum=value["support_mass_sum"]*.5)
            for name, value in groups.items()}
    return row


def distribution(n, mean):
    return dict(regions=n, dominant_counts=[0, 0, n, 0], soft_label_sum=[n*(1-mean), 0, n*mean, 0],
                damaged_positive_regions=n, damaged_soft_sum_non_damaged_dominant=0.)


class ReportContracts(unittest.TestCase):
    def test_sum_confusion_not_average_scores(self):
        a, b = fixture("a"), fixture("b", scale=3)
        # Make the larger image perfect for j2/j3. Averaging image metrics is wrong.
        for arm in ("j2", "j3"):
            joint = b["transitions"]["j1_to_"+arm]["gt_old_new"]
            for g in range(4):
                for old in range(4):
                    total = sum(joint[g][old])
                    joint[g][old] = [total if new == g else 0 for new in range(4)]
            b["metrics"][arm]["confusion"] = [[sum(sum(joint[g][old]) for old in range(4)) if new == g else 0 for new in range(4)] for g in range(4)]
            for name, cohort in b["difficult_cohorts"][arm].items():
                final = cohort["count"] if name.startswith("damaged") else 0
                cohort.update(final_damaged=final, evidence_damaged_and_final_damaged=final)
        result = aggregate_memory([a, b])
        expected = [[a["metrics"]["j2"]["confusion"][i][j]+b["metrics"]["j2"]["confusion"][i][j] for j in range(4)] for i in range(4)]
        self.assertEqual(result["metrics"]["j2"], confusion_metrics(expected))
        naive = sum(confusion_metrics(r["metrics"]["j2"]["confusion"])["valid_miou"] for r in (a,b))/2
        self.assertNotAlmostEqual(result["metrics"]["j2"]["valid_miou"], naive)

    def test_fixed_cohort_and_transition_denominators(self):
        result = aggregate_memory([fixture("a"), fixture("b", 2)])
        c = result["difficult_cohorts"]["j2"]["damaged_j1_wrong"]
        self.assertEqual(c["count"], 6)
        self.assertEqual(c["mean_supported_Q_mass"], .5)
        self.assertEqual(c["final_damaged_given_evidence_damaged"], 1.)
        t = result["transitions"]["j1_to_j2"]
        self.assertEqual(t["damaged_to_intact_harm"], 3)
        self.assertEqual(t["intact_gt_correct_to_damaged_harm"], 3)
        self.assertEqual(t["classes"]["Damaged"]["rescue"], 6)
        self.assertEqual(result["retrieval"]["j2"]["Damaged"]["hit_at_k"], 1.)
        self.assertEqual(result["retrieval"]["j2"]["Damaged"]["same_class_fraction_of_valid_neighbors"], .5)

    def test_no_damaged_is_null_not_zero_success(self):
        result = aggregate_memory([fixture(no_damaged=True)])
        c = result["difficult_cohorts"]["j2"]["damaged_j1_wrong"]
        self.assertIsNone(c["final_damaged_fraction_of_cohort"])
        self.assertIsNone(result["retrieval"]["j2"]["Damaged"]["hit_at_k"])
        self.assertEqual(c["count"], 0)

    def test_duplicate_names_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_memory([fixture(), fixture()])

    def test_key_change_rejected(self):
        a, b = fixture("a"), fixture("b")
        b["transitions"].pop("j1_to_j1")
        with self.assertRaisesRegex(ValueError, "keys changed"):
            aggregate_memory([a, b])

    def test_nonfinite_and_inconsistent_cohorts_rejected(self):
        a = fixture()
        a["difficult_cohorts"]["j2"]["damaged_j1_wrong"]["support_mass_sum"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            aggregate_memory([a])
        a = fixture()
        for arm in ("j2", "j3"):
            a["difficult_cohorts"][arm]["damaged_j1_wrong"]["final_damaged"] = 1
            a["difficult_cohorts"][arm]["damaged_j1_wrong"]["evidence_damaged_and_final_damaged"] = 1
        with self.assertRaisesRegex(ValueError, "rescue disagrees"):
            aggregate_memory([a])

    def test_transition_marginals_rejected(self):
        row = fixture()
        row["transitions"]["j1_to_j2"]["gt_old_new"][0][0][0] += 1
        with self.assertRaisesRegex(ValueError, "marginal"):
            aggregate_memory([row])

    def test_formal_gate_and_failed_450_rejected(self):
        good = dict(complete=True, gates_passed=True, full_validation_evidence=True, smoke=False,
                    image_count=450, test_accessed=False, reproduction=dict(passed=True))
        self.assertTrue(gate_status(good, good, 450)["formal_full_val"])
        for key in ("complete", "gates_passed", "full_validation_evidence"):
            bad = dict(good, **{key: False})
            self.assertFalse(gate_status(bad, good, 450)["formal_full_val"])
        bad = dict(good, failure=dict(message="CUDA error"))
        self.assertFalse(gate_status(bad, good, 450)["formal_full_val"])
        self.assertFalse(gate_status(good, None, 450)["formal_full_val"])
        self.assertFalse(gate_status(good, good, 408)["formal_full_val"])
        smoke = dict(good, smoke=True, image_count=2, reproduction=dict(passed=None))
        self.assertFalse(gate_status(smoke, smoke, 2)["formal_full_val"])

    def test_guard_failure_and_stale_success_flags_rejected(self):
        good = dict(complete=True, gates_passed=True, full_validation_evidence=True, smoke=False,
                    image_count=450, test_accessed=False, reproduction=dict(passed=True),
                    followup_source_unchanged=True,
                    pool_guard=dict(restored=True, failures=[], restore_conflicts=[]))
        self.assertTrue(gate_status(good, good, 450)["formal_full_val"])
        for guard in (None, {}, dict(restored=False),
                      dict(restored=True, failures=[{"message":"CUDA fault"}]),
                      dict(restored=True, restore_conflicts=["old.module.pool"])):
            bad = dict(good, pool_guard=guard)
            self.assertFalse(gate_status(bad, good, 450)["formal_full_val"])
        bad = dict(good, followup_source_unchanged=False)
        self.assertFalse(gate_status(bad, good, 450)["formal_full_val"])

    def test_successful_new_subset_is_not_full(self):
        subset = dict(complete=True, gates_passed=True, full_validation_evidence=False,
                      full_val_evidence=False, smoke=False, diagnostic_subset=True, image_count=409,
                      test_accessed=False, reproduction=dict(passed=None),
                      pool_guard=dict(restored=True, failures=[], restore_conflicts=[]))
        result = gate_status(subset, subset, 409)
        self.assertFalse(result["formal_full_val"])
        self.assertIn("smoke/subset or unknown smoke flag", result["reasons"])

    def test_event_composition_and_eligibility(self):
        data = dict(bank=distribution(4, .75), by_event={"large":distribution(2,.88), "small":distribution(2,.62)},
            train_shards_scanned=True, matched_source_shards=True,
            all_train=dict(eligible_regions=distribution(110, 96/110)),
            all_train_by_event={"large":dict(eligible_regions=distribution(100,.9)),
                                "small":dict(eligible_regions=distribution(10,.6))})
        result = inventory_report(data)
        c = result["composition"]
        self.assertAlmostEqual(c["source_reweighted_to_bank_events"], .75)
        self.assertAlmostEqual(c["within_event_selection_drop_pp"], 0.)
        self.assertAlmostEqual(c["event_composition_drop_pp"], c["total_drop_pp"])
        self.assertEqual(result["eligible_dominant_D_after_same_event_exclusion"], {"large":2,"small":2})
        data["matched_source_shards"] = False
        self.assertIsNone(inventory_report(data)["composition"])

    def test_inventory_empty_D_and_inconsistent_counts(self):
        data = dict(bank=distribution(0, 0), by_event={"e":distribution(0,0)},
            train_shards_scanned=True, matched_source_shards=True,
            all_train=dict(eligible_regions=distribution(0,0)),
            all_train_by_event={"e":dict(eligible_regions=distribution(0,0))})
        result = inventory_report(data)
        self.assertIsNone(result["composition"]["total_drop_pp"])
        data["bank"] = distribution(1,.9)
        with self.assertRaisesRegex(ValueError, "totals"):
            inventory_report(data)

    def test_real_user_inventory_schema_and_fourteen_event_decomposition(self):
        # Counts and event means are from the user's completed CPU inventory,
        # not invented Val scores. Event means were printed to four decimals.
        rows = [
            ("bata-explosion",289,64,77.1959,72.7088),
            ("beirut-explosion",881,64,83.1304,83.4794),
            ("congo-volcano",36,36,55.2362,55.2362),
            ("haiti-earthquake",81,64,57.1258,58.4706),
            ("hawaii-wildfire",709,64,75.8432,79.0355),
            ("la_palma-volcano",175,64,75.8134,74.5815),
            ("libya-flood",4401,64,85.8212,86.0180),
            ("marshall-wildfire",49,49,63.4673,63.4673),
            ("mexico-hurricane",16492,64,86.9408,86.9704),
            ("morocco-earthquake",135,64,72.8027,75.1966),
            ("myanmar-hurricane",687,64,67.8390,62.0690),
            ("noto-earthquake",81,64,52.4966,51.1390),
            ("turkey-earthquake",9148,64,79.7930,79.9901),
            ("ukraine-conflict",17765,64,84.6087,83.3514),
        ]
        data = dict(train_shards_scanned=True, matched_source_shards=True,
            bank=dict(regions=3411, dominant_counts=[896,896,853,766],
                soft_label_sum=[1269.4625647934154,868.4401372433786,656.6559235123841,616.4413760610814],
                damaged_positive_regions=1127, damaged_soft_sum_non_damaged_dominant=34.145322975322415),
            all_train=dict(eligible_regions=dict(regions=2988032, dominant_counts=[2512742,394774,50929,29587],
                soft_label_sum=[2527299.8314465797,382348.564657235,49399.848117587695,28983.755776235572],
                damaged_positive_regions=179701, damaged_soft_sum_non_damaged_dominant=6624.391633001715)),
            by_event={e: distribution(nb,mb/100) for e,ns,nb,ms,mb in rows},
            all_train_by_event={e:dict(eligible_regions=distribution(ns,ms/100)) for e,ns,nb,ms,mb in rows},
            bank_quality=dict(purity={"lt_080":distribution(472,.65), "080_095":distribution(298,.88),
                                      "ge_095":distribution(83,.98)}))
        result = inventory_report(data)
        c = result["composition"]
        self.assertAlmostEqual(100*c["source_D_mean"], 83.9904, places=4)
        self.assertAlmostEqual(100*c["bank_D_mean"], 72.9790, places=4)
        self.assertAlmostEqual(c["event_composition_drop_pp"], 10.5312, places=4)
        self.assertAlmostEqual(c["within_event_selection_drop_pp"], .4802, places=4)
        self.assertEqual(result["eligible_dominant_D_after_same_event_exclusion"]["congo-volcano"],817)
        self.assertEqual(result["eligible_dominant_D_after_same_event_exclusion"]["mexico-hurricane"],789)
        self.assertEqual(result["bank_D_purity_bins"]["lt_080"]["count"],472)

    def test_new_difficult_retrieval_and_old_compatibility(self):
        result = aggregate_memory([add_difficult(fixture("a")), add_difficult(fixture("b", 3))])
        c = result["difficult_retrieval"]["j2"]["damaged_j1_wrong"]
        self.assertEqual(c["weighted_support_coverage"], .5)
        self.assertEqual(c["damaged_hit_mass_per_supported_mass"], .75)
        self.assertEqual(c["damaged_neighbor_fraction_per_supported_mass"], .25)
        self.assertIsNone(aggregate_memory([fixture()])["difficult_retrieval"])
        empty = aggregate_memory([add_difficult(fixture(no_damaged=True))])
        self.assertIsNone(empty["difficult_retrieval"]["j2"]["damaged_j1_wrong"]["damaged_hit_mass_per_supported_mass"])
        with self.assertRaisesRegex(ValueError, "keys changed"):
            aggregate_memory([fixture("a"), add_difficult(fixture("b"))])

    def test_new_difficult_retrieval_mismatch_rejected(self):
        row = add_difficult(fixture())
        row["difficult_retrieval"]["j2"]["damaged_j1_wrong"]["count"] += 1
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            difficult_retrieval_rates(row["difficult_retrieval"], row["difficult_cohorts"])

    def test_partial_file_report_keeps_failed_status_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = dict(complete=False, gates_passed=False, full_validation_evidence=False,
                           smoke=False, test_accessed=False, failure=dict(images_completed=1))
            (root/"summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (root/"images.jsonl").write_text(json.dumps(fixture())+"\n", encoding="utf-8")
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            result = build_report(memory=root)
            self.assertFalse(result["memory"]["status"]["formal_full_val"])
            self.assertEqual(result["memory"]["image_count"], 1)
            self.assertEqual(sum(c["formal_full_val"] for c in result["check_status"].values()), 0)
            self.assertEqual(before, {p.name:p.read_bytes() for p in root.iterdir()})

    def test_final_journal_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"summary.json").write_text("{}", encoding="utf-8")
            (root/"images.jsonl").write_text(json.dumps(fixture())+"\n", encoding="utf-8")
            (root/"audit.json").write_text(json.dumps(dict(image_count=2)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "image counts disagree"):
                build_report(memory=root)

    def test_missing_arguments_and_empty_aggregation(self):
        with self.assertRaisesRegex(ValueError, "Specify"):
            build_report()
        self.assertEqual(aggregate_memory([])["image_count"], 0)
        self.assertIsNone(confusion_metrics([[0]*4 for _ in range(4)])["valid_miou"])


if __name__ == "__main__":
    unittest.main()
