"""CPU mathematical contracts; these are not BRIGHT/CUDA experiment evidence."""

import json
import math
import unittest

import torch
import torch.nn.functional as F

try:
    from .anchor import anchor_terms, gradient_alignment, serialized_anchor_terms
except ImportError:
    from anchor import anchor_terms, gradient_alignment, serialized_anchor_terms


def image_probability(values, *, dtype=torch.float64, requires_grad=False):
    return torch.tensor(values, dtype=dtype).reshape(1, 4, 1, 1).requires_grad_(requires_grad)


class AnchorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(78)
        self.logits = torch.randn(2, 4, 3, 5, dtype=torch.float64, requires_grad=True)
        self.student = self.logits.softmax(dim=1)
        self.teacher = torch.randn_like(self.logits).softmax(dim=1).detach().requires_grad_(True)
        self.target = torch.randint(0, 4, (2, 3, 5))
        self.target[0, 0, 0] = 255

    def terms(self, **kwargs):
        return anchor_terms(self.student, self.teacher, self.target,
                            student_logits=self.logits, **kwargs)

    def test_direction_is_teacher_to_student(self):
        teacher = image_probability([0.7, 0.1, 0.1, 0.1])
        student = image_probability([0.1, 0.2, 0.3, 0.4])
        got = anchor_terms(student, teacher, torch.zeros(1, 1, 1, dtype=torch.long))["kl"]
        expected = sum(float(p) * math.log(float(p / q)) for p, q in zip(teacher.flatten(), student.flatten()))
        reverse = sum(float(q) * math.log(float(q / p)) for p, q in zip(teacher.flatten(), student.flatten()))
        self.assertAlmostEqual(got.item(), expected, places=12)
        self.assertGreater(abs(got.item() - reverse), 0.1)

    def test_teacher_detached_and_student_logits_get_gradient(self):
        self.terms()["weighted_kl"].backward()
        self.assertIsNone(self.teacher.grad)
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertGreater(self.logits.grad.abs().sum().item(), 0)

    def test_matches_torch_kl_and_manual_masked_reduction(self):
        terms = self.terms()
        pixel = F.kl_div(self.logits.log_softmax(1), self.teacher.detach(), reduction="none").sum(1)
        expected = torch.stack([pixel[n][self.target[n] != 255].mean() for n in range(2)]).mean()
        torch.testing.assert_close(terms["kl"], expected)

    def test_chain_rule_exact(self):
        terms = self.terms()
        torch.testing.assert_close(terms["kl"], terms["localization_kl"] + terms["severity_kl_weighted"],
                                   atol=1e-12, rtol=1e-12)

    def test_equal_distributions_have_zero_loss_and_logit_gradient(self):
        terms = anchor_terms(self.student, self.student.detach(), self.target, student_logits=self.logits)
        self.assertLess(abs(terms["kl"].item()), 1e-14)
        terms["kl"].backward()
        self.assertLess(self.logits.grad.abs().max().item(), 1e-14)

    def test_chain_rule_gradient_matches_joint_gradient(self):
        terms = self.terms()
        joint, = torch.autograd.grad(terms["kl"], self.logits, retain_graph=True)
        parts, = torch.autograd.grad(terms["localization_kl"] + terms["severity_kl_weighted"], self.logits)
        torch.testing.assert_close(joint, parts, atol=1e-12, rtol=1e-12)

    def test_class_and_correctness_contributions_partition_objective(self):
        terms = self.terms()
        torch.testing.assert_close(terms["per_class_contribution"].sum(), terms["kl"])
        torch.testing.assert_close(terms["teacher_correct_kl_contribution"] +
                                   terms["teacher_wrong_kl_contribution"], terms["kl"])
        self.assertEqual(terms["valid_counts"].sum().item(), 29)

    def test_background_is_included_without_confidence_or_correctness_gate(self):
        self.target.fill_(0)
        terms = self.terms()
        self.assertGreater(terms["kl"].item(), 0)
        torch.testing.assert_close(terms["per_class_contribution"][0], terms["kl"])
        self.assertEqual(terms["valid_counts"].tolist(), [30, 0, 0, 0])

    def test_gt_label_permutation_does_not_change_objective(self):
        first = self.terms()
        permuted_target = torch.where(self.target == 255, self.target, (self.target + 1) % 4)
        second = anchor_terms(self.student, self.teacher, permuted_target, student_logits=self.logits)
        torch.testing.assert_close(first["kl"], second["kl"])
        self.assertFalse(torch.equal(first["per_class_contribution"], second["per_class_contribution"]))

    def test_ignore_pixel_has_zero_gradient(self):
        self.terms()["kl"].backward()
        self.assertEqual(self.logits.grad[0, :, 0, 0].abs().sum().item(), 0)

    def test_all_ignore_zero_still_connected(self):
        self.target.fill_(255)
        terms = self.terms()
        self.assertEqual(terms["kl"].item(), 0)
        terms["weighted_kl"].backward()
        self.assertEqual(self.logits.grad.abs().sum().item(), 0)
        self.assertEqual(serialized_anchor_terms(terms)["per_class_conditional_kl"], [None] * 4)

    def test_all_ignore_image_remains_in_image_average(self):
        only_first = anchor_terms(self.student[:1], self.teacher[:1], self.target[:1],
                                  student_logits=self.logits[:1])["kl"]
        self.target[1].fill_(255)
        torch.testing.assert_close(self.terms()["kl"], only_first / 2)

    def test_per_image_average_not_global_valid_pixel_average(self):
        self.target[0].fill_(255)
        self.target[0, 0, 0] = 2
        terms = self.terms()
        pixel = F.kl_div(self.logits.log_softmax(1), self.teacher.detach(), reduction="none").sum(1)
        expected = (pixel[0, 0, 0] + pixel[1].mean()) / 2
        global_mean = (pixel[0, 0, 0] + pixel[1].sum()) / 16
        torch.testing.assert_close(terms["kl"], expected)
        self.assertGreater(abs(terms["kl"].item() - global_mean.item()), 1e-4)

    def test_duplicate_batch_does_not_change_objective(self):
        single = self.terms()["kl"]
        duplicate = anchor_terms(self.student.repeat(2, 1, 1, 1), self.teacher.repeat(2, 1, 1, 1),
                                 self.target.repeat(2, 1, 1), student_logits=self.logits.repeat(2, 1, 1, 1))["kl"]
        torch.testing.assert_close(single, duplicate)

    def test_unbatched_matches_batched(self):
        terms = anchor_terms(self.student[0], self.teacher[0], self.target[0], student_logits=self.logits[0])
        other = anchor_terms(self.student[:1], self.teacher[:1], self.target[:1], student_logits=self.logits[:1])
        torch.testing.assert_close(terms["kl"], other["kl"])

    def test_zero_teacher_mass_and_zero_student_probability_are_finite(self):
        teacher = image_probability([1, 0, 0, 0])
        student = image_probability([0, 1, 0, 0], requires_grad=True)
        terms = anchor_terms(student, teacher, torch.zeros(1, 1, 1, dtype=torch.long))
        self.assertTrue(torch.isfinite(terms["kl"]))
        self.assertEqual(terms["severity_kl_weighted"].item(), 0)
        torch.testing.assert_close(terms["kl"], terms["localization_kl"])
        terms["kl"].backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(terms["student_clamped_count"].item(), 3)

    def test_teacher_only_foreground_chain_rule(self):
        terms = anchor_terms(image_probability([0.3, 0.1, 0.2, 0.4]),
                             image_probability([0, 0.2, 0.3, 0.5]), torch.ones(1, 1, 1, dtype=torch.long))
        torch.testing.assert_close(terms["kl"], terms["localization_kl"] + terms["severity_kl_weighted"])

    def test_extreme_logits_underflow_has_finite_nonzero_correction_gradient(self):
        logits = image_probability([1000, -1000, -1000, -1000], dtype=torch.float32, requires_grad=True)
        student = logits.softmax(1)
        teacher = image_probability([0, 0, 1, 0], dtype=torch.float32)
        terms = anchor_terms(student, teacher, torch.full((1, 1, 1), 2), student_logits=logits)
        self.assertEqual(student[0, 2, 0, 0].item(), 0)
        self.assertAlmostEqual(terms["kl"].item(), 2000.0)
        torch.testing.assert_close(terms["kl"], terms["localization_kl"] + terms["severity_kl_weighted"])
        terms["kl"].backward()
        torch.testing.assert_close(logits.grad.flatten(), torch.tensor([1., 0., -1., 0.]))
        self.assertEqual(terms["probability_floor"].item(), 0)

    def test_half_inputs_compute_float32_and_keep_gradients(self):
        logits = self.logits.detach().half().requires_grad_()
        # Torch 2.1 CPU has no half softmax kernel; emulate mixed-precision output.
        terms = anchor_terms(logits.float().softmax(1).half(), self.teacher.detach().half(), self.target, student_logits=logits)
        self.assertEqual(terms["kl"].dtype, torch.float32)
        terms["kl"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_gradcheck_logit_path(self):
        teacher = image_probability([0.1, 0.2, 0.3, 0.4])
        target = torch.ones(1, 1, 1, dtype=torch.long)
        logits = torch.randn(1, 4, 1, 1, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: anchor_terms(z.softmax(1), teacher, target, student_logits=z)["kl"], (logits,)))

    def test_weight_only_scales_additive_term(self):
        zero = self.terms(weight=0)
        more = self.terms(weight=0.4)
        torch.testing.assert_close(zero["kl"], more["kl"])
        self.assertEqual(zero["weighted_kl"].item(), 0)
        torch.testing.assert_close(more["weighted_kl"], more["kl"] * 0.4)

    def test_serialization_has_no_nan(self):
        self.target.fill_(0)
        result = serialized_anchor_terms(self.terms())
        self.assertEqual(result["per_class_conditional_kl"][1:], [None, None, None])
        json.dumps(result, allow_nan=False)

    def test_gradient_alignment_does_not_populate_grads_or_break_backward(self):
        terms = self.terms()
        original = F.cross_entropy(self.logits, self.target, ignore_index=255)
        report = gradient_alignment(original, terms["weighted_kl"], self.logits)
        self.assertEqual(report["gradient_space"], "logits")
        self.assertIsNone(self.logits.grad)
        self.assertGreater(report["anchor_grad_norm"], 0)
        self.assertLessEqual(abs(report["cosine"]), 1)
        (original + terms["weighted_kl"]).backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())

    def test_gradient_alignment_zero_and_colinear(self):
        original = self.logits.square().mean()
        same = gradient_alignment(original, 0.2 * original, self.logits)
        self.assertAlmostEqual(same["cosine"], 1, places=12)
        self.assertAlmostEqual(same["anchor_to_original_grad_norm"], 0.2, places=12)
        zero = gradient_alignment(original, 0 * original, self.logits)
        self.assertIsNone(zero["cosine"])

    def test_invalid_weight_and_labels_rejected(self):
        for weight in [-1, float("nan"), float("inf"), True, "0.1"]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                self.terms(weight=weight)
        for invalid in [-1, 4, 254]:
            self.target[0, 0, 0] = invalid
            with self.subTest(label=invalid), self.assertRaises(ValueError):
                self.terms()

    def test_invalid_probabilities_and_mismatch_rejected(self):
        for value in [float("nan"), float("inf"), -0.1, 1.1]:
            bad = self.student.detach().clone()
            bad[0, 0, 0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                anchor_terms(bad, self.teacher, self.target)
        with self.assertRaises(ValueError):
            anchor_terms(self.student * 0.5, self.teacher, self.target)
        with self.assertRaises(ValueError):
            anchor_terms(self.student, self.teacher[:1], self.target)
        with self.assertRaises(ValueError):
            anchor_terms(self.student, self.teacher, self.target, student_logits=self.logits + torch.randn_like(self.logits))
        with self.assertRaises(TypeError):
            anchor_terms(self.student, self.teacher, self.target.float())
        with self.assertRaises(ValueError):
            anchor_terms(self.student, self.teacher, self.target[:, :, :-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
