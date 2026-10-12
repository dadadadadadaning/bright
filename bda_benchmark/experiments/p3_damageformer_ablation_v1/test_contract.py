import unittest

try:
    from .variants import VARIANTS, get_variant
except ImportError:
    from variants import VARIANTS, get_variant


class ContractTests(unittest.TestCase):
    def test_progressive_contract(self):
        self.assertEqual(set(VARIANTS), {"a0", "a1", "a2", "a3"})
        self.assertEqual(get_variant("a0").roi_alpha, 1.0)
        self.assertEqual(get_variant("a1").roi_alpha, 0.1)
        self.assertFalse(get_variant("a1").use_bags)
        self.assertTrue(get_variant("a2").use_bags)
        self.assertFalse(get_variant("a2").use_hierarchy)
        self.assertTrue(get_variant("a3").use_hierarchy)

    def test_no_inference_residual_variant(self):
        for spec in VARIANTS.values():
            self.assertIn("claim", spec.__dict__)


if __name__ == "__main__":
    unittest.main()
