import unittest

try:
    from bda_benchmark.model.DamageFormer_S8_P5_DCMF import get_p5_variant_spec
    TORCH_AVAILABLE = True
except ImportError:
    get_p5_variant_spec = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is available on the training server")
class VariantContractTest(unittest.TestCase):
    def test_raw_cross_attention_changes_only_the_declared_factor(self):
        control = get_p5_variant_spec("c0")
        raw_attention = get_p5_variant_spec("c2_raw")
        filtered_attention = get_p5_variant_spec("c2")

        self.assertFalse(control.filter_augmentation)
        self.assertFalse(control.cross_attention)
        self.assertFalse(raw_attention.filter_augmentation)
        self.assertTrue(raw_attention.cross_attention)
        self.assertFalse(raw_attention.common_unique_decoupling)
        self.assertTrue(filtered_attention.filter_augmentation)
        self.assertTrue(filtered_attention.cross_attention)


if __name__ == "__main__":
    unittest.main()
