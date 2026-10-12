import unittest

try:
    import torch
    from bda_benchmark.experiments.p5_ablation_validation_v1.audit_model import shift_zero
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    shift_zero = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is available on the training server")
class PerturbationTest(unittest.TestCase):
    def test_shift_has_no_wraparound(self):
        value = torch.zeros(1, 1, 3, 4)
        value[0, 0, 1, 0] = 1
        shifted = shift_zero(value, dx=2, dy=0)
        self.assertEqual(float(shifted[0, 0, 1, 2]), 1.0)
        self.assertEqual(float(shifted[..., :2].sum()), 0.0)

    def test_out_of_frame_is_zero(self):
        value = torch.ones(1, 1, 2, 2)
        shifted = shift_zero(value, dx=5, dy=0)
        self.assertEqual(float(shifted.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
