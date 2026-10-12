import unittest

from .gates import final_evidence_gate


class GateTests(unittest.TestCase):
    def test_only_full_reproduced_immutable_450_is_formal(self):
        good = {"files": True, "runtime": True, "frozen_models": True}
        self.assertTrue(final_evidence_gate(450, 450, {"passed": True}, good, True))
        self.assertFalse(final_evidence_gate(449, 450, {"passed": True}, good, True))
        self.assertFalse(final_evidence_gate(450, 2, {"passed": True}, good, True))
        self.assertFalse(final_evidence_gate(450, 450, {"passed": False}, good, True))
        self.assertFalse(final_evidence_gate(450, 450, {"passed": True}, {"files": False}, True))
        self.assertFalse(final_evidence_gate(450, 450, {"passed": True}, good, False))

if __name__ == "__main__":
    unittest.main()
