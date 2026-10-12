import unittest
import torch
from .model import EvidenceReadout
from .variants import VARIANTS


def aux():
    probs=torch.full((1,9,4),1/9)
    return {"region_logits":torch.zeros(1,1,3), "association_probabilities":probs,
            "candidate_ids":torch.zeros(1,9,4,dtype=torch.long),
            "region_relation":torch.zeros(1,64,2,2), "region_context":torch.zeros(1,128,2,2),
            "region_confidence":torch.ones(1,1,2,2), "region_alignment_entropy":torch.zeros(1,1,2,2),
            "building_ids":torch.ones(1,2,2,dtype=torch.long)}


class ContractTests(unittest.TestCase):
    def test_variants_are_isolated(self):
        self.assertEqual(set(VARIANTS), {"w0_full_residual","w1_id_calibration","w2_id_logit","w3_id_mass","w5_logit4_kd"})
    def test_all_readouts_are_identity_initialised(self):
        logits=torch.tensor([[[[0.,0.],[0.,0.]],[[2.,2.],[2.,2.]],[[1.,1.],[1.,1.]],[[.5,.5],[.5,.5]]]])
        for name in VARIANTS:
            out=EvidenceReadout(name)(logits, {} if name=="w1_id_calibration" else aux())["damage_logits"]
            self.assertTrue(torch.equal(out, logits), name)
            self.assertTrue(torch.equal(out.argmax(1), logits.argmax(1)), name)
            self.assertTrue(torch.allclose(out.softmax(1), logits.softmax(1), atol=1e-6), name)
    def test_mass_mixer_preserves_background_destroyed_and_id_mass(self):
        module=EvidenceReadout("w3_id_mass")
        module.id_gate_head[-1].bias.data.fill_(10); module.id_expert_residual_head[-1].bias.data.copy_(torch.tensor([-1.,1.]))
        logits=torch.randn(1,4,2,2); before=logits.softmax(1); after=module(logits,aux())["damage_logits"].softmax(1)
        self.assertTrue(torch.allclose(before[:,0],after[:,0],atol=1e-6))
        self.assertTrue(torch.allclose(before[:,3],after[:,3],atol=1e-6))
        self.assertTrue(torch.allclose(before[:,1:3].sum(1),after[:,1:3].sum(1),atol=1e-6))
    def test_no_test_protocol(self):
        source=Path(__file__).with_name("run.sh").read_text(encoding="utf-8")
        self.assertIn("TEST ACCESS IS FORBIDDEN", source)


from pathlib import Path
if __name__ == "__main__": unittest.main()
