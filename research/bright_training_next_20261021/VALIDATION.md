# Validation record — 2026-10-21

Environment: Python 3.10.22, PyTorch 2.1.2+cpu. No remote GPU or real BRIGHT training was executed here.

Archive SHA256: `994ee877c6edf332e4ebae70c48c81442877dcb6ab053e8cab4323ebb0075e3d`.

14 unittest cases passed (13-case main suite, followed by the added certificate integration case). Tests import the built zipapp, not an unbundled substitute.

- All inherited helper modules match the fixed resume archive byte for byte. In particular, the revised NLL reduction checks remain intact.
- Original continuation integration cases pass both without and with the new scoped observation wrapper: successful continuation, partial paired-update failure followed by recovery, and endpoint checkpoint evaluation recovery without extra optimizer updates.
- The actual original joint classifier, with a deliberately nonzero final layer, gives identical H logits when support features, labels and candidate order change. D responds to those changes. H responds to current-image changes.
- Actual AdamW updates change the H backend while Reader gradients stay absent; D Reader gradients are present.
- A fresh synthetic D/H workflow executes actual model/optimizer updates, writes and reopens its checkpoint, checks RNG and optimizer counts, and promotes reports only after runtime closure.
- Injecting a final validation exception preserves the completed checkpoint and leaves all new reports pending. Failure location is saved.
- The original evaluator handles D/H on 450 synthetic entries, produces the D−H contrast, restores RNG, and labels CPU results as smoke. Formal validation rejects those smoke results.
- Certificate checking accepts a freshly finalized synthetic standalone report bound to the expected checkpoint and rejects a changed per-image journal. A wrong parent checkpoint is rejected before training.

The synthetic report fixtures used to test promotion are explicitly artificial test data. They do not establish real Val450 coverage or accuracy. The real evaluator test independently verifies that CPU results cannot be accepted as formal evidence.

The earlier intermittent `Invalid full-image probabilities: grid_probq` cause remains unresolved. The extension retains the original acceptance rule and adds observations, so a recurrence can still stop the remote run with concrete failure evidence.
