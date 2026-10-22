import json
from pathlib import Path
import tempfile
import unittest
from compare_runs import compare
from analyze_results import CLASSES


class CompareRunTests(unittest.TestCase):
    def write_run(self, folder, variant, better=False):
        folder.mkdir()
        cm = [[10, 1, 0, 0], [2, 5, 1, 0], [1, 2, 3, 0], [0, 1, 1, 4]]
        pred = [r[:] for r in cm]
        if better: pred[1][0] -= 1; pred[1][1] += 1
        names = [f"image_{i}" for i in range(450)]
        plan_keys = ("source", "addon_sha256", "extension_files", "post500_lr_multiplier", "scheduler_horizon",
                     "warmup_steps", "maximum_lr", "weight_decay", "grad_clip", "optimizer", "endpoint_step")
        manifest = dict(initial_head_sha256="same", train_names=["train1"], val_names=names,
                        actual_training_names=["train1"], fixed_train_names=["train1"], protocol_info={},
                        parameter_count=10, support_arm="d", runtime_config={"output": str(folder), "lr": 1e-4},
                        plan={key: "same" for key in plan_keys})
        (folder/"manifest.json").write_text(json.dumps(manifest))
        (folder/"summary.json").write_text(json.dumps({"complete": True, "smoke": False}))
        dest = folder/"val_step_002500"; dest.mkdir()
        rows = [dict(image=n, event="event1", confusion={"grid_probq": cm, variant: pred}) for n in names]
        (dest/"images.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        report = dict(format="BRIGHT_MEMORY_READOUT_DESIGN_V1_EVALUATION", complete=True,
                      final_integrity_pending=False, phase="complete", smoke=False, split="val", seed=3407,
                      step=2500, class_names=CLASSES, requested_names=names, image_count=450, variant=variant,
                      metrics={"grid_probq": {"confusion": [[x*450 for x in row] for row in cm]},
                               variant: {"confusion": [[x*450 for x in row] for row in pred]}})
        path=dest/"result.json"; path.write_text(json.dumps(report))
        return path

    def test_aligned_aggregate_and_per_image(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.write_run(Path(temp)/"a", "original")
            b=self.write_run(Path(temp)/"b", "read_aux", True)
            result, rows=compare(a,b)
            self.assertGreater(result["differences"]["miou4_pp"], 0)
            self.assertEqual(len(rows),450)
            self.assertTrue(result["grid_confusion_identical"])

    def test_rejects_protocol_and_aggregate_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.write_run(Path(temp)/"a", "original")
            b=self.write_run(Path(temp)/"b", "read_aux", True)
            d=json.loads(b.read_text()); d["seed"]=42; b.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,"seed"): compare(a,b)
            d["seed"]=3407; d["metrics"]["read_aux"]["confusion"][0][0]+=1; b.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,"aggregate"): compare(a,b)


if __name__ == "__main__": unittest.main()
