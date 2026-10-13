import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

import run_probe
from test_probe_core import scene_fixture


def tensor_sha(t):
    h = hashlib.sha256(str(t.dtype).encode())
    h.update(json.dumps(list(t.shape), separators=(",", ":")).encode())
    h.update(t.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def runtime_fixture():
    return dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                cuda_matmul_allow_tf32=False, cudnn_allow_tf32=True, cudnn_benchmark=False,
                cudnn_deterministic=False, deterministic_algorithms=False,
                deterministic_warn_only=False, float32_matmul_precision="highest", cudnn_enabled=True,
                environment={name:os.environ.get(name) for name in
                             ("CUDA_VISIBLE_DEVICES","NVIDIA_TF32_OVERRIDE","OMP_NUM_THREADS","MKL_NUM_THREADS")})


def captured_fixture(root):
    (root / "check_02").mkdir(parents=True)
    scene = scene_fixture()
    runtime = runtime_fixture()
    scene.update(format="PRT_SEQUENCE_SCENE_V1", diagnostic_only=True, training=False,
                 test_accessed=False, full_validation_evidence=False, index=109, index_one_based=110,
                 provenance=dict(protocol_sha256="p"*64, original_source_sha256="s"*64, runtime_settings=runtime),
                 tensor_sha256={k:tensor_sha(v) for k,v in scene["tensors"].items()},
                 state_sha256={name:{k:tensor_sha(v) for k,v in scene[name].items()}
                               for name in ("parent_classifier_state", "private_heads_state")})
    torch.save(scene, root / "check_02/scene.pt")
    check = dict(format="PRT_SEQUENCE_CHECK_V1", mode="capture", complete=True, scene_captured=True,
                 scene_file="scene.pt", training=False, test_accessed=False, updates_per_arm=dict(p=0,r=0,t=0),
                 scene_image=scene["image"], scene_index=109, protocol_sha256="p"*64,
                 original_source_sha256="s"*64, runtime_settings=runtime,
                 scene_sha256=run_probe.file_sha(root / "check_02/scene.pt"))
    run_probe.write_record(root / "check_02/check.json", check)
    return scene, check


class RunnerTests(unittest.TestCase):
    def test_valid_scene_load_keeps_cpu_scope_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            captured_fixture(root)
            before = run_probe.file_sha(root / "check_02/scene.pt")
            scene, check, ledger = run_probe.load_scene(root)
            self.assertEqual(scene["index"], 109)
            self.assertEqual(scene["tensors"]["feature"].device.type, "cpu")
            self.assertEqual(check["updates_per_arm"], dict(p=0,r=0,t=0))
            self.assertEqual(len(ledger), 2)
            self.assertEqual(before, run_probe.file_sha(root / "check_02/scene.pt"))

    def test_scene_checksum_corruption_stops_before_unpickling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            captured_fixture(root)
            with (root / "check_02/scene.pt").open("ab") as f:
                f.write(b"changed")
            with mock.patch.object(torch, "load") as loading:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    run_probe.load_scene(root)
                loading.assert_not_called()

    def test_rehashed_inner_tensor_mutation_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scene, check = captured_fixture(root)
            scene["tensors"]["feature"][0,0,0,0] += 1
            torch.save(scene, root / "check_02/scene.pt")
            check["scene_sha256"] = run_probe.file_sha(root / "check_02/scene.pt")
            run_probe.write_record(root / "check_02/check.json", check)
            with self.assertRaisesRegex(ValueError, "Internal"):
                run_probe.load_scene(root)

    def test_trained_or_wrong_identity_scene_rejected(self):
        for field, value in (("updates_per_arm",dict(p=1,r=0,t=0)), ("scene_file","../other.pt")):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _scene, check = captured_fixture(root)
                check[field] = value
                run_probe.write_record(root / "check_02/check.json", check)
                with self.assertRaisesRegex(ValueError, "saved, untrained"):
                    run_probe.load_scene(root)

    def test_backend_modes_change_only_named_policy(self):
        before = runtime_fixture()
        with mock.patch.object(torch, "set_num_interop_threads"):
            try:
                base = run_probe.configure_mode(before, "legacy", "cpu")
                for mode, key in (("no_tf32","cudnn_allow_tf32"),
                                  ("cudnn_deterministic","cudnn_deterministic"),
                                  ("no_cudnn","cudnn_enabled")):
                    settings = run_probe.configure_mode(before, mode, "cpu")
                    changed = {k for k in base if settings[k] != base[k]}
                    self.assertEqual(changed, {"mode", key})
            finally:
                run_probe.configure_mode(before, "legacy", "cpu")

    def test_no_automatic_cpu_fallback(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False), mock.patch.object(torch.cuda, "set_device") as setting:
            with self.assertRaisesRegex(ValueError, "no automatic CPU fallback"):
                run_probe.configure_mode(runtime_fixture(), "legacy", "cuda:0")
            setting.assert_not_called()

    def test_tf32_environment_override_change_is_rejected(self):
        captured = runtime_fixture()
        with mock.patch.dict(os.environ, {"NVIDIA_TF32_OVERRIDE": "changed-from-capture"}):
            with self.assertRaisesRegex(ValueError, "NVIDIA_TF32_OVERRIDE"):
                run_probe.configure_mode(captured, "no_tf32", "cpu")

    def test_flat_file_four_fresh_cpu_processes_use_real_pool(self):
        from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.numeric import runtime_settings
        from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1.core import source_hash
        package = Path(__file__).resolve().parent
        repo = package.parent
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            capture, old = base / "capture", base / "old"
            (old / "fresh_e0").mkdir(parents=True)
            scene, check = captured_fixture(capture)
            runtime = runtime_settings()
            names = ["fake_" + str(i) for i in range(450)]
            names[109] = scene["image"]
            protocol = dict(format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True, full_validation_evidence=True,
                reproduction=dict(passed=True), image_count=450, scope="all", smoke=False, profile="legacy", pool_mode="native",
                input_files_unchanged=True, code_unchanged=True, frozen_evidence_unchanged=True, numerical_unchanged=True,
                selected_images=names, test_accessed=False, runtime_settings=runtime)
            e0 = old / "fresh_e0/protocol.json"
            run_probe.write_record(e0, protocol)
            source = source_hash()
            protocol_sha = run_probe.file_sha(e0)
            scene["provenance"].update(protocol_sha256=protocol_sha, original_source_sha256=source,
                                       runtime_settings=runtime, source_failed_run=str(old))
            scene_path = capture / "check_02/scene.pt"
            torch.save(scene, scene_path)
            check.update(protocol_sha256=protocol_sha, original_source_sha256=source,
                         runtime_settings=runtime, scene_sha256=run_probe.file_sha(scene_path))
            run_probe.write_record(capture / "check_02/check.json", check)
            flat = base / "probe.py"
            flat.write_text((package / "probe_core.py").read_text() + "\n\n" +
                            (package / "run_probe.py").read_text().replace("from __future__ import annotations\n", ""))
            out = base / "results"
            result = subprocess.run([sys.executable, str(flat), "--repo", str(repo),
                                     "--check-root", str(capture), "--device", "cpu", "--output", str(out)],
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = run_probe.read_record(out / "summary.json")
            self.assertTrue(report["diagnostic_complete"])
            self.assertEqual(list(report["modes"]), list(run_probe.MODES))
            self.assertFalse(report["historical_failure_resolved"])
            self.assertFalse(report["downstream_authorized"])
            self.assertEqual(check["scene_sha256"], run_probe.file_sha(scene_path))
            for row in report["overview"].values():
                self.assertTrue(row["fixed_inputs_and_heads_unchanged"])
                self.assertTrue(row["p_fixed_all_within_tolerance"])
                self.assertTrue(row["pool_fixed_all_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
