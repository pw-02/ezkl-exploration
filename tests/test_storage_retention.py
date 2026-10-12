import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from zkinfer.runtime.node.prover_task import LocalProverTask, ProverTaskRunner
from zkinfer.storage.shared_store import SharedStorage


class SetupIdentityTests(unittest.TestCase):
    def test_setup_key_includes_setup_policy(self):
        model = b"same-model-bytes"
        base = {
            "setup_identity_version": 1,
            "backend": "ezkl",
            "calibration_target": "resources",
        }
        changed = dict(base, calibration_target="accuracy")

        key_a = SharedStorage.setup_key_for_model(model, base)
        key_a_again = SharedStorage.setup_key_for_model(model, base)
        key_b = SharedStorage.setup_key_for_model(model, changed)

        self.assertEqual(key_a, key_a_again)
        self.assertNotEqual(key_a, key_b)

    def test_model_content_hash_remains_separate_from_setup_key(self):
        model = b"model"
        identity = {"setup_identity_version": 1, "backend": "ezkl"}
        setup_key = SharedStorage.setup_key_for_model(model, identity)

        with tempfile.TemporaryDirectory() as tmp:
            storage = SharedStorage(
                backend="filesystem",
                setup_root=str(Path(tmp) / "setup"),
                requests_root=str(Path(tmp) / "requests"),
            )
            ref = storage.put_model(model, setup_key=setup_key)

            self.assertIn(setup_key, ref.uri)
            self.assertEqual(ref.sha256, hashlib.sha256(model).hexdigest())


class RunArtifactRetentionTests(unittest.TestCase):
    @staticmethod
    def _proof_stages(tmp_dir: Path, setup_dir: Path):
        proof = tmp_dir / "proof.pf"
        witness = tmp_dir / "witness.json"
        vk = setup_dir / "vk.json"
        settings = setup_dir / "settings.json"
        proof.write_bytes(b"proof")
        witness.write_text("{}", encoding="utf-8")
        vk.write_text("{}", encoding="utf-8")
        settings.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            proof_path=str(proof),
            witness_path=str(witness),
            vk_path=str(vk),
            settings_path=str(settings),
            srs_path="srs/kzg17.srs",
        )

    @staticmethod
    def _task(root: Path, retain: bool) -> LocalProverTask:
        tmp_dir = root / "tmp"
        setup_dir = root / "setup"
        output_dir = root / "run"
        request_store = root / "request-store"
        for path in (tmp_dir, setup_dir, output_dir, request_store):
            path.mkdir(parents=True, exist_ok=True)
        return LocalProverTask(
            node_id="node",
            request_id="req_test",
            job_id="job",
            setup_key="setup",
            model_path=str(setup_dir / "model.onnx"),
            input_path=str(tmp_dir / "input.json"),
            setup_dir=str(setup_dir),
            tmp_dir=str(tmp_dir),
            output_dir=str(output_dir),
            output_prefix=str(request_store),
            output_backend="filesystem",
            lease_id="lease",
            retain_run_artifacts=retain,
        )

    def test_default_run_record_does_not_duplicate_proof_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._task(root, retain=False)
            stages = self._proof_stages(Path(task.tmp_dir), Path(task.setup_dir))

            manifest_ref, _ = ProverTaskRunner()._publish_job_artifacts(task, stages)

            run_dir = Path(task.output_dir)
            request_store = Path(task.output_prefix)
            for name in ("proof.pf", "witness.json", "vk.json", "settings.json"):
                self.assertFalse((run_dir / name).exists())
                self.assertTrue((request_store / name).exists())

            self.assertTrue((run_dir / "manifest.json").exists())
            with open(manifest_ref.uri, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertFalse(manifest["retain_run_artifacts"])
            self.assertNotIn("local_path", manifest["artifacts"]["proof.pf"])

    def test_debug_mode_retains_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._task(root, retain=True)
            stages = self._proof_stages(Path(task.tmp_dir), Path(task.setup_dir))

            ProverTaskRunner()._publish_job_artifacts(task, stages)

            run_dir = Path(task.output_dir)
            for name in ("proof.pf", "witness.json", "vk.json", "settings.json"):
                self.assertTrue((run_dir / name).exists())


if __name__ == "__main__":
    unittest.main()
