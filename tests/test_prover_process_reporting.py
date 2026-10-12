import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from zkinfer.runtime.node.prover_process import (
    _ExpectedCalibrationCandidateFilter,
    _normalize_disabled_admission_metrics,
    _persist_task_metadata,
)
from zkinfer.runtime.node.prover_task import LocalProverTask


class ProverProcessReportingTests(unittest.TestCase):
    @staticmethod
    def _task(root: Path, phase_admission_enabled: bool = False) -> LocalProverTask:
        return LocalProverTask(
            node_id="node",
            request_id="request",
            job_id="job",
            lease_id="lease",
            setup_key="setup",
            model_path="model.onnx",
            input_path="input.json",
            setup_dir=str(root / "setup"),
            tmp_dir=str(root / "tmp"),
            output_dir=str(root / "output"),
            output_prefix=str(root / "published"),
            output_backend="filesystem",
            phase_admission_enabled=phase_admission_enabled,
        )

    def test_task_metadata_is_persisted_under_job_config(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._task(root)
            _persist_task_metadata(task)
            path = root / "output" / "config" / "task.json"
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["job_id"], "job")
            self.assertEqual(payload["lease_id"], "lease")
            self.assertFalse(payload["phase_admission_enabled"])

    def test_disabled_admission_drops_stage_prediction_fields(self):
        task = self._task(Path("/tmp"), phase_admission_enabled=False)
        metrics = {
            "phase_admission_enabled": False,
            "phase_admission_wait_time(s)": 0.0,
            "phase_admission_proving_wait_time(s)": 0.0,
            "phase_admission_proving_predicted_memory(GB)": 712.0,
            "phase_admission_proving_predicted_cpu_machine(%)": 100.0,
            "phase_admission_proving_estimate_source": "fallback",
            "phase_admission_proving_estimate_samples": 0,
            "stage_proving_max_memory(GB)": 12.0,
        }
        _normalize_disabled_admission_metrics(task, metrics)
        self.assertEqual(metrics["phase_admission_enabled"], False)
        self.assertEqual(metrics["phase_admission_wait_time(s)"], 0.0)
        self.assertEqual(metrics["stage_proving_max_memory(GB)"], 12.0)
        self.assertNotIn("phase_admission_proving_predicted_memory(GB)", metrics)
        self.assertNotIn("phase_admission_proving_estimate_source", metrics)
        self.assertNotIn("phase_admission_proving_wait_time(s)", metrics)

    def test_enabled_admission_metrics_are_preserved(self):
        task = self._task(Path("/tmp"), phase_admission_enabled=True)
        metrics = {"phase_admission_proving_predicted_memory(GB)": 8.0}
        _normalize_disabled_admission_metrics(task, metrics)
        self.assertEqual(metrics["phase_admission_proving_predicted_memory(GB)"], 8.0)

    def test_expected_calibration_candidate_rejection_is_filtered(self):
        filter_ = _ExpectedCalibrationCandidateFilter()
        expected = logging.LogRecord(
            "ezkl.execute", logging.ERROR, __file__, 1,
            "calibration failed extended k is too large", (), None,
        )
        genuine = logging.LogRecord(
            "ezkl.execute", logging.ERROR, __file__, 1,
            "could not compile circuit", (), None,
        )
        self.assertFalse(filter_.filter(expected))
        self.assertTrue(filter_.filter(genuine))


if __name__ == "__main__":
    unittest.main()
