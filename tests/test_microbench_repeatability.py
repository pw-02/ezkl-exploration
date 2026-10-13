import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

from experiments.microbench.repeatability import (
    fresh_output,
    pin_proving_settings,
    prepare,
    select_jobs,
    settings_fingerprint,
    setup_key,
    worker,
    write_json,
)
from experiments.microbench.analysis import analyze_rows, estimate


def job(number, duration, memory):
    return {
        "job_status": "COMPLETED",
        "job_id": f"job_{number}",
        "job_name": f"model_sub_model_{number}",
        "setup_key": f"key_{number}",
        "job_runtime(s)": str(duration),
        "max_process_memory(GB)": str(memory),
    }


def row(rep, memory, condition="cold", fingerprint="same", stage="PROVING"):
    return {
        "model": "model",
        "partition": 1,
        "condition": condition,
        "setup_key": "key",
        "settings_fingerprint": fingerprint,
        "stage": stage,
        "repetition": rep,
        "peak_memory_gib": memory,
        "duration_s": rep,
        "memory_samples": 1 if memory is not None else 0,
        "avg_cpu_machine_pct": 20,
    }


class SelectionTests(unittest.TestCase):
    def test_selection_deduplicates_roles(self):
        selected = select_jobs([job(7, 10, 20)])
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(selected[0]["criteria"]), 3)

    def test_duration_and_memory_are_selected_independently(self):
        selected = select_jobs([
            job(1, 1, 50), job(2, 10, 1), job(3, 100, 2),
            job(4, 200, 3), job(5, 300, 4),
        ])
        self.assertEqual([item["partition"] for item in selected], [3, 5, 1])

    def test_invalid_population_is_rejected(self):
        with self.assertRaises(ValueError):
            select_jobs([])
        failed = job(1, 1, 1)
        failed["job_status"] = "FAILED"
        with self.assertRaises(ValueError):
            select_jobs([failed])

    def test_settings_fingerprint_detects_change(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            write_json(a, {"a": 1, "b": 2})
            write_json(b, {"b": 2, "a": 1})
            self.assertEqual(settings_fingerprint(a), settings_fingerprint(b))
            write_json(b, {"a": 1, "b": 3})
            self.assertNotEqual(settings_fingerprint(a), settings_fingerprint(b))

    def test_output_never_overwrites_existing_directory(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench"
            fresh_output(path)
            with self.assertRaises(FileExistsError):
                fresh_output(path)

    def test_cold_calibration_pins_reference_configuration(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = root / "setup/settings.json"
            reference = root / "reference_settings.json"
            fingerprint = root / "fingerprint"
            write_json(settings, {"run_args": {"input_scale": 13}})
            pin_proving_settings(settings, reference, fingerprint, root / "first", False)
            write_json(settings, {"run_args": {"input_scale": 12}})
            actual = pin_proving_settings(
                settings, reference, fingerprint, root / "second", False
            )
            self.assertEqual(actual["run_args"]["input_scale"], 13)


class PredictionTests(unittest.TestCase):
    def test_estimator_warmup_and_switch(self):
        self.assertIsNone(estimate([10]))
        self.assertEqual(estimate([10, 20]), 20)
        self.assertEqual(estimate([10, 20, 30, 40]), 40)
        self.assertEqual(estimate([10, 20, 30, 40, 50]), 46)

    def test_future_values_cannot_affect_previous_prediction(self):
        _, predictions, _ = analyze_rows([
            row(1, 1), row(2, 2), row(3, 1000), row(4, 4)
        ])
        memory = [r for r in predictions if r["metric"] == "peak_memory_gib"]
        self.assertEqual(memory[0]["predicted"], 2)
        self.assertEqual(memory[0]["prior_observations"], 2)
        self.assertEqual(memory[0]["shortfall"], 998)

    def test_no_history_leak_between_conditions(self):
        rows = [
            row(1, 1), row(2, 2), row(3, 3),
            row(1, 100, "warm"), row(2, 200, "warm"),
        ]
        _, predictions, _ = analyze_rows(rows)
        memory = [r for r in predictions if r["metric"] == "peak_memory_gib"]
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory[0]["condition"], "cold")


class PreparationTests(unittest.TestCase):
    def fixtures(self, root):
        model = root / "model.onnx"
        model.write_bytes(b"model")
        baseline = root / "baseline.json"
        write_json(baseline, {"input_data": [[0.0]]})
        inputs = []
        for k in range(1, 4):
            path = root / f"input{k}.json"
            write_json(path, {"input_data": [[float(k)]]})
            inputs.append(str(path))
        config = root / "config.json"
        write_json(config, {
            "mnist_gan": {
                "model": str(model),
                "baseline_input": str(baseline),
                "inputs": inputs,
            }
        })
        source = root / "source"
        source.mkdir()
        data = job(1, 10, 2)
        data["job_name"] = "mnist_gan_sub_model_1"
        data["setup_key"] = setup_key(b"fixed component")
        with (source / "job_report.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data))
            writer.writeheader()
            writer.writerow(data)
        module = ModuleType("zkinfer.graph.onnx_splitter")

        def split(**kwargs):
            values = json.loads(Path(kwargs["input_data_path"]).read_text())
            return [SimpleNamespace(
                name="sub_model_1",
                model=SimpleNamespace(SerializeToString=lambda: b"fixed component"),
                input_data=values,
            )]

        module.split_onnx_model_with_inputs = split
        args = SimpleNamespace(
            source=source,
            models=["mnist_gan"],
            config=config,
            output=root / "prepared",
            repetitions=3,
            seed=0,
            input_mode="files",
            noise_scale=.01,
            integer_upper_bound=None,
        )
        return args, module

    def test_full_model_inputs_materialize_fixed_partition(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, module = self.fixtures(root)
            with patch.dict("sys.modules", {
                "onnxruntime": ModuleType("onnxruntime"),
                "zkinfer.graph.onnx_splitter": module,
            }):
                prepare(args)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(len(manifest["cases"]), 1)
            self.assertEqual(len(manifest["cases"][0]["inputs"]), 4)


class WorkerTests(unittest.TestCase):
    def fake_runtime(self, calls):
        prover = ModuleType("zkinfer.runtime.node.prover_task")
        prover.LocalProverTask = SimpleNamespace(
            from_dict=lambda data: SimpleNamespace(**data)
        )

        class Stages:
            def __init__(self, **kwargs):
                self.input_data_path = kwargs["input_data_path"]
                self.settings_path = str(
                    Path(kwargs["proving_cache_root_dir"]) / "settings.json"
                )

            def calibrate_settings_with_fallback(self):
                calls.append(("calibration_input", self.input_data_path))
                hit = Path(self.settings_path).exists()
                write_json(self.settings_path, {"run_args": {"logrows": 12}})
                return hit, 0, 0

        class Runner:
            def run(self, task):
                stages = self._build_proof_stages(
                    task, str(Path(task.output_dir) / "status.txt")
                )
                hit, _, _ = stages.calibrate_settings_with_fallback()
                calls.append(("proof_input", stages.input_data_path))
                metrics = {
                    key: hit for key in [
                        "ezkl_calibrate_settings_used_cache",
                        "ezkl_compile_circuit_used_cache",
                        "ezkl_key_gen_used_cache",
                    ]
                }
                metrics["ezkl_fused_stage_trace_available"] = True
                return SimpleNamespace(
                    metrics=metrics,
                    to_dict=lambda: {"metrics": metrics},
                )

        prover.StageAwareEZKLProofStages = Stages
        prover.ProverTaskRunner = Runner

        admission = ModuleType("zkinfer.runtime.node.stage_admission")

        def controller(
            *, root_dir, setup_key, enabled, fallback_memory_gb,
            memory_safety_reserve_gb, history_percentile
        ):
            self.assertFalse(enabled)
            self.assertEqual(memory_safety_reserve_gb, 8)
            self.assertEqual(fallback_memory_gb, 0)
            self.assertEqual(history_percentile, .9)
            return None

        admission.StageAdmissionController = controller
        return {
            "zkinfer.runtime.node.prover_task": prover,
            "zkinfer.runtime.node.stage_admission": admission,
        }

    def payload(self, root, condition="cold"):
        out = root / "run"
        out.mkdir()
        task = {
            "output_dir": str(out),
            "input_path": "varied.json",
            "setup_dir": str(root / "setup"),
            "model_path": "model.onnx",
            "tmp_dir": str(root / "tmp"),
            "stage_admission_dir": str(root / "admission"),
            "setup_key": "key",
            "memory_safety_reserve_gb": 8,
            "predicted_peak_memory_gb": 0,
            "history_percentile": .9,
        }
        payload = {
            "task": task,
            "condition": condition,
            "fingerprint_path": str(root / "fingerprint"),
            "calibration_input": "reference.json",
            "srs_dir": str(root / "srs"),
            "verify": False,
        }
        path = root / "task.json"
        write_json(path, payload)
        return path, out

    def test_worker_uses_current_stage_admission_api(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, out = self.payload(root)
            calls = []
            with patch.dict("sys.modules", self.fake_runtime(calls)):
                self.assertEqual(worker(path), 0)
            self.assertEqual(
                calls,
                [
                    ("calibration_input", "reference.json"),
                    ("proof_input", "varied.json"),
                ],
            )
            self.assertTrue(json.loads((out / "result.json").read_text())["ok"])

    def test_worker_constructs_real_disabled_stage_admission_controller(self):
        try:
            from zkinfer.runtime.node.stage_admission import StageAdmissionController
        except ModuleNotFoundError as exc:
            self.skipTest(f"Runtime dependency unavailable: {exc}")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, out = self.payload(root)
            modules = self.fake_runtime([])
            modules[
                "zkinfer.runtime.node.stage_admission"
            ].StageAdmissionController = StageAdmissionController
            with patch.dict("sys.modules", modules):
                self.assertEqual(worker(path), 0)
            self.assertTrue((root / "admission" / "reservations").is_dir())


if __name__ == "__main__":
    unittest.main()
