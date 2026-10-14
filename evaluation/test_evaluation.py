"""Run with: python -m unittest discover -s evaluation -p test_evaluation.py -v"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("evaluation_runner", "run.py")
traffic = load("evaluation_traffic", "traffic.py")


class EvaluationTests(unittest.TestCase):
    def test_seeded_schedule_and_mixture(self):
        first = traffic.make_schedule(["a", "b", "c"], 20, "poisson", 0.2, 42)
        self.assertEqual(first, traffic.make_schedule(["a", "b", "c"], 20, "poisson", 0.2, 42))
        self.assertEqual(first[0]["scheduled_s"], 0)
        self.assertTrue(all(a["scheduled_s"] < b["scheduled_s"] for a, b in zip(first, first[1:])))
        burst = traffic.make_schedule(["a", "b", "c"], 20, "burst", None, 42)
        self.assertTrue(all(item["scheduled_s"] == 0 for item in burst))
        self.assertEqual([i["workload"] for i in first], [i["workload"] for i in burst])

    def test_invalid_schedule(self):
        with self.assertRaises(ValueError):
            traffic.make_schedule(["a"], 2, "poisson", 0, 42)

    def test_admission_plans_only_change_mode(self):
        plans = []
        for baseline in ("whole-job", "stage"):
            args = runner.parser().parse_args(["--baseline", baseline, "--output", "/tmp/evaluation-test-plan"])
            runner.validate(args)
            plans.append(runner.build_plan(args)["runtime_overrides"])
        differences = [key for key in plans[0] if plans[0][key] != plans[1][key]]
        self.assertEqual(differences, ["node.stage_admission.mode"])
        self.assertIsNone(plans[0]["node.max_concurrent_jobs"])

    def test_fixed_and_monolithic_plans(self):
        args = runner.parser().parse_args(["--baseline", "fixed-4"])
        plan = runner.build_plan(args)
        self.assertEqual(plan["runtime_overrides"]["node.max_concurrent_jobs"], 4)
        self.assertFalse(plan["runtime_overrides"]["node.stage_admission.enabled"])
        args.baseline = "monolithic"
        self.assertIn("--monolithic", runner.build_plan(args)["commands"]["traffic"])

    def test_dry_run_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "absent"
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main(["--baseline", "stage", "--output", str(out)])
            self.assertFalse(out.exists())

    def test_all_matrix_baselines_exist(self):
        for path in Path(__file__).parent.glob("*/matrix.json"):
            data = json.loads(path.read_text())
            for baseline in data.get("baselines", []):
                self.assertIn(baseline, runner.BASELINES)

    def test_failure_summary_does_not_claim_successful_makespan(self):
        records = [dict(status="completed", start_s=0, end_s=10, latency_s=10),
                   dict(status="failed", start_s=0, end_s=5, latency_s=5)]
        summary = traffic.summarize(records, 10)
        self.assertEqual(summary["statuses"], {"completed": 1, "failed": 1})
        self.assertIsNone(summary["batch_makespan_s"])
        self.assertEqual(summary["drain_inclusive_successes_per_s"], 0.1)

    def test_traffic_records_success_submission_error_and_failed_request(self):
        channel = SimpleNamespace(close=lambda: None)
        counter = 0
        lock = threading.Lock()
        def submit(request, timeout):
            nonlocal counter
            self.assertGreater(timeout, 0)
            with lock:
                counter += 1
                index = counter
            if index == 2:
                raise RuntimeError("synthetic submission error")
            return SimpleNamespace(request_id=str(index))
        def status(request, timeout):
            return SimpleNamespace(status="failed" if request.request_id == "1" else "completed",
                                   completed_jobs=1, total_jobs=1, message="test")
        stub = SimpleNamespace(SubmitInferenceRequest=submit, GetRequestStatus=status)
        pb = SimpleNamespace(InferenceRequest=lambda **kw: SimpleNamespace(**kw),
                             RequestStatusRequest=lambda **kw: SimpleNamespace(**kw))
        rpc = SimpleNamespace(ZKJobServiceStub=lambda channel: stub)
        modules = {"grpc": SimpleNamespace(insecure_channel=lambda *a, **kw: channel, RpcError=RuntimeError),
                   "zkinfer.proto": SimpleNamespace(zkservice_pb2=pb, zkservice_pb2_grpc=rpc)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            root = Path(tmp)
            cfgdir = root / "experiments/config/workload"
            cfgdir.mkdir(parents=True)
            (cfgdir / "test.yaml").write_text("name: test\nonnx_file: test.onnx\ninput_file: input.json\n")
            out = root / "output"
            with patch.object(traffic, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                code = traffic.main(["--workload", "test", "--requests", "4", "--output", str(out),
                                     "--poll-seconds", "0.01", "--timeout", "5"])
            self.assertEqual(code, 1)
            records = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 4)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["statuses"], {"submission_error": 1, "failed": 1, "completed": 2})
            self.assertTrue(all("start_s" in record and "end_s" in record for record in records))

    def test_cold_profiled_run_requires_both_histories_and_copies_no_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = root / "experiments/config/workload"
            configs.mkdir(parents=True)
            (configs / "test.yaml").write_text("name: test\n")
            args = runner.parser().parse_args(["--baseline", "stage", "--workload", "test",
                                               "--profile-dir", str(root), "--execute"])
            with patch.object(runner, "ROOT", root), self.assertRaisesRegex(ValueError, "matching coordinator profiles"):
                runner.validate(args)
            (root / "node.json").write_text('{}')
            coordinator = root / "coordinator"
            key = coordinator / "key"
            key.mkdir(parents=True)
            (key / "cold_resource_profile.json").write_text('{"cold": true}')
            (key / "resource_profile.json").write_text('{"warm": true}')
            (key / "pk.key").write_text('must not be copied')
            args.coordinator_profile_dir = coordinator
            with patch.object(runner, "ROOT", root):
                runner.validate(args)
            out = root / "trial"
            runner.copy_profile_snapshots(args, out)
            copied = list((out / "coordinator-profiles/key").iterdir())
            self.assertEqual([p.name for p in copied], ["cold_resource_profile.json"])
            self.assertEqual(json.loads(copied[0].read_text()), {"cold": True})
            overrides = runner.build_plan(args)["runtime_overrides"]
            self.assertFalse(overrides["storage.reusable_setup.enabled"])
            self.assertTrue(overrides["storage.resource_profiles.enabled"])

    def test_warm_profile_selection_does_not_select_cold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "key").mkdir()
            for filename in ("resource_profile.json", "cold_resource_profile.json"):
                (root / "key" / filename).write_text('{}')
            args = runner.parser().parse_args(["--baseline", "stage", "--setup", "warm",
                                               "--coordinator-profile-dir", str(root)])
            self.assertEqual([p.name for p in runner.coordinator_profile_files(args)], ["resource_profile.json"])


if __name__ == "__main__":
    unittest.main()
