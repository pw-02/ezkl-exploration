"""Admission regression tests; no EZKL installation or proof generation needed."""
import ast
import __future__
import importlib.util
import json
import logging
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


admission = load_module("admission_under_test", "zkinfer/runtime/node/stage_admission.py")
config = load_module("config_under_test", "zkinfer/config/runtime.py")


def pid_exists(pid):
    # The test sandbox has process signals but no mounted /proc filesystem.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def competing_job(root, attempted, entered, release):
    with patch.object(admission.psutil, "virtual_memory", return_value=SimpleNamespace(
        total=64 * 1024**3, available=64 * 1024**3
    )), patch.object(admission, "process_tree_memory_gb", return_value=0), \
            patch.object(admission.psutil, "pid_exists", side_effect=pid_exists):
        controller = admission.StageAdmissionController(
            root, "second", True, 40, memory_safety_reserve_gb=0, mode="whole_job"
        )
        attempted.set()
        with controller.hold_job():
            entered.set()
            release.wait(5)


class WholeJobAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vm = patch.object(admission.psutil, "virtual_memory", return_value=SimpleNamespace(
            total=64 * 1024**3, available=64 * 1024**3
        ))
        self.vm.start()
        self.addCleanup(self.vm.stop)
        self.rss = patch.object(admission, "process_tree_memory_gb", return_value=0)
        self.rss.start()
        self.addCleanup(self.rss.stop)
        self.pids = patch.object(admission.psutil, "pid_exists", side_effect=pid_exists)
        self.pids.start()
        self.addCleanup(self.pids.stop)

    def controller(self, **kwargs):
        options = dict(root_dir=self.temp.name, setup_key="job", enabled=True,
                       fallback_memory_gb=40, memory_safety_reserve_gb=0, mode="whole_job")
        options.update(kwargs)
        return admission.StageAdmissionController(**options)

    def test_stage_transitions_keep_peak_and_exception_releases(self):
        controller = self.controller()
        controller.profiles.update_from_metrics("job", {"stage_preparation_max_memory(GB)": 1})
        with self.assertRaisesRegex(ValueError, "proof failed"):
            with controller.hold_job():
                for stage in ("PREPARATION", "KEY_GEN", "PROVING"):
                    with controller.hold(stage) as info:
                        self.assertEqual(info["wait_time_s"], 0)
                        self.assertEqual(controller.current_reservations()["memory_gb"], 40)
                    self.assertEqual(controller.current_reservations()["memory_gb"], 40)
                raise ValueError("proof failed")
        self.assertEqual(controller.current_reservations()["count"], 0)

    def test_stage_mode_still_releases_each_stage(self):
        controller = self.controller(mode="stage")
        with controller.hold_job():
            self.assertEqual(controller.current_reservations()["count"], 0)
            with controller.hold("PROVING"):
                self.assertEqual(controller.current_reservations()["memory_gb"], 40)
            self.assertEqual(controller.current_reservations()["count"], 0)

    def test_disabled_mode_does_not_reserve(self):
        controller = self.controller(enabled=False)
        with controller.hold_job(), controller.hold("PROVING"):
            self.assertEqual(controller.current_reservations()["count"], 0)

    def test_requires_outer_reservation(self):
        with self.assertRaisesRegex(RuntimeError, "outer hold_job"):
            with self.controller().hold("PROVING"):
                pass

    def test_oversized_job_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            with self.controller(fallback_memory_gb=65).hold_job():
                pass

    def test_competing_process_waits_until_job_finishes(self):
        ctx = multiprocessing.get_context("fork")
        attempted, entered, release = (ctx.Event() for _ in range(3))
        child = ctx.Process(target=competing_job, args=(self.temp.name, attempted, entered, release))
        try:
            with self.controller().hold_job():
                child.start()
                self.assertTrue(attempted.wait(5))
                self.assertFalse(entered.wait(0.4))
            self.assertTrue(entered.wait(5))
            release.set()
            child.join(5)
            self.assertEqual(child.exitcode, 0)
            self.assertEqual(self.controller().current_reservations()["count"], 0)
        finally:
            release.set()
            if child.pid is not None:
                child.join(2)
                if child.is_alive():
                    child.terminate()
                    child.join(2)

    def test_dead_process_reservation_cleaned(self):
        ctx = multiprocessing.get_context("fork")
        attempted, entered, release = (ctx.Event() for _ in range(3))
        child = ctx.Process(target=competing_job, args=(self.temp.name, attempted, entered, release))
        child.start()
        try:
            self.assertTrue(entered.wait(5))
        finally:
            child.terminate()
            child.join(5)
        with self.controller().hold_job():
            self.assertEqual(self.controller().current_reservations()["count"], 1)

    def test_config_default_and_validation(self):
        for mode in ("stage", "whole_job"):
            cfg = OmegaConf.create({"node": {"stage_admission": {"mode": mode}}})
            self.assertEqual(config._resolve_stage_admission_config(cfg).mode, mode)
        cfg = OmegaConf.create({"node": {}})
        self.assertEqual(config._resolve_stage_admission_config(cfg).mode, "stage")
        cfg = OmegaConf.create({"node": {"stage_admission": {"mode": "typo"}}})
        with self.assertRaises(ValueError):
            config._resolve_stage_admission_config(cfg)

    def test_launcher_forwards_mode_and_explicit_null(self):
        # Load the actual command builder without importing the backend/client.
        tree = ast.parse((ROOT / "experiments/run_local_exp.py").read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"node_command", "common_overrides"}]
        ns = {"DictConfig": object, "List": list, "Path": Path,
              "sys": SimpleNamespace(executable="python")}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "launcher", "exec"), ns)
        cfg = OmegaConf.load(ROOT / "experiments/config/config.yaml")
        cfg.node.max_concurrent_jobs = None
        cfg.node.stage_admission.mode = "whole_job"
        command = ns["node_command"](cfg)
        self.assertIn("node.max_concurrent_jobs=null", command)
        self.assertIn("node.stage_admission.mode=whole_job", command)
        cfg.node.max_concurrent_jobs = 4
        self.assertIn("node.max_concurrent_jobs=4", ns["node_command"](cfg))

    def test_runner_holds_through_publication_and_releases_on_failure(self):
        # Exercise the real runner lifecycle with proof computation replaced by
        # a lightweight callback; importing EZKL is unnecessary for this test.
        tree = ast.parse((ROOT / "zkinfer/runtime/node/prover_task.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "ProverTaskRunner")
        ns = {"os": os, "json": json, "logging": logging,
              "ProverTaskResult": lambda **kwargs: SimpleNamespace(**kwargs)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "runner", "exec",
                     flags=__future__.annotations.compiler_flag), ns)
        for fail_at in (None, "proof", "publication"):
            with self.subTest(fail_at=fail_at):
                controller = self.controller()
                task = SimpleNamespace(output_dir=self.temp.name, tmp_dir=self.temp.name,
                                       setup_key="job", stage_admission_enabled=True,
                                       admission_mode="whole_job", transfer_metrics={})
                def proof(**kwargs):
                    with controller.hold("PROVING"):
                        self.assertEqual(controller.current_reservations()["memory_gb"], 40)
                        if fail_at == "proof":
                            raise ValueError("proof")
                    return {}, None
                def publish(*args):
                    self.assertEqual(controller.current_reservations()["memory_gb"], 40)
                    if fail_at == "publication":
                        raise ValueError("publication")
                    return None, {}
                stages = SimpleNamespace(admission=controller, admission_metrics={},
                                         run_all=proof, _update_status=Mock())
                runner = ns["ProverTaskRunner"]()
                runner._build_proof_stages = Mock(return_value=stages)
                runner._start_resource_logger = Mock(return_value=(None, "trace.csv"))
                runner._stop_process = Mock()
                runner._collect_metrics = Mock(return_value={})
                runner._publish_job_artifacts = publish
                if fail_at:
                    with self.assertRaisesRegex(ValueError, fail_at):
                        runner.run(task)
                else:
                    runner.run(task)
                self.assertEqual(controller.current_reservations()["count"], 0)
                runner._stop_process.assert_called_once()


if __name__ == "__main__":
    unittest.main()
