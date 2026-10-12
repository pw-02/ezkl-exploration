from __future__ import annotations

import csv
import json
import logging
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from analysis.section_3.trace_layout import find_job_dirs, job_identity

import zkinfer.proto.zkservice_pb2 as pb
from zkinfer.config.runtime import build_runtime_config
from zkinfer.profiling.resource_logger import infer_runs_dir
from zkinfer.profiling.resource_parser import parse_resource_usage_file
from zkinfer.runtime.coordinator import Coordinator
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import InferenceRequest, JobStatus, NodeResources, ProofJob, RequestStatus
from zkinfer.runtime.node.supervisor import NodeSupervisor, to_pb_artifact
from zkinfer.runtime.node.prover_task import ProverTaskRunner
from zkinfer.storage.io import load_json, save_json


def runtime_config(root: Path):
    cfg = OmegaConf.load(Path(__file__).parents[1] / "zkinfer/config/config.yaml")
    cfg.coordinator.runs_dir = str(root / "requests")
    cfg.coordinator.logs_dir = str(root / "logs")
    cfg.coordinator.state_db_path = str(root / "state.sqlite3")
    cfg.node.runs_dir = str(root / "node-runs")
    cfg.node.logs_dir = str(root / "node-logs")
    cfg.node.node_id = "test-node"
    cfg.node.local_cache_dir = str(root / "node-cache")
    cfg.node.runtime_state_dir = str(root / "runtime-state")
    cfg.storage.artifact_prefix = str(root / "artifacts")
    cfg.storage.transfer_prefix = str(root / "transfer")
    cfg.storage.setup_cache_prefix = str(root / "setup")
    cfg.storage.setup_cache_enabled = True
    return build_runtime_config(cfg, process_role="coordinator")


def add_request(coordinator: Coordinator, setup_key="circuit"):
    request = InferenceRequest("model", "model.onnx", "input.json", "none", 1, "fifo")
    coordinator._create_request_dirs(request)
    empty = ArtifactRef.empty()
    job = ProofJob(
        "model", "partition", request.request_id, empty, empty,
        ArtifactRef("filesystem", coordinator.proving_cache.root_dir), empty,
        predicted_peak_memory_gb=1, artifact_key=setup_key,
        profiling_file_path=str(Path(coordinator.proving_cache.root_dir) / "resource_profile.json"),
    )
    request.proof_jobs = [job]
    coordinator._enqueue_request(request)
    return request, job


class ResultAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = runtime_config(Path(self.tmp.name))
        self.coordinator = Coordinator(self.cfg, logging.getLogger("test-coordinator"))
        self.addCleanup(self.coordinator.shutdown)
        self.request, self.job = add_request(self.coordinator)
        self.node = NodeResources("node", memory_total_gb=64, memory_available_gb=60)
        self.coordinator.poll_node(self.node)

    def test_missing_stale_and_nonterminal_results_are_rejected(self):
        lease_id = self.job.lease_id
        for status, lease in (("COMPLETED", ""), ("COMPLETED", "old"), ("QUEUED", lease_id)):
            self.assertFalse(self.coordinator.submit_job_result("node", self.job.job_id, status, lease))
            self.assertEqual(self.job.job_status, JobStatus.IN_PROGRESS)

    def test_accepted_completion_is_idempotent_and_conflicting_retry_is_rejected(self):
        lease = self.job.lease_id
        manifest = ArtifactRef("filesystem", "/attempt-a/manifest.json")
        with patch.object(self.coordinator, "_finalize_job") as report, \
                patch.object(self.coordinator, "_schedule_request_finalization") as finalize:
            self.assertTrue(self.coordinator.submit_job_result("node", self.job.job_id, "COMPLETED", lease, manifest))
            self.assertTrue(self.coordinator.submit_job_result("node", self.job.job_id, "COMPLETED", lease, manifest))
            self.assertFalse(self.coordinator.submit_job_result(
                "node", self.job.job_id, "COMPLETED", lease,
                ArtifactRef("filesystem", "/attempt-b/manifest.json"),
            ))
            self.assertEqual(report.call_count, 1)
            self.assertEqual(finalize.call_count, 1)

    def test_old_attempt_cannot_complete_a_job_released_to_same_node(self):
        old_lease = self.job.lease_id
        with self.coordinator.lock:
            self.coordinator._requeue_active_job_locked(self.job.job_id)
        self.coordinator.poll_node(self.node)
        self.assertNotEqual(old_lease, self.job.lease_id)
        self.assertFalse(self.coordinator.submit_job_result("node", self.job.job_id, "COMPLETED", old_lease))

    def test_completion_retry_after_coordinator_restart_reads_terminal_job(self):
        lease = self.job.lease_id
        manifest = ArtifactRef("filesystem", "/attempt-a/manifest.json")
        with patch.object(self.coordinator, "_finalize_job"), \
                patch.object(self.coordinator, "_schedule_request_finalization"):
            self.coordinator.submit_job_result("node", self.job.job_id, "COMPLETED", lease, manifest)
        self.request.request_status = RequestStatus.COMPLETED
        self.request.completed_time = datetime.now(timezone.utc)
        self.coordinator.state_store.save_request(self.request)
        restarted = Coordinator(self.cfg, logging.getLogger("test-restarted"))
        self.addCleanup(restarted.shutdown)
        self.assertNotIn(self.request.request_id, restarted.active_requests)
        self.assertTrue(restarted.submit_job_result("node", self.job.job_id, "COMPLETED", lease, manifest))

    def test_request_finalization_waits_for_all_job_reports(self):
        self.job.job_status = JobStatus.COMPLETED
        self.coordinator.pending_job_reports.add(self.job.job_id)
        with patch.object(self.coordinator.finalization_pool, "submit") as submit:
            self.coordinator._schedule_request_finalization(self.request.request_id)
            submit.assert_not_called()

    def test_concurrent_last_results_cannot_finalize_before_a_delayed_report(self):
        second = ProofJob(
            "model", "second", self.request.request_id,
            self.job.model_ref, self.job.input_ref, self.job.setup_cache_ref,
            self.job.output_prefix_ref, predicted_peak_memory_gb=1,
        )
        second.job_status = JobStatus.QUEUED
        self.request.proof_jobs.append(second)
        self.coordinator.scheduler.add_jobs([second])
        self.coordinator.poll_node(self.node)
        report_started, release_report = threading.Event(), threading.Event()

        def delayed_report(job, perf_metrics=None):
            if job.job_id == self.job.job_id:
                report_started.set()
                if not release_report.wait(timeout=5):
                    raise AssertionError("report was not released")

        with patch.object(self.coordinator, "_finalize_job", side_effect=delayed_report), \
                patch.object(self.coordinator.finalization_pool, "submit") as submit, \
                ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(
                self.coordinator.submit_job_result, "node", self.job.job_id,
                "COMPLETED", self.job.lease_id,
            )
            try:
                self.assertTrue(report_started.wait(timeout=5))
                self.assertTrue(self.coordinator.submit_job_result(
                    "node", second.job_id, "COMPLETED", second.lease_id,
                ))
                submit.assert_not_called()
            finally:
                release_report.set()
            self.assertTrue(first.result(timeout=5))
            submit.assert_called_once()

    def test_shared_profile_accumulates_concurrent_requests_and_refreshes_queued_jobs(self):
        jobs = []
        for _ in range(8):
            _, job = add_request(self.coordinator)
            job.job_status = JobStatus.COMPLETED
            job.started_time = job.queued_time
            job.completed_time = datetime.now(timezone.utc)
            jobs.append(job)
        _, queued = add_request(self.coordinator)
        barrier = threading.Barrier(len(jobs))

        def report(job):
            barrier.wait(timeout=5)
            self.coordinator._finalize_job(job, {
                "max_process_memory(GB)": 12,
                "stage_proving_max_memory(GB)": 12,
                "stage_proving_avg_cpu_machine(%)": 40,
            })

        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            list(pool.map(report, jobs))
        profile = load_json(jobs[0].profiling_file_path)
        self.assertEqual(len(profile["_resource_history"]["observations"]), len(jobs))
        self.assertEqual(queued.predicted_peak_memory_gb, 12)
        self.assertEqual(len(queued.profiling_data["_resource_history"]["observations"]), len(jobs))


class NodeAttemptTests(unittest.TestCase):
    def test_retries_use_disjoint_paths_and_hold_artifacts_until_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = NodeSupervisor(runtime_config(root), logging.getLogger("test-node"))
            model, input_file = root / "model.onnx", root / "input.json"
            model.write_bytes(b"model")
            input_file.write_text('{"input_data": [[1]]}')
            lease = pb.JobLease(
                job_available=True, request_id="request", job_id="job", lease_id="lease-a",
                setup_key="circuit",
                model=to_pb_artifact(ArtifactRef("filesystem", str(model))),
                input=to_pb_artifact(ArtifactRef("filesystem", str(input_file))),
                output_prefix=to_pb_artifact(ArtifactRef("filesystem", str(root / "published"))),
            )
            with ExitStack() as first_pins:
                first = node.prepare_local_task(lease, first_pins)
                lease.lease_id = "lease-b"
                with ExitStack() as second_pins:
                    second = node.prepare_local_task(lease, second_pins)
                    self.assertNotEqual(first.output_prefix, second.output_prefix)
                    self.assertNotEqual(first.output_dir, second.output_dir)
                    self.assertNotEqual(first.tmp_dir, second.tmp_dir)
                    self.assertEqual(first.model_path, second.model_path)
                    self.assertEqual(first.lease_id, "lease-a")
                    manifests = []
                    for task in (first, second):
                        sources = {}
                        for attribute, name in (("proof_path", "proof.pf"), ("witness_path", "witness.json"), ("vk_path", "vk.json"), ("settings_path", "settings.json")):
                            source = Path(task.tmp_dir) / name
                            source.write_text(task.lease_id)
                            sources[attribute] = str(source)
                        sources["srs_path"] = "unchanged.srs"
                        manifest, _ = ProverTaskRunner()._publish_job_artifacts(task, SimpleNamespace(**sources))
                        manifests.append(manifest)
                    self.assertNotEqual(manifests[0].uri, manifests[1].uri)
                    self.assertEqual((Path(first.output_prefix) / "proof.pf").read_text(), "lease-a")
                    self.assertEqual((Path(second.output_prefix) / "proof.pf").read_text(), "lease-b")
                    node.cache.max_bytes = 1
                    node.cache.evict_if_needed()
                    self.assertTrue(Path(first.model_path).exists())
                    self.assertTrue(Path(first.input_path).exists())
            self.assertEqual(node.cache._pin_counts, {})

    def test_rejected_ack_is_logged_as_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = Mock()
            node = NodeSupervisor(runtime_config(Path(tmp)), logger)
            node.stub = Mock()
            node.stub.SubmitJobResult.return_value = pb.StatusAck(ok=False)
            node.submit_job_result(pb.JobLease(job_id="job", lease_id="old"), "COMPLETED")
            logger.warning.assert_called_once()
            logger.info.assert_not_called()


class ResourceTraceTests(unittest.TestCase):
    def test_trace_discovery_selects_accepted_attempt_and_supports_legacy_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "req/artifacts/jobs/legacy"
            stale = root / "req/artifacts/jobs/current/attempts/stale"
            accepted = root / "req/artifacts/jobs/current/attempts/winner"
            unreported = root / "req/artifacts/jobs/unreported/attempts/unknown"
            for directory in (old, stale, accepted, unreported):
                directory.mkdir(parents=True)
                (directory / "resource_usage.csv").write_text("trace")
            reports = root / "req/reports"
            reports.mkdir()
            (reports / "job_report.csv").write_text(
                "request_id,job_id,lease_id\nreq,current,winner\nreq,legacy,\n"
            )
            self.assertEqual(set(find_job_dirs(root)), {old, accepted})
            self.assertEqual(job_identity(accepted), ("req", "current"))

    def test_halo2_subphases_also_contribute_to_proving_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resource.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["type", "stage", "memory_gb", "cpu_percent", "cpu_machine_percent"])
                writer.writeheader()
                for stage, memory, cpu in (("PROVING::FFT", 12, 40), ("PROVING::MSM", 20, 80)):
                    writer.writerow({"type": "process_tree", "stage": stage, "memory_gb": memory, "cpu_percent": cpu * 4, "cpu_machine_percent": cpu})
            metrics = parse_resource_usage_file(str(path))
            self.assertEqual(metrics["stage_proving_max_memory(GB)"], 20)
            self.assertEqual(metrics["stage_proving_avg_cpu_machine(%)"], 60)
            self.assertEqual(metrics["stage_proving_samples"], 2)
            self.assertEqual(metrics["stage_proving_fft_max_memory(GB)"], 12)

    def test_attempt_scoped_status_resolves_runs_directory(self):
        root = Path("/runs")
        self.assertEqual(infer_runs_dir(str(root / "req/artifacts/jobs/job/status.txt")), str(root))
        self.assertEqual(infer_runs_dir(str(root / "req/artifacts/jobs/job/attempts/lease/status.txt")), str(root))


class AtomicProfileTests(unittest.TestCase):
    def test_failed_json_write_preserves_previous_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            save_json({"samples": 3}, str(path))
            path.chmod(0o640)
            with self.assertRaises(TypeError):
                save_json({"unserializable": object()}, str(path))
            self.assertEqual(load_json(str(path)), {"samples": 3})
            save_json({"samples": 4}, str(path))
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(list(Path(tmp).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
