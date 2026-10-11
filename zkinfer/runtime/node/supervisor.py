from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import grpc
import hydra
import psutil
from omegaconf import DictConfig

import zkinfer.proto.zkservice_pb2 as pb
import zkinfer.proto.zkservice_pb2_grpc as pb_grpc
from zkinfer.config.runtime import RuntimeConfig, build_runtime_config
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.node.cache import NodeCache
from zkinfer.runtime.node.phase_admission import (
    HistoricalResourceProfiles,
    ResourceAdmissionController,
)
from zkinfer.runtime.node.prover_task import LocalProverTask, ProverTaskResult
from zkinfer.utils.network import get_ip


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def from_pb_artifact(ref) -> ArtifactRef:
    if ref is None or not getattr(ref, "backend", "") or not getattr(ref, "uri", ""):
        return ArtifactRef.empty()
    return ArtifactRef(
        backend=ref.backend,
        uri=ref.uri,
        sha256=ref.sha256,
        size_bytes=int(ref.size_bytes or 0),
        s3_bucket=ref.s3_bucket,
    )


def to_pb_artifact(ref: Optional[ArtifactRef]):
    if ref is None or ref.is_empty:
        return pb.ArtifactRef()
    return pb.ArtifactRef(
        backend=ref.backend or "",
        uri=ref.uri or "",
        sha256=ref.sha256 or "",
        size_bytes=int(ref.size_bytes or 0),
        s3_bucket=ref.s3_bucket or "",
    )


class NodeSupervisor:
    """One control agent per machine.

    The coordinator assigns whole proof jobs. Each proof executes in an isolated
    subprocess. ``max_active_jobs`` can impose an explicit whole-job concurrency
    baseline; otherwise local overlap is controlled by stage resource admission.
    """

    def __init__(self, runtime_config: RuntimeConfig, logger: Optional[logging.Logger] = None):
        self.cfg = runtime_config
        self.node_id = self.cfg.node.node_id or get_ip()
        self.logger = logger or logging.getLogger("node-supervisor")
        self.target = f"{self.cfg.coordinator.host}:{self.cfg.coordinator.port}"
        self.poll_interval_sec = self.cfg.node.job_poll_interval_sec
        self.phase_cfg = self.cfg.node.phase_admission
        self.grpc_max_message_bytes = self.cfg.coordinator.grpc_max_message_mb * 1024 * 1024

        self.cache = NodeCache(
            root_dir=self.cfg.node.local_cache_dir,
            max_bytes=self.cfg.node.local_cache_max_bytes,
            default_s3_bucket=self.cfg.artifact_store.s3_bucket
            or self.cfg.file_transfer.s3_bucket
            or "",
        )
        self.phase_admission_dir = Path(self.cfg.node.runtime_state_dir)
        self.phase_admission_dir.mkdir(parents=True, exist_ok=True)
        self.resource_profiles = HistoricalResourceProfiles(
            self.phase_admission_dir / "resource_profiles"
        )
        self.admission_probe = ResourceAdmissionController(
            root_dir=str(self.phase_admission_dir),
            setup_key="__supervisor__",
            enabled=self.phase_cfg.enabled,
            memory_reserve_gb=self.phase_cfg.memory_reserve_gb,
            cpu_budget_percent=self.phase_cfg.cpu_budget_percent,
            fallback_memory_gb=0.0,
            fallback_cpu_percent=0.0,
            profile_percentile=self.phase_cfg.profile_percentile,
        )

        self.active_jobs: Dict[str, threading.Thread] = {}
        self.active_tasks: Dict[str, LocalProverTask] = {}
        self.active_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.channel = None
        self.stub = None

    def connect(self) -> None:
        if self.channel:
            self.channel.close()
        self.channel = grpc.insecure_channel(
            self.target,
            options=[
                ("grpc.max_send_message_length", self.grpc_max_message_bytes),
                ("grpc.max_receive_message_length", self.grpc_max_message_bytes),
            ],
        )
        self.stub = pb_grpc.ZKJobServiceStub(self.channel)
        self.logger.info("Node %s connected to coordinator at %s", self.node_id, self.target)

    def reconnect(self) -> None:
        self.logger.info("Reconnecting to coordinator...")
        self.connect()
        time.sleep(self.poll_interval_sec)

    def safe_grpc_call(self, call_fn, action: str, max_retries: int = 3):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return call_fn()
            except grpc.RpcError as exc:
                last_error = exc
                self.logger.error(
                    "%s failed, attempt %d/%d: %s code=%s",
                    action,
                    attempt,
                    max_retries,
                    exc,
                    exc.code(),
                )
                self.reconnect()
            except Exception as exc:
                last_error = exc
                self.logger.error(
                    "%s failed, attempt %d/%d: %s",
                    action,
                    attempt,
                    max_retries,
                    exc,
                    exc_info=True,
                )
                time.sleep(self.poll_interval_sec)
        raise last_error

    def active_job_ids(self) -> list[str]:
        with self.active_lock:
            return sorted(self.active_jobs.keys())

    def active_job_count(self) -> int:
        with self.active_lock:
            return len(self.active_jobs)

    @staticmethod
    def _read_task_stage(task: LocalProverTask) -> str:
        status_file = Path(task.output_dir) / "status.txt"
        try:
            return status_file.read_text(encoding="utf-8").strip() or "STARTING"
        except OSError:
            return "STARTING"

    def active_phases(self) -> Dict[str, str]:
        with self.active_lock:
            items = list(self.active_tasks.items())
        return {job_id: self._read_task_stage(task) for job_id, task in items}

    def can_accept_more_work(self) -> bool:
        max_active_jobs = self.cfg.node.max_active_jobs
        if max_active_jobs is not None and self.active_job_count() >= max_active_jobs:
            return False

        memory = psutil.virtual_memory()
        available_gb = float(memory.available) / (1024**3)
        return available_gb > self.phase_cfg.memory_reserve_gb

    def build_node_status(self):
        memory = psutil.virtual_memory()
        cpu_total = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        cpu_percent = psutil.cpu_percent(interval=None)
        actual_available_gb = float(memory.available) / (1024**3)
        actual_cpu_available = max(0.0, float(cpu_total) * (1.0 - cpu_percent / 100.0))

        reservations = (
            self.admission_probe.current_reservations()
            if self.phase_cfg.enabled
            else {"memory_gb": 0.0, "cpu_percent": 0.0, "count": 0.0}
        )
        total_memory_gb = float(memory.total) / (1024**3)
        resource_available_gb = max(
            0.0,
            total_memory_gb
            - self.phase_cfg.memory_reserve_gb
            - float(reservations.get("memory_gb") or 0.0),
        )
        reported_available_gb = min(
            max(0.0, actual_available_gb - self.phase_cfg.memory_reserve_gb),
            resource_available_gb,
        )

        cpu_from_reservations = max(
            0.0,
            float(cpu_total)
            * (
                1.0
                - min(100.0, float(reservations.get("cpu_percent") or 0.0))
                / 100.0
            ),
        )
        reported_cpu_available = min(actual_cpu_available, cpu_from_reservations)

        return pb.NodeStatus(
            node_id=self.node_id,
            memory_total_gb=total_memory_gb,
            memory_available_gb=reported_available_gb,
            cpu_total_cores=float(cpu_total),
            cpu_available_cores=reported_cpu_available,
            cache_limit_gb=float(self.cfg.node.local_cache_max_bytes) / (1024**3),
            cache_used_gb=float(self.cache.total_bytes()) / (1024**3),
            active_job_ids=self.active_job_ids(),
            cached_setup_keys=self.cache.cached_setup_keys(),
        )

    def poll_for_job(self):
        return self.safe_grpc_call(
            lambda: self.stub.PollNode(self.build_node_status()),
            action="PollNode",
        )

    def submit_job_result(
        self,
        lease,
        status: str,
        artifact_manifest_ref: Optional[ArtifactRef] = None,
        message: str = "",
        metrics: Optional[Dict] = None,
    ) -> None:
        def _call():
            return self.stub.SubmitJobResult(
                pb.NodeJobResult(
                    node_id=self.node_id,
                    lease_id=lease.lease_id,
                    request_id=lease.request_id,
                    job_id=lease.job_id,
                    status=status,
                    message=message or "",
                    perf_metrics_json=json.dumps(metrics or {}),
                    artifact_manifest=to_pb_artifact(artifact_manifest_ref),
                )
            )

        self.safe_grpc_call(_call, action=f"SubmitJobResult({status})")
        self.logger.info("Submitted %s for job %s", status, lease.job_id)

    def send_heartbeat(self) -> None:
        phases = self.active_phases()

        def _call():
            return self.stub.SendNodeHeartbeat(
                pb.NodeHeartbeat(
                    node_id=self.node_id,
                    active_job_ids=self.active_job_ids(),
                    status="RUNNING",
                    message=json.dumps({"phases": phases}, sort_keys=True),
                )
            )

        self.safe_grpc_call(_call, action="SendNodeHeartbeat", max_retries=1)

    def heartbeat_loop(self) -> None:
        while not self.stop_event.wait(max(1, self.cfg.heartbeat.interval_sec)):
            try:
                self.send_heartbeat()
            except Exception:
                self.logger.warning("Node heartbeat failed", exc_info=True)

    def start_lease(self, lease) -> None:
        with self.active_lock:
            if lease.job_id in self.active_jobs:
                self.logger.warning("Ignoring duplicate lease for active job %s", lease.job_id)
                return
            thread = threading.Thread(
                target=self._run_lease_thread,
                args=(lease,),
                name=f"prover-controller-{lease.job_id[:8]}",
                daemon=False,
            )
            self.active_jobs[lease.job_id] = thread
            thread.start()
        self.logger.info(
            "Started prover subprocess controller for job %s lease=%s",
            lease.job_id,
            lease.lease_id,
        )

    def _run_lease_thread(self, lease) -> None:
        try:
            self.run_lease(lease)
        finally:
            with self.active_lock:
                self.active_jobs.pop(lease.job_id, None)
                self.active_tasks.pop(lease.job_id, None)

    def _run_prover_subprocess(self, task: LocalProverTask) -> ProverTaskResult:
        task_file = Path(task.tmp_dir) / "prover_task.json"
        result_file = Path(task.tmp_dir) / "prover_result.json"
        log_file = Path(task.output_dir) / "prover_process.log"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        with open(task_file, "w", encoding="utf-8") as file:
            json.dump(task.to_dict(), file, indent=2)

        started = time.perf_counter()
        with open(log_file, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "zkinfer.runtime.node.prover_process",
                    "--task_file",
                    str(task_file),
                    "--result_file",
                    str(result_file),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            return_code = proc.wait()
        wall_time = time.perf_counter() - started

        if not result_file.exists():
            raise RuntimeError(
                f"Prover subprocess exited with code {return_code} without a result; see {log_file}"
            )
        with open(result_file, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not payload.get("ok"):
            raise RuntimeError(
                f"Prover subprocess failed ({payload.get('error_type', 'error')}): "
                f"{payload.get('error', 'unknown error')}; see {log_file}"
            )
        result = ProverTaskResult.from_dict(payload["result"])
        result.metrics["prover_subprocess_wall_time(s)"] = wall_time
        result.metrics["prover_subprocess_pid"] = int(payload.get("pid") or 0)
        return result

    def run_lease(self, lease) -> None:
        local_task = None
        setup_pinned = False
        try:
            local_task = self.prepare_local_task(lease)
            with self.active_lock:
                self.active_tasks[lease.job_id] = local_task

            if self.cache.setup_ready(local_task.setup_key):
                self.cache.pin_setup(local_task.setup_key)
                setup_pinned = True
                result = self._run_prover_subprocess(local_task)
                setup_publish_metrics = {}
            else:
                with self.cache.setup_guard(local_task.setup_key, pin=True):
                    if self.cache.setup_ready(local_task.setup_key):
                        result = self._run_prover_subprocess(local_task)
                        setup_publish_metrics = {}
                    else:
                        result = self._run_prover_subprocess(local_task)
                        setup_publish_metrics = self.cache.publish_setup_cache_with_metrics(
                            local_task.setup_cache_dir,
                            from_pb_artifact(lease.setup_cache),
                        )

            self.resource_profiles.update_from_metrics(
                local_task.setup_key,
                result.metrics,
            )

            metrics = {
                "node_id": self.node_id,
                "node_cache_bytes": self.cache.total_bytes(),
                **result.metrics,
                **self._flatten_setup_publish_metrics(setup_publish_metrics),
            }
            self.submit_job_result(
                lease=lease,
                status="COMPLETED",
                artifact_manifest_ref=result.artifact_manifest_ref,
                message=result.message,
                metrics=metrics,
            )
        except Exception as exc:
            self.logger.error(
                "Job %s failed on node %s: %s",
                lease.job_id,
                self.node_id,
                exc,
                exc_info=True,
            )
            self.submit_job_result(
                lease=lease,
                status="FAILED",
                message=str(exc),
                metrics={"node_id": self.node_id},
            )
        finally:
            if setup_pinned and local_task:
                self.cache.unpin_setup(local_task.setup_key)
            if local_task and os.path.exists(local_task.tmp_dir):
                shutil.rmtree(local_task.tmp_dir, ignore_errors=True)

    def prepare_local_task(self, lease) -> LocalProverTask:
        prepare_started = time.perf_counter()
        request_dir = Path(self.cfg.node.runs_dir) / lease.request_id
        tmp_dir = request_dir / "tmp" / lease.job_id
        output_dir = request_dir / "artifacts" / "jobs" / lease.job_id
        tmp_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_metrics = {}
        model_materialize_time_s = 0.0
        if self.cfg.jobs.transfer_only:
            model_path = ""
        else:
            model_path, model_metrics = self.cache.materialize_with_metrics(
                from_pb_artifact(lease.model), filename_hint="model.onnx"
            )
            model_materialize_time_s = float(
                model_metrics.get("materialize_time(s)", 0.0) or 0.0
            )

        input_path, input_metrics = self.cache.materialize_with_metrics(
            from_pb_artifact(lease.input), filename_hint="input.json"
        )
        input_materialize_time_s = float(
            input_metrics.get("materialize_time(s)", 0.0) or 0.0
        )

        setup_key = lease.setup_key or lease.job_id
        setup_cache_dir, setup_metrics = self.cache.restore_setup_cache_with_metrics(
            from_pb_artifact(lease.setup_cache), setup_key
        )
        output_ref = from_pb_artifact(lease.output_prefix)
        transfer_metrics = {
            "model": model_metrics,
            "input": input_metrics,
            "setup": setup_metrics,
            "prepare_local_task_time(s)": time.perf_counter() - prepare_started,
        }

        return LocalProverTask(
            node_id=self.node_id,
            request_id=lease.request_id,
            job_id=lease.job_id,
            setup_key=setup_key,
            model_path=model_path,
            input_path=input_path,
            setup_cache_dir=setup_cache_dir,
            tmp_dir=str(tmp_dir),
            output_dir=str(output_dir),
            output_prefix=output_ref.uri or str(output_dir),
            output_backend=output_ref.backend or self.cfg.artifact_store.backend,
            output_s3_bucket=output_ref.s3_bucket
            or self.cfg.artifact_store.s3_bucket
            or "",
            transfer_only=self.cfg.jobs.transfer_only,
            resource_monitor_interval_sec=self.cfg.node.resource_monitor_interval_sec,
            model_materialize_time_s=model_materialize_time_s,
            input_materialize_time_s=input_materialize_time_s,
            transfer_metrics=transfer_metrics,
            phase_admission_dir=str(self.phase_admission_dir),
            phase_admission_enabled=self.phase_cfg.enabled,
            phase_memory_reserve_gb=self.phase_cfg.memory_reserve_gb,
            phase_cpu_budget_percent=self.phase_cfg.cpu_budget_percent,
            phase_profile_percentile=self.phase_cfg.profile_percentile,
            phase_fallback_cpu_percent=self.phase_cfg.fallback_cpu_percent,
            predicted_peak_memory_gb=float(lease.predicted_peak_memory_gb or 0.0),
        )

    @staticmethod
    def _flatten_setup_publish_metrics(metrics: Dict) -> Dict[str, float | int]:
        return {
            "transfer_setup_bytes_uploaded": int(metrics.get("bytes_uploaded", 0) or 0),
            "transfer_setup_publish_time(s)": float(
                metrics.get("publish_time(s)", 0.0) or 0.0
            ),
            "transfer_pk_compress_time(s)": float(
                metrics.get("pk_compress_time(s)", 0.0) or 0.0
            ),
            "transfer_setup_publish_eviction_count": int(
                metrics.get("eviction_count", 0) or 0
            ),
            "transfer_setup_publish_evicted_bytes": int(
                metrics.get("evicted_bytes", 0) or 0
            ),
        }

    def _join_active_threads(self) -> None:
        while True:
            with self.active_lock:
                threads = list(self.active_jobs.values())
            if not threads:
                return
            for thread in threads:
                thread.join(timeout=1)

    def run(self) -> None:
        self.connect()
        heartbeat_thread = threading.Thread(
            target=self.heartbeat_loop,
            name="node-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            while True:
                if not self.can_accept_more_work():
                    time.sleep(self.poll_interval_sec)
                    continue
                lease = self.poll_for_job()
                if not lease.job_available:
                    time.sleep(self.poll_interval_sec)
                    continue
                self.start_lease(lease)
        finally:
            self.stop_event.set()
            heartbeat_thread.join(timeout=5)
            self._join_active_threads()
            if self.channel:
                self.channel.close()


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    runtime_config = build_runtime_config(cfg=cfg, process_role="node")
    node_id = cfg.node.node_id or get_ip()
    logger = setup_logger(
        "node-supervisor",
        log_file=f"{runtime_config.node.logs_dir}/node_{node_id}.log",
    )
    logger.info("Starting NodeSupervisor...")
    NodeSupervisor(runtime_config=runtime_config, logger=logger).run()


if __name__ == "__main__":
    main()
