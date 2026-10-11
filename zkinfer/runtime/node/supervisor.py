from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
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
from zkinfer.runtime.node.prover_task import LocalProverTask, ProverTaskRunner
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
    """One node-level agent per machine or instance.

    The supervisor owns local cache, resource reporting, admission control, and
    local prover subprocesses. Prover workers themselves are stateless local
    tasks that only receive local filesystem paths.
    """

    def __init__(self, runtime_config: RuntimeConfig, logger: Optional[logging.Logger] = None):
        self.cfg = runtime_config
        self.node_id = self.cfg.node.node_id or get_ip()
        self.logger = logger or logging.getLogger("node-supervisor")
        self.target = f"{self.cfg.coordinator.host}:{self.cfg.coordinator.port}"
        self.poll_interval_sec = self.cfg.node.job_poll_interval_sec
        self.max_concurrent_jobs = max(1, int(self.cfg.node.max_concurrent_jobs or 1))
        self.grpc_max_message_bytes = self.cfg.coordinator.grpc_max_message_mb * 1024 * 1024

        self.cache = NodeCache(
            root_dir=self.cfg.node.local_cache_dir,
            max_bytes=self.cfg.node.local_cache_max_bytes,
            default_s3_bucket=self.cfg.artifact_store.s3_bucket or self.cfg.file_transfer.s3_bucket or "",
        )
        self.runner = ProverTaskRunner(logger=self.logger)
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_jobs,
            thread_name_prefix="prover-task",
        )
        self.active_jobs: Dict[str, Future] = {}
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

    def can_accept_more_work(self) -> bool:
        return self.active_job_count() < self.max_concurrent_jobs

    def build_node_status(self):
        memory = psutil.virtual_memory()
        cpu_total = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_available = max(0.0, float(cpu_total) * (1.0 - cpu_percent / 100.0))
        cache_used_bytes = self.cache.total_bytes()

        return pb.NodeStatus(
            node_id=self.node_id,
            memory_total_gb=float(memory.total) / 1e9,
            memory_available_gb=float(memory.available) / 1e9,
            cpu_total_cores=float(cpu_total),
            cpu_available_cores=cpu_available,
            cache_limit_gb=float(self.cfg.node.local_cache_max_bytes) / (1024 ** 3),
            cache_used_gb=float(cache_used_bytes) / (1024 ** 3),
            max_concurrent_jobs=self.max_concurrent_jobs,
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
        def _call():
            return self.stub.SendNodeHeartbeat(
                pb.NodeHeartbeat(
                    node_id=self.node_id,
                    active_job_ids=self.active_job_ids(),
                    status="RUNNING",
                    message="",
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
            future = self.executor.submit(self.run_lease, lease)
            self.active_jobs[lease.job_id] = future
            future.add_done_callback(lambda fut, job_id=lease.job_id: self._finish_active_job(job_id, fut))

        self.logger.info("Started local prover task for job %s lease=%s", lease.job_id, lease.lease_id)

    def _finish_active_job(self, job_id: str, future: Future) -> None:
        with self.active_lock:
            self.active_jobs.pop(job_id, None)

        try:
            future.result()
        except Exception:
            self.logger.error("Local prover task failed after result handling: %s", job_id, exc_info=True)

    def run_lease(self, lease) -> None:
        local_task = None
        try:
            local_task = self.prepare_local_task(lease)
            result = self.runner.run(local_task)

            setup_publish_metrics = {}
            if not self.cfg.jobs.transfer_only:
                setup_publish_metrics = self.cache.publish_setup_cache_with_metrics(
                    local_task.setup_cache_dir,
                    from_pb_artifact(lease.setup_cache),
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
            self.logger.error("Job %s failed on node %s: %s", lease.job_id, self.node_id, exc, exc_info=True)
            self.submit_job_result(
                lease=lease,
                status="FAILED",
                message=str(exc),
                metrics={"node_id": self.node_id},
            )

        finally:
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
                from_pb_artifact(lease.model),
                filename_hint="model.onnx",
            )
            model_materialize_time_s = float(model_metrics.get("materialize_time(s)", 0.0) or 0.0)

        input_path, input_metrics = self.cache.materialize_with_metrics(
            from_pb_artifact(lease.input),
            filename_hint="input.json",
        )
        input_materialize_time_s = float(input_metrics.get("materialize_time(s)", 0.0) or 0.0)

        setup_key = lease.setup_key or lease.job_id
        setup_cache_dir, setup_metrics = self.cache.hydrate_setup_cache_with_metrics(
            from_pb_artifact(lease.setup_cache),
            setup_key,
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
            output_s3_bucket=output_ref.s3_bucket or self.cfg.artifact_store.s3_bucket or "",
            transfer_only=self.cfg.jobs.transfer_only,
            resource_monitor_interval_sec=self.cfg.node.resource_monitor_interval_sec,
            model_materialize_time_s=model_materialize_time_s,
            input_materialize_time_s=input_materialize_time_s,
            transfer_metrics=transfer_metrics,
        )

    @staticmethod
    def _flatten_setup_publish_metrics(metrics: Dict) -> Dict[str, float | int]:
        return {
            "transfer_setup_bytes_uploaded": int(metrics.get("bytes_uploaded", 0) or 0),
            "transfer_setup_publish_time(s)": float(metrics.get("publish_time(s)", 0.0) or 0.0),
            "transfer_pk_compress_time(s)": float(metrics.get("pk_compress_time(s)", 0.0) or 0.0),
            "transfer_setup_publish_eviction_count": int(metrics.get("eviction_count", 0) or 0),
            "transfer_setup_publish_evicted_bytes": int(metrics.get("evicted_bytes", 0) or 0),
        }

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
            self.executor.shutdown(wait=True)
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
