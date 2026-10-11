import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import grpc
import hydra
import psutil
from omegaconf import DictConfig

import zkinfer.proto.zkservice_pb2 as pb
import zkinfer.proto.zkservice_pb2_grpc as pb_grpc
from zkinfer.backends.ezkl.ezkl_prover import EZKLProofStages
from zkinfer.config.runtime import RuntimeConfig, build_runtime_config
from zkinfer.profiling.msm_fft_parser import (
    read_csv_first_row,
    summarize_fft_report,
    summarize_msm_report,
)
from zkinfer.profiling.resource_parser import parse_resource_usage_file
from zkinfer.runtime.worker_cache import WorkerLocalCache
from zkinfer.storage.s3 import upload_file
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


@dataclass
class LocalJobPaths:
    model_path: str
    input_path: str
    tmp_dir: str
    artifact_dir: str
    setup_cache_dir: str
    output_prefix: str
    output_backend: str
    output_s3_bucket: str = ""
    model_materialize_time_s: float = 0.0
    input_materialize_time_s: float = 0.0


class ZKProofWorker:
    """Worker process that executes proof jobs assigned by the coordinator.

    gRPC is only the control plane. The worker moves model/input/setup/proof
    data directly through the data plane and keeps a bounded local hot cache.
    """

    def __init__(self, runtime_config: RuntimeConfig, logger: Optional[logging.Logger] = None):
        self.cfg = runtime_config
        self.logger = logger or logging.getLogger("worker")
        self.worker_id = self.cfg.worker.worker_id or get_ip()
        self.target = f"{self.cfg.coordinator.host}:{self.cfg.coordinator.port}"
        self.poll_interval_sec = self.cfg.worker.job_poll_interval_sec
        self.resource_monitor_interval_sec = self.cfg.worker.resource_monitor_interval_sec
        self.transfer_only = self.cfg.jobs.transfer_only
        self.grpc_max_message_bytes = self.cfg.coordinator.grpc_max_message_mb * 1024 * 1024
        self.channel = None
        self.stub = None
        self.mem_gb = round(psutil.virtual_memory().total / 1e9, 2)
        self.local_cache = WorkerLocalCache(
            root_dir=self.cfg.worker.local_cache_dir,
            max_bytes=self.cfg.worker.local_cache_max_bytes,
            default_s3_bucket=self.cfg.artifact_store.s3_bucket or self.cfg.proving_cache.s3_bucket or "",
        )

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
        self.logger.info("Connected to coordinator at %s", self.target)

    def reconnect(self) -> None:
        self.logger.info("Attempting to reconnect to coordinator...")
        self.connect()
        time.sleep(self.poll_interval_sec)

    def safe_grpc_call(self, call_fn, action: str, max_retries: int = 3):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return call_fn()
            except grpc.RpcError as exc:
                last_error = exc
                self.logger.error("%s failed, attempt %d/%d: %s code=%s", action, attempt, max_retries, exc, exc.code())
                self.reconnect()
            except Exception as exc:
                last_error = exc
                self.logger.error("%s failed, attempt %d/%d: %s", action, attempt, max_retries, exc, exc_info=True)
                time.sleep(self.poll_interval_sec)
        raise last_error

    def cached_artifact_keys(self) -> list[str]:
        return self.local_cache.cached_setup_keys()

    def get_next_job(self):
        memory = psutil.virtual_memory()
        reserved_gb = max(0.0, (memory.total - memory.available) / 1e9)
        return self.safe_grpc_call(
            lambda: self.stub.GetNextJob(
                pb.WorkerRequest(
                    worker_id=self.worker_id,
                    memory_budget_gb=self.mem_gb,
                    reserved_memory_gb=reserved_gb,
                    free_slots=1,
                    cached_artifact_keys=self.cached_artifact_keys(),
                )
            ),
            action="GetNextJob",
        )

    @staticmethod
    def artifact_ref(backend: str, uri: str, s3_bucket: str = "", sha256: str = "", size_bytes: int = 0):
        return pb.ArtifactRef(
            backend=backend or "",
            uri=uri or "",
            sha256=sha256 or "",
            size_bytes=int(size_bytes or 0),
            s3_bucket=s3_bucket or "",
        )

    def submit_job_result(
        self,
        job_id: str,
        request_id: str,
        status: str,
        artifact_manifest=None,
        message: Optional[str] = None,
        perf_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _call():
            return self.stub.SubmitJobResult(
                pb.JobResult(
                    worker_id=self.worker_id,
                    request_id=request_id,
                    job_id=job_id,
                    status=status,
                    message=message or "",
                    perf_metrics_json=json.dumps(perf_metrics) if perf_metrics else "",
                    artifact_manifest=artifact_manifest or pb.ArtifactRef(),
                )
            )

        self.safe_grpc_call(_call, action=f"SubmitJobResult({status})")
        self.logger.info("Submitted final result for job %s: %s", job_id, status)

    def prepare_local_paths(self, assignment, input_only: bool = False) -> LocalJobPaths:
        job_id = assignment.job_id
        worker_request_dir = Path(self.cfg.worker.runs_dir) / assignment.request_id
        tmp_dir = worker_request_dir / "tmp" / job_id
        artifact_dir = worker_request_dir / "artifacts" / "jobs" / job_id
        tmp_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_materialize_time_s = 0.0
        if input_only:
            model_path = ""
        else:
            start = time.perf_counter()
            model_path = self.local_cache.materialize(assignment.model, filename_hint="model.onnx")
            model_materialize_time_s = time.perf_counter() - start

        start = time.perf_counter()
        input_path = self.local_cache.materialize(assignment.input, filename_hint="input.json")
        input_materialize_time_s = time.perf_counter() - start

        setup_cache_dir = self.local_cache.hydrate_setup_cache(
            assignment.setup_cache,
            assignment.artifact_key or assignment.job_id,
        )

        output_ref = assignment.output_prefix
        output_prefix = output_ref.uri or str(artifact_dir)

        return LocalJobPaths(
            model_path=model_path,
            input_path=input_path,
            tmp_dir=str(tmp_dir),
            artifact_dir=str(artifact_dir),
            setup_cache_dir=setup_cache_dir,
            output_prefix=output_prefix,
            output_backend=output_ref.backend or self.cfg.artifact_store.backend,
            output_s3_bucket=output_ref.s3_bucket or self.cfg.artifact_store.s3_bucket or "",
            model_materialize_time_s=model_materialize_time_s,
            input_materialize_time_s=input_materialize_time_s,
        )

    def start_monitoring_processes(self, job_id: str, artifact_dir: str, status_file: str):
        worker_pid = str(os.getpid())
        resource_usage_file = os.path.join(artifact_dir, "resource_usage.csv")
        resource_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "zkinfer.profiling.resource_logger",
                "--output_file",
                resource_usage_file,
                "--pid",
                worker_pid,
                "--interval",
                str(self.resource_monitor_interval_sec),
            ]
        )
        heartbeat_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "zkinfer.runtime.heartbeat",
                "--target",
                self.target,
                "--worker_id",
                self.worker_id,
                "--job_id",
                job_id,
                "--status_file",
                status_file,
                "--parent_pid",
                worker_pid,
                "--interval_sec",
                str(self.cfg.heartbeat.interval_sec),
            ]
        )
        return heartbeat_proc, resource_proc, resource_usage_file

    def stop_processes(self, *processes) -> None:
        for proc in processes:
            if not proc:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()

    def collect_metrics(
        self,
        artifact_dir: str,
        resource_usage_file: str,
        proof_stages: EZKLProofStages,
        ezkl_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            resource_metrics = parse_resource_usage_file(resource_usage_file)
        except Exception as exc:
            self.logger.error("Failed to parse resource usage file: %s", exc, exc_info=True)
            resource_metrics = {
                "max_process_memory(GB)": None,
                "max_system_memory(GB)": None,
                "avg_process_cpu_raw(%)": None,
                "avg_process_cpu_machine(%)": None,
                "avg_system_cpu(%)": None,
            }

        return {
            "worker_id": self.worker_id,
            "worker_cache_bytes": self.local_cache.total_bytes(),
            **resource_metrics,
            "pk_file_size(GB)": proof_stages.get_pk_file_size_gb(),
            "vk_file_size(GB)": proof_stages.get_vk_file_size_gb(),
            **read_csv_first_row(os.path.join(artifact_dir, "halo2_circuit.csv")),
            **read_csv_first_row(os.path.join(artifact_dir, "halo2_prover_cpu.csv")),
            **summarize_fft_report(os.path.join(artifact_dir, "halo2_ffts.csv")),
            **summarize_msm_report(os.path.join(artifact_dir, "halo2_msms.csv")),
            **ezkl_metrics,
        }

    def build_proof_stages(self, local_paths: LocalJobPaths, status_file: str) -> EZKLProofStages:
        # EZKL sees a local filesystem setup cache. The WorkerLocalCache hydrates
        # it from shared storage before proving and publishes it back afterwards.
        return EZKLProofStages(
            input_data_path=local_paths.input_path,
            onnx_model_path=local_paths.model_path,
            artifact_dir=local_paths.artifact_dir,
            proving_cache_enabled=self.cfg.proving_cache.enabled,
            proving_cache_overwrite=self.cfg.proving_cache.overwrite,
            proving_cache_type="filesystem",
            proving_cache_root_dir=local_paths.setup_cache_dir,
            proving_cache_s3_bucket=None,
            proving_cache_s3_prefix=None,
            status_file=status_file,
            local_tmp_dir=local_paths.tmp_dir,
            logger=self.logger,
        )

    def _publish_artifact_file(self, local_path: str, local_paths: LocalJobPaths, artifact_name: str) -> str:
        if local_paths.output_backend == "filesystem":
            output_path = os.path.join(local_paths.output_prefix, artifact_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(local_path, output_path)
            return output_path

        if local_paths.output_backend == "s3":
            if not local_paths.output_s3_bucket:
                raise ValueError("S3 output artifact requires a bucket")
            key = f"{local_paths.output_prefix.rstrip('/')}/{artifact_name}"
            upload_file(local_path, local_paths.output_s3_bucket, key)
            return key

        raise ValueError(f"Unsupported output backend: {local_paths.output_backend}")

    def preserve_job_artifacts(self, assignment, local_paths: LocalJobPaths, proof_stages: EZKLProofStages):
        artifact_sources = {
            "proof.pf": proof_stages.proof_path,
            "witness.json": proof_stages.witness_path,
            "vk.json": proof_stages.vk_path,
            "settings.json": proof_stages.settings_path,
        }

        for artifact_name, source_path in artifact_sources.items():
            if not source_path or not os.path.exists(source_path):
                raise FileNotFoundError(f"Required artifact {artifact_name} was not generated: {source_path}")

        artifact_paths: Dict[str, str] = {}
        artifact_metadata = {}
        tmp_dir = Path(local_paths.tmp_dir).resolve()

        for artifact_name, source_path in artifact_sources.items():
            output_path = os.path.join(local_paths.artifact_dir, artifact_name)
            source = Path(source_path).resolve()
            try:
                source.relative_to(tmp_dir)
                source_is_temporary = True
            except ValueError:
                source_is_temporary = False

            if source_is_temporary:
                shutil.move(str(source), output_path)
            else:
                shutil.copy2(str(source), output_path)

            published_uri = self._publish_artifact_file(output_path, local_paths, artifact_name)
            artifact_paths[artifact_name] = output_path
            artifact_metadata[artifact_name] = {
                "file": artifact_name,
                "local_path": output_path,
                "published_uri": published_uri,
                "size_bytes": os.path.getsize(output_path),
            }

        setup_cache_dir = local_paths.setup_cache_dir.rstrip("/")
        setup_cache_artifacts = {
            "settings": f"{setup_cache_dir}/settings.json",
            "compiled_circuit": f"{setup_cache_dir}/network.compiled",
            "proving_key_local_hot_cache": f"{setup_cache_dir}/pk.json",
            "proving_key_shared_compressed": f"{assignment.setup_cache.uri.rstrip('/')}/pk.json.gz",
            "verification_key": f"{setup_cache_dir}/vk.json",
        }

        manifest = {
            "request_id": assignment.request_id,
            "job_id": assignment.job_id,
            "worker_id": self.worker_id,
            "storage_backend": local_paths.output_backend,
            "s3_bucket": local_paths.output_s3_bucket,
            "model_source": assignment.model.uri,
            "input_source": assignment.input.uri,
            "setup_cache_dir": setup_cache_dir,
            "shared_setup_cache_prefix": assignment.setup_cache.uri,
            "setup_cache_artifacts": setup_cache_artifacts,
            "srs_path": proof_stages.srs_path,
            "artifacts": artifact_metadata,
        }

        manifest_path = os.path.join(local_paths.artifact_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=4)

        published_manifest = self._publish_artifact_file(manifest_path, local_paths, "manifest.json")
        manifest_ref = self.artifact_ref(
            backend=local_paths.output_backend,
            uri=published_manifest,
            s3_bucket=local_paths.output_s3_bucket,
            size_bytes=os.path.getsize(manifest_path),
        )
        self.logger.info(
            "Preserved and published proof artifacts for job %s to %s:%s",
            assignment.job_id,
            local_paths.output_backend,
            local_paths.output_prefix,
        )
        return artifact_paths, manifest_ref

    def run_assignment(self, assignment) -> None:
        job_id = assignment.job_id
        request_id = assignment.request_id
        local_paths = None
        heartbeat_proc = None
        resource_proc = None

        try:
            local_paths = self.prepare_local_paths(assignment, input_only=self.transfer_only)

            if self.transfer_only:
                parse_start = time.perf_counter()
                with open(local_paths.input_path, "r", encoding="utf-8") as file:
                    json.load(file)
                input_parse_time = time.perf_counter() - parse_start

                metrics = {
                    "worker_id": self.worker_id,
                    "transfer_only": True,
                    "model_materialize_time(s)": 0.0,
                    "input_materialize_time(s)": local_paths.input_materialize_time_s,
                    "input_parse_time(s)": input_parse_time,
                    "input_transfer_bytes": os.path.getsize(local_paths.input_path),
                    "worker_cache_bytes": self.local_cache.total_bytes(),
                }
                self.submit_job_result(
                    job_id=job_id,
                    request_id=request_id,
                    status="COMPLETED",
                    perf_metrics=metrics,
                    message="Transfer-only benchmark completed",
                )
                self.logger.info("Completed transfer-only job %s", job_id)
                return

            status_file = os.path.join(local_paths.artifact_dir, "status.txt")
            heartbeat_proc, resource_proc, resource_usage_file = self.start_monitoring_processes(
                job_id=job_id,
                artifact_dir=local_paths.artifact_dir,
                status_file=status_file,
            )

            proof_stages = self.build_proof_stages(local_paths=local_paths, status_file=status_file)
            ezkl_metrics, _ = proof_stages.run_all(setup_only=False)

            self.local_cache.publish_setup_cache(local_paths.setup_cache_dir, assignment.setup_cache)

            metrics = self.collect_metrics(
                artifact_dir=local_paths.artifact_dir,
                resource_usage_file=resource_usage_file,
                proof_stages=proof_stages,
                ezkl_metrics={
                    **ezkl_metrics,
                    "model_materialize_time(s)": local_paths.model_materialize_time_s,
                    "input_materialize_time(s)": local_paths.input_materialize_time_s,
                },
            )

            _, manifest_ref = self.preserve_job_artifacts(
                assignment=assignment,
                local_paths=local_paths,
                proof_stages=proof_stages,
            )

            self.submit_job_result(
                job_id=job_id,
                request_id=request_id,
                status="COMPLETED",
                artifact_manifest=manifest_ref,
                perf_metrics=metrics,
                message="Proof computation completed successfully",
            )
            self.logger.info("Completed job %s", job_id)

        except Exception as exc:
            self.logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
            self.submit_job_result(job_id=job_id, request_id=request_id, status="FAILED", message=str(exc))

        finally:
            self.stop_processes(heartbeat_proc, resource_proc)
            if local_paths and os.path.exists(local_paths.tmp_dir):
                self.logger.info("Cleaning up temporary directory: %s", local_paths.tmp_dir)
                shutil.rmtree(local_paths.tmp_dir, ignore_errors=True)

    def run_once(self) -> None:
        assignment = self.get_next_job()
        if not assignment.job_available:
            self.logger.info("No jobs available. Sleeping...")
            time.sleep(self.poll_interval_sec)
            return
        self.logger.info("Received job %s", assignment.job_id)
        self.run_assignment(assignment)

    def run(self) -> None:
        self.connect()
        while True:
            self.run_once()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    runtime_config = build_runtime_config(cfg=cfg, process_role="worker")
    worker_id = cfg.worker.worker_id or get_ip()
    logger = setup_logger("worker", log_file=f"{runtime_config.worker.logs_dir}/worker_{worker_id}.log")
    logger.info("Starting ZKProofWorker...")
    worker = ZKProofWorker(runtime_config=runtime_config, logger=logger)
    worker.run()


if __name__ == "__main__":
    main()
