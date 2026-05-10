import json
import logging
import os
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

from zkinfer.backends.ezkl.metrics import (
    read_csv_first_row,
    summarize_fft_report,
    summarize_msm_report,
)
from zkinfer.backends.ezkl.prover import EZKLProofStages

from zkinfer.profiling.resource_parser import parse_resource_usage_file

from zkinfer.storage.s3 import download_file

from zkinfer.utils.utils import get_ip

@dataclass
class LocalJobPaths:
    model_path: str
    input_path: str
    tmp_dir: str
    cache_prefix: str
    s3_read_time: float = 0.0


class ZKProofWorker:
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger("worker")

        self.worker_id = cfg.worker.worker_id or get_ip()
        self.target = f"{cfg.coordinator.host}:{cfg.coordinator.port}"

        self.poll_interval_sec = cfg.worker.job_poll_interval_sec
        self.resource_monitor_interval_sec = cfg.worker.resource_monitor_interval_sec
        self.send_proofs_to_coordinator = cfg.worker.send_proofs_to_coordinator

        self.grpc_max_message_bytes = cfg.coordinator.grpc_max_message_mb * 1024 * 1024

        self.channel = None
        self.stub = None

        self.num_cpus_physical = psutil.cpu_count(logical=False)
        self.num_cpus_logical = psutil.cpu_count(logical=True)
        self.mem_gb = round(psutil.virtual_memory().total / 1e9, 2)

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
    
    def get_next_job(self):
        return self.safe_grpc_call(
            lambda: self.stub.GetNextJob(
                pb.WorkerRequest(worker_id=self.worker_id)
            ),
            action="GetNextJob",
        )

    def submit_job_result(
        self,
        job_id: str,
        status: str,
        proof: Optional[bytes] = None,
        message: Optional[str] = None,
        perf_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _call():
            return self.stub.SubmitJobResult(
                pb.JobResult(
                    job_id=job_id,
                    proof=proof or b"",
                    status=status,
                    perf_metrics=json.dumps(perf_metrics) if perf_metrics else "",
                    message=message or "",
                )
            )

        self.safe_grpc_call(_call, action=f"SubmitJobResult({status})")
        self.logger.info("Submitted final result for job %s: %s", job_id, status)

    def prepare_local_paths(self, assignment) -> LocalJobPaths:
        job_id = assignment.job_id
        model_path = assignment.model_path
        input_path = assignment.input_path

        tmp_dir = os.path.join(self.cfg.paths.tmp_dir, job_id)
        os.makedirs(tmp_dir, exist_ok=True)

        cache_prefix = os.path.dirname(model_path)
        s3_read_time = 0.0

        if assignment.file_transfer_type != "s3":
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input path does not exist: {input_path}")

            return LocalJobPaths(
                model_path=model_path,
                input_path=input_path,
                tmp_dir=tmp_dir,
                cache_prefix=cache_prefix,
                s3_read_time=0.0,
            )

        s3_bucket = assignment.file_transfer_s3_bucket

        local_model_path = os.path.join(tmp_dir, os.path.basename(model_path))
        local_input_path = os.path.join(tmp_dir, "input.json")

        start = time.perf_counter()
        download_file(s3_bucket, model_path, local_model_path)
        s3_read_time += time.perf_counter() - start

        start = time.perf_counter()
        download_file(s3_bucket, input_path, local_input_path)
        s3_read_time += time.perf_counter() - start

        return LocalJobPaths(
            model_path=local_model_path,
            input_path=local_input_path,
            tmp_dir=tmp_dir,
            cache_prefix=cache_prefix,
            s3_read_time=s3_read_time,
        )

    def start_monitoring_processes(
        self,
        job_id: str,
        tmp_dir: str,
        status_file: str,
    ):
        worker_pid = str(os.getpid())
        resource_usage_file = os.path.join(tmp_dir, "resource_usage.log")

        resource_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "zkinfer.profiling.resource_logger",
                "--log_file",
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
        tmp_dir: str,
        resource_usage_file: str,
        proof_stages: EZKLProofStages,
        ezkl_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            max_process_mem, max_system_mem, avg_process_cpu, avg_system_cpu = (
                parse_resource_usage_file(resource_usage_file)
            )
        except Exception as exc:
            self.logger.error("Failed to parse resource usage file: %s", exc, exc_info=True)
            max_process_mem = None
            max_system_mem = None
            avg_process_cpu = None
            avg_system_cpu = None

        metrics = {
            "worker_id": self.worker_id,
            "max_process_memory(GB)": max_process_mem,
            "max_system_memory(GB)": max_system_mem,
            "avg_process_cpu(%)": avg_process_cpu,
            "avg_system_cpu(%)": avg_system_cpu,
            "pk_file_size(GB)": proof_stages.get_pk_file_size_gb(),
            "vk_file_size(GB)": proof_stages.get_vk_file_size_gb(),
        }

        circuit_info = read_csv_first_row(os.path.join(tmp_dir, "halo2_circuit.csv"))
        prover_info_cpu = read_csv_first_row(os.path.join(tmp_dir, "halo2_prover_cpu.csv"))
        fft_summary = summarize_fft_report(os.path.join(tmp_dir, "halo2_ffts.csv"))
        msm_summary = summarize_msm_report(os.path.join(tmp_dir, "halo2_msms.csv"))

        return {
            **metrics,
            **circuit_info,
            **prover_info_cpu,
            **fft_summary,
            **msm_summary,
            **ezkl_metrics,
        }

    def read_proof_bytes(self, proof_path: str) -> bytes:
        if not self.send_proofs_to_coordinator:
            return b""

        if not os.path.exists(proof_path):
            raise FileNotFoundError(f"Proof file does not exist: {proof_path}")

        with open(proof_path, "rb") as file:
            return file.read()

    def run_assignment(self, assignment) -> None:
        job_id = assignment.job_id
        local_paths = None
        heartbeat_proc = None
        resource_proc = None

        try:
            local_paths = self.prepare_local_paths(assignment)

            status_file = os.path.join(local_paths.tmp_dir, "status.txt")
            heartbeat_proc, resource_proc, resource_usage_file = self.start_monitoring_processes(
                job_id=job_id,
                tmp_dir=local_paths.tmp_dir,
                status_file=status_file,
            )

            os.environ["EZKL_LOG_DIR"] = local_paths.tmp_dir

            proof_stages = EZKLProofStages(
                input_data_path=local_paths.input_path,
                onnx_model_path=local_paths.model_path,
                proving_cache_enabled=assignment.proving_cache_enabled,
                proving_cache_overwrite=assignment.proving_cache_overwrite,
                proving_cache_type=assignment.proving_cache_type,
                proving_cache_root_dir=assignment.proving_cache_root_dir,
                proving_cache_s3_bucket=assignment.proving_cache_s3_bucket or None,
                proving_cache_s3_prefix=assignment.proving_cache_s3_prefix,
                status_file=status_file,
                local_tmp_dir=local_paths.tmp_dir,
                logger=self.logger,
            )

            try:
                ezkl_metrics = proof_stages.run_all(setup_only=False)
            except Exception as exc:
                self.logger.error("Proof stages failed for job %s: %s", job_id, exc, exc_info=True)
                self.submit_job_result(
                    job_id=job_id,
                    status="FAILED",
                    message=f"Proof stages failed: {exc}",
                )
                return

            metrics = self.collect_metrics(
                tmp_dir=local_paths.tmp_dir,
                resource_usage_file=resource_usage_file,
                proof_stages=proof_stages,
                ezkl_metrics=ezkl_metrics,
            )

            proof_bytes = self.read_proof_bytes(proof_stages.proof_path)

            self.submit_job_result(
                job_id=job_id,
                status="COMPLETED",
                proof=proof_bytes,
                perf_metrics=metrics,
                message="Proof computation completed successfully",
            )

            self.logger.info("Completed job %s", job_id)

        except Exception as exc:
            self.logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
            self.submit_job_result(
                job_id=job_id,
                status="FAILED",
                message=str(exc),
            )

        finally:
            self.stop_processes(heartbeat_proc, resource_proc)

            if local_paths and os.path.exists(local_paths.tmp_dir):
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


def setup_worker_logger(cfg: DictConfig) -> logging.Logger:
    os.makedirs(cfg.paths.logs_dir, exist_ok=True)
    log_file = os.path.join(cfg.paths.logs_dir, "worker.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger("worker")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_worker_logger(cfg)
    logger.info("Starting ZKProofWorker...")

    worker = ZKProofWorker(cfg, logger)
    worker.run()


if __name__ == "__main__":
    main()