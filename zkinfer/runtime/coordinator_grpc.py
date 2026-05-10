import json
import logging
import os
import sys
import time
from concurrent import futures
from typing import Optional

import grpc
import hydra
from omegaconf import DictConfig, OmegaConf

import zkinfer.proto.zkservice_pb2 as pb2
import zkinfer.proto.zkservice_pb2_grpc as pb_grpc
from zkinfer.config.runtime import build_runtime_config
from zkinfer.runtime.coordinator import Coordinator


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
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_config(cfg: DictConfig) -> None:
    if cfg.file_transfer.type == "s3" and not cfg.file_transfer.s3_bucket:
        raise ValueError("file_transfer.s3_bucket must be set when file_transfer.type=s3")

    if cfg.backend.proving_cache.type == "s3" and not cfg.backend.proving_cache.s3_bucket:
        raise ValueError(
            "backend.proving_cache.s3_bucket must be set when "
            "backend.proving_cache.type=s3"
        )


class ZKCoordinatorGrpcService(pb_grpc.ZKJobServiceServicer):
    """Thin gRPC adapter around the core Coordinator."""

    def __init__(
        self,
        coordinator: Coordinator,
        logger: Optional[logging.Logger] = None,
    ):
        self.coordinator = coordinator
        self.logger = logger or logging.getLogger(__name__)

    def SubmitInferenceRequest(self, request, context):
        try:
            request_id = self.coordinator.submit_request(
                name=request.name,
                onnx_model_path=request.onnx_model_path,
                input_data_path=request.input_data_path,
                split_mode=request.split_mode,
                ops_per_chunk=request.ops_per_chunk,
                schedule=request.schedule,
            )
            return pb2.InferenceRequestAck(request_id=request_id)

        except Exception as exc:
            self._set_error(
                context,
                grpc.StatusCode.INTERNAL,
                exc,
                "Error submitting inference request",
            )
            return pb2.InferenceRequestAck(request_id="")

    def GetNextJob(self, request, context):
        try:
            job = self.coordinator.get_next_job()
            if job is None:
                return pb2.JobAssignment(job_available=False)

            return pb2.JobAssignment(
                job_id=job.job_id,
                model_path=job.model_path,
                input_path=job.input_path,
                job_available=True,
                file_transfer_type=self.coordinator.file_transfer.type,
                file_transfer_root_dir=self.coordinator.file_transfer.root_dir,
                file_transfer_s3_bucket=self.coordinator.file_transfer.s3_bucket or "",
                file_transfer_s3_prefix=self.coordinator.file_transfer.s3_prefix,
                proving_cache_enabled=self.coordinator.proving_cache.enabled,
                proving_cache_type=self.coordinator.proving_cache.type,
                proving_cache_overwrite=self.coordinator.proving_cache.overwrite,
                proving_cache_root_dir=self.coordinator.proving_cache.root_dir,
                proving_cache_s3_bucket=self.coordinator.proving_cache.s3_bucket or "",
                proving_cache_s3_prefix=self.coordinator.proving_cache.s3_prefix,
            )

        except Exception as exc:
            self._set_error(
                context,
                grpc.StatusCode.INTERNAL,
                exc,
                "Error fetching next job",
            )
            return pb2.JobAssignment(job_available=False)

    def SubmitJobResult(self, request, context):
        try:
            perf_metrics = json.loads(request.perf_metrics) if request.perf_metrics else None

            ok = self.coordinator.submit_job_result(
                job_id=request.job_id,
                zk_proof=request.proof,
                status=request.status,
                perf_metrics=perf_metrics,
                message=request.message,
            )
            return pb2.StatusAck(ok=ok)

        except json.JSONDecodeError as exc:
            self._set_error(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                exc,
                "Invalid perf_metrics JSON",
            )
            return pb2.StatusAck(ok=False)

        except Exception as exc:
            self._set_error(
                context,
                grpc.StatusCode.INTERNAL,
                exc,
                "Error submitting job result",
            )
            return pb2.StatusAck(ok=False)

    def SendHeartbeat(self, request, context):
        try:
            self.coordinator.record_heartbeat(
                worker_id=request.worker_id,
                job_id=request.job_id,
                status=request.status,
                message=request.message,
            )
            return pb2.HeartbeatAck(success=True)

        except Exception as exc:
            self._set_error(
                context,
                grpc.StatusCode.INTERNAL,
                exc,
                "Error recording heartbeat",
            )
            return pb2.HeartbeatAck(success=False)

    def _set_error(self, context, code, exc: Exception, message: str) -> None:
        self.logger.error("%s: %s", message, exc, exc_info=True)
        context.set_code(code)
        context.set_details(f"{message}: {exc}")


def create_grpc_server(cfg: DictConfig, service: ZKCoordinatorGrpcService) -> grpc.Server:
    max_message_bytes = cfg.coordinator.grpc_max_message_mb * 1024 * 1024

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=cfg.coordinator.grpc_thread_pool_size
        ),
        options=[
            ("grpc.max_send_message_length", max_message_bytes),
            ("grpc.max_receive_message_length", max_message_bytes),
        ],
    )

    pb_grpc.add_ZKJobServiceServicer_to_server(service, server)
    return server


@hydra.main(config_path="../config", config_name="config", version_base=None)
def serve(cfg: DictConfig):
    validate_config(cfg)

    os.makedirs(cfg.paths.logs_dir, exist_ok=True)
    log_file = os.path.join(cfg.paths.logs_dir, "coordinator.log")
    logger = setup_logger("coordinator", log_file)

    logger.info("Starting ZK Coordinator gRPC Service")
    logger.debug("Loaded config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    runtime_config = build_runtime_config(cfg)

    coordinator = Coordinator(
        config=runtime_config,
        logger=logger,
    )

    service = ZKCoordinatorGrpcService(
        coordinator=coordinator,
        logger=logger,
    )

    server = create_grpc_server(cfg, service)

    bind_address = f"{cfg.coordinator.host}:{cfg.coordinator.port}"
    server.add_insecure_port(bind_address)
    server.start()

    logger.info("Coordinator gRPC server running at %s", bind_address)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.warning("Shutting down coordinator...")
        server.stop(0)


if __name__ == "__main__":
    serve()