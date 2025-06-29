import grpc
from concurrent import futures
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import zkservice_pb2 as pb2
import zkservice_pb2_grpc as pb_grpc
from zkInfer.job_manager import InferenceRequestManager
import logging
import sys
import json


def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up a logger that logs to both console and file (if given)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
    # logger.propagate = False  
    # Prevent duplicated logs if called multiple times
    if not logger.hasHandlers():
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.stream.reconfigure(encoding='utf-8')  # Python 3.7+

        logger.addHandler(ch)

        # File handler
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


class ZKJobDispatcher(pb_grpc.ZKJobServiceServicer):
    def __init__(self, 
                 s3_bucket, 
                 cache_setup,
                 cache_backend,
                 overwrite_cache,
                 data_exchange_backend='local',
                num_prover_workers=1, 
                                 logger=None, 

    ):
        self.logger = logger
        self.s3_bucket = s3_bucket
        self.cache_setup = cache_setup
        self.overwrite_cache = overwrite_cache
        self.num_prover_workers = num_prover_workers
        self.cache_backend = cache_backend
        self.data_exchange_backend = data_exchange_backend
        self.manager = InferenceRequestManager(logger=logger)

    # API for submitting new inference requests
    def SubmitInferenceRequest(self, request, context):
        try:
            # if not request.name:
            #     raise ValueError("Request name cannot be empty")
            # if not request.onnx_model_path:
            #     raise ValueError("ONNX model path cannot be empty")
            # if not request.input_data_path:
            #     raise ValueError("Input data path cannot be empty")
            # if request.split_mode not in ["auto", "fixed"]:
            #     raise ValueError("Invalid split mode. Must be 'auto' or 'fixed'")
            # if request.split_mode == "fixed" and request.ops_per_chunk <= 0:
            #     raise ValueError("ops_per_chunk must be greater than 0 for fixed split mode")
            request_id = self.manager.submit_request(
                name=request.name,
                onnx_model_path=request.onnx_model_path,
                input_data_path=request.input_data_path,
                split_mode=request.split_mode,
                ops_per_chunk=request.ops_per_chunk,
                logger=self.logger,
                num_prover_workers=self.num_prover_workers,
                data_exchange_backend=self.data_exchange_backend,
                s3_bucket=self.s3_bucket,
                cache_setup=self.cache_setup,
                overwrite_cache=self.overwrite_cache ,
                cache_backend=self.cache_backend,
            )
            return pb2.InferenceRequestAck(request_id=request_id)
        except Exception as e:
            self.logger.error(f"❌ Error submitting inference request: {str(e)}", exc_info=True)
            context.set_details(f"Error submitting inference request: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.InferenceRequestAck(request_id="")
    
    # Worker pulls a job
    def GetNextJob(self, request, context):
        try:
            job = self.manager.get_next_job()
            if job:
                # Convert Job to proto message
                return pb2.JobAssignment(
                    job_id=job.job_id,
                    model_path=job.model_path,
                    input_path=job.input_path,
                    s3_bucket=self.s3_bucket,
                    cache_setup=self.cache_setup,
                    overwrite_cache=self.overwrite_cache,
                    cache_backend=self.cache_backend,
                    share_data_via_s3=True if self.data_exchange_backend == "s3" else False,
                    job_available=True)
            else:
                return pb2.JobAssignment(job_available=False)
        except Exception as e:
            self.logger.error(f"❌ Error fetching next job: {str(e)}", exc_info=True)
            context.set_details(f"Error fetching next job: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.JobAssignment(job_available=False)
        

    # Worker submits results
    def SubmitJobResult(self, request, context):
        try:
            ok = self.manager.submit_job_result(
                job_id=request.job_id,
                zk_proof=request.proof,
                status=request.status,
                perf_metrics=json.loads(request.perf_metrics) if request.perf_metrics else None,
                message=request.message,
            )
            return pb2.StatusAck(ok=ok)
        except Exception as e:
            self.logger.error(f"❌ Error submitting job result: {str(e)}", exc_info=True)
            context.set_details(f"Error submitting job result: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.StatusAck(ok=False)
    
    def SendHeartbeat(self, request, context):
        try:
            self.manager.record_heartbeat(
                worker_id=request.worker_id,
                job_id=request.job_id,
                status=request.status,
                message=request.message
            )
            return pb2.HeartbeatAck(success=True)
        except Exception as e:
            self.logger.error(f"❌ Error in heartbeat: {str(e)}", exc_info=True)
            context.set_details(f"Error in heartbeat: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.HeartbeatAck(success=False)
        

@hydra.main(config_path="../conf", config_name="config")
def serve(cfg: DictConfig):
    logger = setup_logger(name="dispatcher", log_file="dispatcher.log")

    logger.info("🚀= Starting ZK Dispatcher Service")
    # logger.info(f"Loaded Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    
    dispatcher_cfg = cfg.dispatcher
    port = dispatcher_cfg.port
    max_workers = dispatcher_cfg.max_workers
    num_prover_workers = dispatcher_cfg.num_prover_workers
    s3_bucket = cfg.s3_bucket
    cache_setup = cfg.cache_setup
    overwrite_cache = cfg.overwrite_cache
    cache_backend = cfg.cache_backend
    data_exchange_backend = cfg.data_exchange_backend
    # print("\n" + OmegaConf.to_yaml(cfg))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=[
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ])
    pb_grpc.add_ZKJobServiceServicer_to_server(ZKJobDispatcher(
        s3_bucket=s3_bucket,
        cache_setup=cache_setup,
        cache_backend=cache_backend,
        overwrite_cache=overwrite_cache,
        data_exchange_backend=data_exchange_backend,
        num_prover_workers=num_prover_workers,
        logger=logger
    ), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"✅ Dispatcher gRPC server running on port {port}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.warning("⛔ Shutting down dispatcher...")
        server.stop(0)



if __name__ == "__main__":
    serve()
