import grpc
from concurrent import futures
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import zkservice_pb2 as pb2
import zkservice_pb2_grpc as pb_grpc
from zkInfer.job_manager import JobManager
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
    def __init__(self, logger=None, 
                 num_prover_workers=1, 
                 s3_bucket=None, 
                 cache_setup=False,
                 overwrite_setup=False):
        
        self.logger = logger
        self.s3_bucket = s3_bucket
        self.cache_setup = cache_setup
        self.overwrite_setup = overwrite_setup
        self.manager = InferenceRequestManager(logger=logger)

    # API for submitting new inference requests
    def SubmitInferenceRequest(self, request, context):
        request_id = self.manager.submit_request(
            model_name=request.model_name,
            onnx_model_path=request.onnx_model_path,
            input_data_path=request.input_data_path,
            split_mode=request.split_mode,
            ops_per_chunk=request.ops_per_chunk,
            logger=self.logger,
            num_prover_workers=self.manager.num_prover_workers,
            overwrite_cached_setup=self.overwrite_setup,
            s3_bucket=self.s3_bucket
        )
        return pb2.InferenceRequestAck(request_id=request_id)
    
    # Worker pulls a job
    def GetNextProofJob(self, request, context):
        job = self.manager.get_next_job()
        if job:
            # Convert ProofJob to proto message
            return pb2.ProofJobResponse(has_job=True, job=job.to_proto())
        else:
            return pb2.ProofJobResponse(has_job=False)
        
    # Worker submits results
    def SubmitProofResult(self, request, context):
        ok = self.manager.submit_job_result(
            job_id=request.job_id,
            zk_proof=request.zk_proof,
            status=request.status,
            profiling_data=request.profiling_data,
            error_message=getattr(request, "error_message", None)
        )
        return pb2.SubmitAck(ok=ok)
    
     # (Optionally) User queries status
    def GetRequestStatus(self, request, context):
        status = self.manager.get_request_status(request.request_id)
        if status is None:
            return pb2.RequestStatus(not_found=True)
        # Fill out proto status message
        return pb2.RequestStatus(
            progress=status["progress"],
            all_done=status["all_done"],
            any_failed=status["any_failed"],
            # etc...
        )


    
    def SubmitJob(self, request, context):
        try:
            job_name = request.job_name
            split_mode = request.split_mode or "auto"
            ops_per_chunk = request.ops_per_chunk if split_mode == "fixed" else None
            self.logger.info(
                f"💼 Submitting job '{job_name}' with split_mode='{split_mode}'"
                f"{f', ops_per_chunk={ops_per_chunk}' if ops_per_chunk else ''}")

            job_id = self.job_manager.register_job(
                job_name=job_name,
                onnx_model_path=request.onnx_model_path,
                input_data_path=request.input_data_path,
                split_mode=split_mode,
                ops_per_chunk=ops_per_chunk,
            )
            self.logger.info(f"✅ Job '{job_name}' registered with ID: {job_id}")

            return pb.JobSubmissionResponse(job_id=job_id)
        except Exception as e:
            self.logger.error(f"❌ Error submitting job: {str(e)}")
            context.set_details(f"Error submitting job: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb.JobSubmissionResponse(job_id="")

    def GetJobStatus(self, request, context):
        status = self.job_manager.get_job_status(request.job_id)
        return pb.JobStatusResponse(job_id=request.job_id, status=status)

    def GetNextSubJob(self, request, context):
        subjob = self.job_manager.get_next_sub_job(request.worker_id)
        if subjob is None:
            return pb.SubJobAssignment(job_available=False)

        return pb.SubJobAssignment(
            job_id=subjob["job_id"],
            sub_model_name=subjob["sub_model_name"],
            sub_job_id=subjob["sub_job_id"],
            model_path=subjob["model_path"],
            input_json=subjob["input_json"],
            s3_bucket=self.s3_bucket,
            cache_setup=self.cache_setup,
            overwrite_setup=self.overwrite_setup,
            job_available=True
        )
    

    def SubmitJobResult(self, request, context):
        try:
            self.job_manager.finalize_sub_job(
                job_id=request.job_id,
                sub_job_id=request.sub_job_id,
                status=request.status,
                proof=request.proof,
                message=request.message,
                ezkl_perf= json.loads(request.ezkl_json),
                halo2_perf= json.loads(request.halo2_json)

            )

            #kill dispacther now that the sub-job is done
            sys.exit(0)  # Uncomment to exit the dispatcher after job completion

            return pb.StatusAck(success=True, message="Result received")
        except Exception as e:
            self.logger.error(f"❌ Error finalizing sub-job: {str(e)}")
            context.set_details(f"Error finalizing sub-job: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb.StatusAck(success=False, message=str(e))

    def SendHeartbeat(self, request, context):
        try:
            self.job_manager.record_heartbeat(
                job_id=request.job_id,
                sub_job_id=request.sub_job_id,
                worker_id=request.worker_id,
                status=request.status,
                message=request.message
            )
            return pb.HeartbeatAck(success=True)
        except Exception as e:
            self.logger.error(f"❌ Error in heartbeat: {str(e)}")
            context.set_details(f"Error in heartbeat: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb.HeartbeatAck(success=False)

    # def ListActiveSubJobs(self, request, context):
    #     live = self.job_manager.list_active_jobs()
    #     resp = pb.SubJobStatusResponse()
    #     for sid, info in live.items():
    #         resp.statuses[sid].worker_id = info["worker"]
    #         resp.statuses[sid].status = info["status"]
    #         resp.statuses[sid].progress = info["progress"] / 100.0
    #         resp.statuses[sid].message = info["message"]
    #         resp.statuses[sid].last_seen = info["last_seen"]
    #     return resp
    
    # def SendPerfReport(self, request, context):
    #     try:
    #         self.job_manager.record_performance_report(
    #             job_id=request.job_id,
    #             sub_job_id=request.sub_job_id,
    #             worker_id=request.worker_id,
    #             ezkl_perf=json.loads(request.ezkl_json),
    #             halo2_perf=json.loads(request.halo2_json)
    #         )
    #         return pb.StatusAck(success=True, message="Performance report received")
    #     except Exception as e:
    #         self.logger.error(f"❌ Error in performance report: {str(e)}")
    #         context.set_details(f"Error in performance report: {str(e)}")
    #         context.set_code(grpc.StatusCode.INTERNAL)
    #         return pb.StatusAck(success=False, message=str(e))

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
    overwrite_setup = cfg.overwrite_setup
    print("\n" + OmegaConf.to_yaml(cfg))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_ZKJobServiceServicer_to_server(ZKJobDispatcher(logger, num_prover_workers, s3_bucket, cache_setup, overwrite_setup), server)
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
