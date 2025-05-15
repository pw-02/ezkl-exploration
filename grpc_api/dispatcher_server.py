import grpc
from concurrent import futures
import time
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from grpc_api import zkservice_pb2 as pb
from grpc_api import zkservice_pb2_grpc as pb_grpc

from zkInfer.job_manager import JobManager
from zkInfer.zk_jobs import GlobalProvingJob

logger = logging.getLogger("zk_dispatcher")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class ZKJobDispatcher(pb_grpc.ZKJobServiceServicer):
    def __init__(self):
        self.job_manager = JobManager()
    
    def SubmitGlobalJob(self, request, context):
        job_name = request.job_name
        split_mode = request.split_mode or "auto"
        ops_per_chunk = request.ops_per_chunk if split_mode == "fixed" else None

        logger.info(
            f"💼 Submitting job '{job_name}' with split_mode='{split_mode}'"
            f"{f', ops_per_chunk={ops_per_chunk}' if ops_per_chunk else ''}"
        )

        job_id = self.job_manager.submit_global_job(
            job_name=job_name,
            onnx_model_path=request.onnx_model_path,
            input_data_path=request.input_data_path,
            split_mode=split_mode,
            ops_per_chunk=ops_per_chunk,
        )

        logger.info(f"🆔 Assigned Job ID: {job_id}")
        return pb.JobIDResponse(job_id=job_id)
    

    def GetJobStatus(self, request, context):
        status = self.job_manager.get_job_status(request.job_id)
        return pb.JobStatusResponse(job_id=request.job_id, status=status)
    


    def FetchNextSubJob(self, request, context):
        subjob = self.job_manager.fetch_next_sub_job(request.worker_id)
        if subjob is None:
            return pb.SubJobResponse(available=False)

        return pb.SubJobResponse(
            job_id=subjob["job_id"],
            sub_job_id=subjob["sub_job_id"],
            model_path=subjob["model_path"],
            input_path=subjob["input_path"],
            output_dir=subjob["output_dir"],
            available=True
        )
    
    
    def GetJobSummary(self, request, context):
        job_id = request.job_id
        job = self.job_manager.global_jobs.get(job_id)

        if not job:
            return pb.JobSummaryResponse(
                job_id=job_id,
                job_name="UNKNOWN",
                status="UNKNOWN",
                total_sub_jobs=0,
                completed_sub_jobs=0,
                sub_jobs=[]
            )

        total = len(job.models_to_prove)
        completed = sum(1 for sid in self.job_manager.sub_job_results if sid.startswith(job_id))
        status = "COMPLETED" if completed == total else "IN_PROGRESS"

        sub_job_infos = []
        for model in job.models_to_prove:
            sub_id = f"{job_id}_{model.job_id}"
            live_info = self.job_manager.active_sub_jobs.get(sub_id)

            if live_info:
                info = pb.SubJobStatusInfo(
                    sub_job_id=sub_id,
                    worker_id=live_info["worker"],
                    status=pb.SubJobStatus.Value(live_info["status"]),
                    progress=live_info["progress"] / 100.0,
                    message=live_info["message"],
                    last_seen=live_info["last_seen"]
                )
            else:
                info = pb.SubJobStatusInfo(
                    sub_job_id=sub_id,
                    worker_id="",
                    status=pb.SubJobStatus.QUEUED,
                    progress=0.0,
                    message="Not started",
                    last_seen=""
                )
            sub_job_infos.append(info)

        return pb.JobSummaryResponse(
            job_id=job_id,
            job_name=job.model_name,
            status=status,
            total_sub_jobs=total,
            completed_sub_jobs=completed,
            sub_jobs=sub_job_infos
        )


    def SubmitSubJobResult(self, request, context):
        self.job_manager.submit_sub_job_result(
            job_id=request.job_id,
            sub_job_id=request.sub_job_id,
            metrics=dict(request.metrics)
        )
        return pb.StatusAck(success=True, message="Result received")

    def SendHeartbeat(self, request, context):
        self.job_manager.record_heartbeat(
            sub_job_id=request.sub_job_id,
            worker_id=request.worker_id,
            status=request.status,
            progress=request.progress,
            message=request.message
        )
        return pb.HeartbeatAck(success=True)

    def ListActiveSubJobs(self, request, context):
        live = self.job_manager.list_active_jobs()
        resp = pb.SubJobStatusResponse()
        for sid, info in live.items():
            resp.statuses[sid].worker_id = info["worker"]
            resp.statuses[sid].status = info["status"]
            resp.statuses[sid].progress = info["progress"] / 100.0
            resp.statuses[sid].message = info["message"]
            resp.statuses[sid].last_seen = info["last_seen"]
        return resp


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def serve(cfg: DictConfig):
    
    logger.info("🚀 Starting ZK Dispatcher Service")
    # logger.info(f"Loaded Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    
    dispatcher_cfg = cfg.dispatcher
    port = dispatcher_cfg.port
    max_workers = dispatcher_cfg.max_workers

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_ZKJobServiceServicer_to_server(ZKJobDispatcher(), server)
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
