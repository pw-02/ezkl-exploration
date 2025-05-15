import time
import logging
import grpc
import hydra
from omegaconf import DictConfig
from grpc_api import zkservice_pb2 as pb, zkservice_pb2_grpc as pb_grpc
from zkInfer.zk_jobs import OnnxModelToProve
from uuid import uuid4
import threading

logger = logging.getLogger("zk.worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ZKProofWorker:
    def __init__(self, cfg: DictConfig):
        self.worker_id = str(uuid4())
        self.cfg = cfg
        self.target = f"{cfg.dispatcher.host}:{cfg.dispatcher.port}"
        self.channel = None
        self.stub = None
        self.heartbeat_thread = None
        self.heartbeat_stop_event = threading.Event()
        self.progress_ref = {"value": 0}
        self.current_sub_job_id = None

    def connect(self):
        if self.channel:
            self.channel.close()
        self.channel = grpc.insecure_channel(self.target)
        self.stub = pb_grpc.ZKJobServiceStub(self.channel)
        logger.info(f"✅ Connected to dispatcher at {self.target}")

    def start_heartbeat(self):
        self.heartbeat_stop_event.clear()
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()

    def stop_heartbeat(self):
        self.heartbeat_stop_event.set()
        if self.heartbeat_thread:
            self.heartbeat_thread.join()

    def _heartbeat_loop(self):
        while not self.heartbeat_stop_event.is_set():
            try:
                self.stub.SendHeartbeat(pb.HeartbeatRequest(
                    worker_id=self.worker_id,
                    sub_job_id=self.current_sub_job_id,
                    status="STARTED",
                    progress=self.progress_ref["value"],
                    message="Proving..."
                ))
            except grpc.RpcError as e:
                logger.warning(f"⚠️ Heartbeat failed: {e.details()}")
            time.sleep(15)

    def fetch_and_run_job(self):
        try:
            response = self.stub.FetchNextSubJob(pb.WorkerIDRequest(worker_id=self.worker_id))
            if not response.available:
                logger.info("⏳ No jobs available. Sleeping...")
                time.sleep(5)
                return

            self.current_sub_job_id = response.sub_job_id
            logger.info(f"📦 Got sub-job {response.sub_job_id} for job {response.job_id}")

            model = OnnxModelToProve(
                job_id=response.sub_job_id,
                job_name=response.sub_job_id,
                input_data_path=response.input_path,
                onnx_model_path=response.model_path
            )

            # Send initial heartbeat
            self.stub.SendHeartbeat(pb.HeartbeatRequest(
                worker_id=self.worker_id,
                sub_job_id=response.sub_job_id,
                status="STARTED",
                progress=0,
                message="Started"
            ))

            self.start_heartbeat()
            metrics = model.generate_zk_proof()  # optionally update self.progress_ref
            self.stop_heartbeat()

             # Send final heartbeat
            self.stub.SendHeartbeat(pb.HeartbeatRequest(
                worker_id=self.worker_id,
                sub_job_id=response.sub_job_id,
                status="DONE",
                progress=0,
                message="Done"
            ))
            
            self.stub.SubmitSubJobResult(pb.SubJobResult(
                job_id=response.job_id,
                sub_job_id=response.sub_job_id,
                metrics=metrics
            ))
            logger.info(f"✅ Completed sub-job {response.sub_job_id}")
            self.current_sub_job_id = None

        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error: {e.details()} (code={e.code()})")
            self.reconnect_if_needed()

    def reconnect_if_needed(self):
        logger.info("🔁 Attempting to reconnect gRPC channel...")
        self.connect()
        time.sleep(5)

    def run(self):
        self.connect()
        while True:
            self.fetch_and_run_job()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    worker = ZKProofWorker(cfg)
    worker.run()


if __name__ == "__main__":
    main()
