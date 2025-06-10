import time
import grpc
import hydra
from omegaconf import DictConfig
from grpc_api import zkservice_pb2 as pb, zkservice_pb2_grpc as pb_grpc
from zkInfer.zk_job import OnnxModelToProve
import sys
import subprocess
from uuid import uuid4
import threading
import psutil
from datetime import datetime
import os
import logging
import concurrent.futures

def sample_process_usage_during_job(process, interval, stop_event, stats):
    peak_mem = 0
    cpu_percents = []
    while not stop_event.is_set():
        try:
            mem = process.memory_info().rss
            peak_mem = max(peak_mem, mem)
            cpu = process.cpu_percent(interval=None)
            cpu_percents.append(cpu)
        except Exception:
            pass
        time.sleep(interval)
    stats["peak_mem_rss"] = peak_mem
    stats["avg_cpu_percent"] = sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0.0


class ZKProofWorker:
    def __init__(self, cfg: DictConfig, logger=None):
        self.worker_id = str(uuid4())
        self.cfg = cfg
        self.target = f"{cfg.dispatcher.host}:{cfg.dispatcher.port}"
        self.channel = None
        self.stub = None
        self.heartbeat_thread = None
        self.heartbeat_stop_event = threading.Event()
        self.progress_ref = {"value": 0}
        self.current_sub_job_id = None
        self.parent_job_id = None
        self.logger = logger

    def connect(self):
        if self.channel:
            self.channel.close()
        self.channel = grpc.insecure_channel(self.target)
        self.stub = pb_grpc.ZKJobServiceStub(self.channel)
        self.logger.info(f"✅ Connected to dispatcher at {self.target}")

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
                    job_id=self.parent_job_id,
                    sub_job_id=self.current_sub_job_id,
                    status="STARTED",
                    message="Proving..."
                ))
            except grpc.RpcError as e:
                self.logger.warning(f"⚠️ Heartbeat failed: {e.details()}")
            time.sleep(15)

    def fetch_and_run_job(self):
        try:
            response = self.stub.FetchNextSubJob(pb.WorkerIDRequest(worker_id=self.worker_id))
            if not response.available:
                self.logger.info("⏳ No jobs available. Sleeping...")
                time.sleep(5)
                return
            self.parent_job_id = response.job_id
            self.current_sub_job_id = response.sub_job_id
            self.logger.info(f"📦 Got sub-job {response.sub_job_id} for job {response.job_id}")

            model = OnnxModelToProve(
                parent_job_id=response.job_id,
                job_name=response.sub_job_id,
                input_data_path=response.input_path,
                onnx_model_path=response.model_path,
                output_dir=response.output_dir
            )

            # Send initial heartbeat
            self.stub.SendHeartbeat(pb.HeartbeatRequest(
                worker_id=self.worker_id,
                job_id=response.job_id,
                sub_job_id=response.sub_job_id,
                status="STARTED",
                message="Started"
            ))

            # Start system log samplers as subprocesses
            usage_file = os.path.join(model.model_dir, 'system_usage.log')
            syslog_proc = subprocess.Popen([
                sys.executable, "zkInfer/sys_logger.py",
                "--log_file", usage_file,
                "--interval", "3"
            ])
            worker_pid = os.getpid()
            procwatch_log = os.path.join(model.model_dir, "process_usage.log")
            watcher_proc = subprocess.Popen([
                sys.executable, "zkInfer/process_watcher.py",
                "--pid", str(worker_pid),
                "--log_file", procwatch_log,
                "--interval", "1"
            ])

            # Run proof in a separate process, so heartbeat/main thread stays alive
            self.start_heartbeat()
            metrics = None
            time.sleep(1)  # Give heartbeat a chance to start
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.generate_zk_proof)
                while not future.done():
                    time.sleep(5)  # keep the loop light; heartbeat runs in its own thread
                metrics = future.result()
            self.stop_heartbeat()

            # Cleanup log samplers
            watcher_proc.terminate()
            try:
                watcher_proc.wait(timeout=2)
            except Exception:
                watcher_proc.kill()

            syslog_proc.terminate()
            try:
                syslog_proc.wait(timeout=3)
            except Exception:
                syslog_proc.kill()

            # Add per-process stats to metrics, if needed
            model.save_reports(metrics)

            # Send final heartbeat
            self.stub.SendHeartbeat(pb.HeartbeatRequest(
                worker_id=self.worker_id,
                sub_job_id=response.sub_job_id,
                job_id=response.job_id,
                status="DONE",
                message="Done"
            ))

            self.stub.SubmitSubJobResult(pb.SubJobResult(
                job_id=response.job_id,
                sub_job_id=response.sub_job_id,
                metrics=metrics
            ))
            self.logger.info(f"✅ Completed sub-job {response.sub_job_id}")
            self.current_sub_job_id = None

        except grpc.RpcError as e:
            self.logger.error(f"❌ gRPC error: {e.details()} (code={e.code()})")
            self.reconnect_if_needed()
        except Exception as e:
            self.logger.error(f"❌ Error during job: {str(e)}")
            self.stop_heartbeat()

    def reconnect_if_needed(self):
        self.logger.info("🔁 Attempting to reconnect gRPC channel...")
        self.connect()
        time.sleep(5)

    def run(self):
        self.connect()
        while True:
            self.fetch_and_run_job()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("worker.log")
        ]
    )

    logger = logging.getLogger("worker")
    logger.info("🔧 Starting ZKProofWorker...")

    worker = ZKProofWorker(cfg, logger)
    worker.run()


if __name__ == "__main__":
    main()
