import os
import sys
import time
import grpc
import hydra
from omegaconf import DictConfig
from uuid import uuid4
import threading
import concurrent.futures
import subprocess
import shutil
import logging
import csv
from grpc_api import zkservice_pb2 as pb, zkservice_pb2_grpc as pb_grpc
from zkInfer.utils import get_fft_summary, get_msm_summary, read_csv_into_dict

def timed(fn):
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = fn(self, *args, **kwargs)
        return time.perf_counter() - start
    return wrapper

def run_zk_proof(
    parent_job_id,
    job_name,
    input_data_path,
    onnx_model_path,
    output_dir,
    logger_name="worker",
    send_status=None,   # function(stage) to send heartbeats if needed
):
    import ezkl
    logger = logging.getLogger(logger_name)
    model_dir = os.path.join(output_dir, job_name)
    os.makedirs(model_dir, exist_ok=True)
    data_dir = os.path.dirname(onnx_model_path)
    settings_path = os.path.join(data_dir, "settings.json")
    compiled_circuit_path = os.path.join(data_dir, "network.compiled")
    pk_path = os.path.join(data_dir, "pk.json")
    vk_path = os.path.join(data_dir, "vk.json")
    witness_path = os.path.join(data_dir, "witness.json")
    proof_path = os.path.join(data_dir, "proof.pf")

    timings = {}

    stages = [
        ('GENERATING_SETTINGS', lambda: ezkl.gen_settings(onnx_model_path, settings_path)),
        ('CALIBRATING', lambda: ezkl.calibrate_settings(input_data_path, onnx_model_path, settings_path, "resources")),
        ('COMPILING', lambda: ezkl.compile_circuit(onnx_model_path, compiled_circuit_path, settings_path)),
        ('GET_SRS', lambda: ezkl.get_srs(settings_path)),
        ('GEN_WITNESS', lambda: ezkl.gen_witness(input_data_path, compiled_circuit_path, witness_path)),
        ('SETUP', lambda: ezkl.setup(compiled_circuit_path, vk_path, pk_path)),
        ('PROVING', lambda: ezkl.prove(witness_path, compiled_circuit_path, pk_path, proof_path, "single")),
        # ('VERIFY', ... ),
    ]

    for stage, fn in stages:
        if send_status:
            send_status(stage)
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        timings[f"ezkl_{stage.lower()}(s)"] = f"{dt:.3f}"
        logger.info(f"{job_name}: {stage} took {dt:.3f}s")

    logger.info(f"{job_name}: All stages completed..")
    # You can call save_reports here as a standalone function if desired
    return timings

def save_reports(
    model_dir, report_dir, model_name, onnx_model_path, input_data_path, timings
):
    model_info = {"name": model_name, "onnx_model_path": onnx_model_path, "input_data_path": input_data_path}
    ezkl_file = os.path.join(report_dir, "ezkl_perf.csv")
    halo2_file = os.path.join(report_dir, "halo2_perf.csv")
    ezkl_perf = {**model_info, **timings}
    with open(ezkl_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ezkl_perf.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(ezkl_perf)
    circuit_info = read_csv_into_dict("halo2_circuit.csv")
    prover_info = read_csv_into_dict("halo2_prover.csv")
    prover_info_cpu = read_csv_into_dict("halo2_prover_cpu.csv")
    fft_data = {}
    msm_data = {}
    for suffix in ["setup", "prover", "verifier"]:
        fft_file = f"halo2_ffts_{suffix}.csv"
        msm_file = f"halo2_msms_{suffix}.csv"
        if os.path.exists(fft_file):
            fft_data.update(get_fft_summary(fft_file, suffix))
        if os.path.exists(msm_file):
            msm_data.update(get_msm_summary(msm_file, suffix))
    full_metrics = {**model_info, **circuit_info, **prover_info, **prover_info_cpu}
    with open(halo2_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=full_metrics.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(full_metrics)
    for f in os.listdir("."):
        if (f.startswith("halo2_fft") and f.endswith(".csv")) or (f.startswith("halo2_msm") and f.endswith(".csv")):
            shutil.move(f, os.path.join(model_dir, f))
        elif f.startswith("halo2_") and f.endswith(".csv"):
            os.remove(f)
        elif f.startswith("worker") and f.endswith(".log"):
            shutil.copy(f, os.path.join(model_dir, f))

class ZKProofWorker:
    def __init__(self, cfg: DictConfig, logger=None):
        self.worker_id = str(uuid4())
        self.cfg = cfg
        self.target = f"{cfg.dispatcher.host}:{cfg.dispatcher.port}"
        self.channel = None
        self.stub = None
        self.heartbeat_thread = None
        self.heartbeat_stop_event = threading.Event()
        self.current_sub_job_id = None
        self.parent_job_id = None
        self.logger = logger
        self.status = "STARTED"

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
                    status=self.status,
                    message=f"{self.status}"
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

            # Start log samplers (optional, unchanged)
            model_dir = os.path.join(response.output_dir, response.sub_job_id)
            os.makedirs(model_dir, exist_ok=True)
            usage_file = os.path.join(model_dir, 'system_usage.log')
            syslog_proc = subprocess.Popen([
                sys.executable, "zkInfer/sys_logger.py",
                "--log_file", usage_file,
                "--interval", "3"
            ])
            worker_pid = os.getpid()
            procwatch_log = os.path.join(model_dir, "process_usage.log")
            watcher_proc = subprocess.Popen([
                sys.executable, "zkInfer/process_watcher.py",
                "--pid", str(worker_pid),
                "--log_file", procwatch_log,
                "--interval", "1"
            ])

            # Define a status callback to send heartbeat at every stage
            def status_callback(stage):
                self.status = stage
                self.stub.SendHeartbeat(pb.HeartbeatRequest(
                    worker_id=self.worker_id,
                    job_id=response.job_id,
                    sub_job_id=response.sub_job_id,
                    status=stage,
                    message=f"Stage: {stage}"
                ))
                self.logger.info(f"Sent heartbeat: {stage}")

            # Run proof in a separate process
            self.start_heartbeat()
            metrics = None
            time.sleep(1)  # Give heartbeat some time to start
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_zk_proof,
                    response.job_id,
                    response.sub_job_id,
                    response.input_path,
                    response.model_path,
                    response.output_dir,
                    "worker",
                    # status_callback,
                )
                while not future.done():
                    time.sleep(5)
                metrics = future.result()
            self.stop_heartbeat()

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

            save_reports(
                model_dir,
                os.path.join(response.output_dir),
                response.sub_job_id,
                response.model_path,
                response.input_path,
                metrics
            )

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
            self.logger.error(f"❌ Error during job: {e}",exc_info=True)
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