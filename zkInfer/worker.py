import json
import os
import sys
import time
import grpc
import hydra
from omegaconf import DictConfig
from uuid import uuid4
import concurrent.futures
import subprocess
import shutil
import logging
import csv
import zkservice_pb2 as pb, zkservice_pb2_grpc as pb_grpc
import onnx
import pandas as pd
def timed(fn):
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = fn(self, *args, **kwargs)
        return time.perf_counter() - start
    return wrapper

def read_csv_into_dict(file_path):
    """Reads a CSV file with one row into a dictionary."""
    data = {}
    try:
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for key, value in row.items():
                    data[key] = value
                break  # only read the first row
    except FileNotFoundError:
        pass
    return data

def get_model_op_info(onnx_model_path):
        model = onnx.load(onnx_model_path)
        model_op_info = {
                'num_ops': len(model.graph.node),
                'num_params': sum(onnx.numpy_helper.to_array(i).size for i in model.graph.initializer),
                'model_ops': [node.op_type for node in model.graph.node]
            }
        return model_op_info

def get_fft_summary(fft_file, prefix):
    """Extract summary stats from FFT CSV report."""
    fft_metrics = {}
    try:
        df = pd.read_csv(fft_file)
        fft_metrics[f'{prefix}_fft_count'] = int(len(df))
        fft_metrics[f'{prefix}_fft_largest'] = int(df['size'].max())
        fft_metrics[f'{prefix}_fft_total_time(s)'] = float(df['duration(s)'].sum())
        fft_metrics[f'{prefix}_fft_avg_time(s)'] = float(df['duration(s)'].mean())
        fft_metrics[f'{prefix}_fft_device'] = str(df['device'].iloc[0])
    except Exception:
        pass
    return fft_metrics


def get_msm_summary(msm_file, prefix):
    """Extract summary stats from MSM CSV report."""
    msm_metrics = {}
    try:
        df = pd.read_csv(msm_file)
        msm_metrics[f'{prefix}_msm_count'] = int(len(df))
        msm_metrics[f'{prefix}_msm_largest'] = int(df['num_coeffs'].max())
        msm_metrics[f'{prefix}_msm_total_time(s)'] = float(df['duration(s)'].sum())
        msm_metrics[f'{prefix}_msm_avg_time(s)'] = float(df['duration(s)'].mean())
        msm_metrics[f'{prefix}_msm_device'] = str(df['device'].iloc[0])
    except Exception:
        pass
    return msm_metrics

class EZKLProofStages:
    def __init__(
        self, job_name, input_data_path, onnx_model_path, output_dir, overwrite=False, logger_name="worker", status_file=None
    ):
        import ezkl  # Only import here for multiprocess safety
        self.ezkl = ezkl
        self.job_name = job_name
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.output_dir = output_dir
        self.logger = logging.getLogger(logger_name)
        self.status_file = status_file
        self.overwrite = overwrite

        self.model_dir = os.path.join(output_dir, job_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.data_dir = os.path.dirname(onnx_model_path)
        self.settings_path = os.path.join(self.data_dir, "settings.json")
        self.calibration_path = os.path.join(self.data_dir, "calibration.json")

        self.compiled_circuit_path = os.path.join(self.data_dir, "network.compiled")
        self.pk_path = os.path.join(self.data_dir, "pk.json")
        self.vk_path = os.path.join(self.data_dir, "vk.json")
        self.witness_path = os.path.join(self.data_dir, "witness.json")
        self.proof_path = os.path.join(self.data_dir, "proof.pf")

    def _update_status(self, stage):
        if self.status_file:
            with open(self.status_file, "w") as f:
                f.write(stage)

    @timed
    def calibrate_settings(self):
        self._update_status("CALIBRATING")
        if self.overwrite or not os.path.exists(self.settings_path):
            self.ezkl.gen_settings(self.onnx_model_path, self.settings_path)
            self.ezkl.calibrate_settings(self.input_data_path, self.onnx_model_path, self.settings_path, "resources")
        
        model_info = {"name": self.job_name, "onnx_model_path": self.onnx_model_path, "input_data_path": self.input_data_path}
        model_op_info = get_model_op_info(self.onnx_model_path)   
        setting_report = os.path.join(self.output_dir, "settings.csv")

        with open(self.settings_path, 'r') as f:
                ezkl_settings = json.load(f)
        model_settings = {**model_info, **model_op_info, **ezkl_settings}
    
        header_written = os.path.exists(setting_report)
        with open(setting_report, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=model_settings.keys())
                if not header_written:
                    writer.writeheader()
                writer.writerow(model_settings)
            #copy the settings file to the output directory
            

        
    # @timed
    # def calibrate_settings(self):
    #     self._update_status("CALIBRATING")
    #     if self.overwrite or not os.path.exists(self.calibration_path):
    #         self.ezkl.calibrate_settings(self.input_data_path, self.onnx_model_path, self.settings_path, "resources")

    @timed
    def compile_circuit(self):
        self._update_status("COMPILING")
        if self.overwrite or not os.path.exists(self.compiled_circuit_path):
            self.ezkl.compile_circuit(self.onnx_model_path, self.compiled_circuit_path, self.settings_path)

    @timed
    def get_srs(self):
        self._update_status("GETTING_SRS")
        self.ezkl.get_srs(self.settings_path)

    @timed
    def gen_witness(self):
        self._update_status("GENERATING_WITNESS")
        self.ezkl.gen_witness(self.input_data_path, self.compiled_circuit_path, self.witness_path)
        assert os.path.exists(self.witness_path)

    @timed
    def setup(self):
        self._update_status("SETTING_UP")
        if self.overwrite or not (os.path.exists(self.pk_path) and os.path.exists(self.vk_path)):
            self.ezkl.setup(self.compiled_circuit_path, self.vk_path, self.pk_path)

    @timed
    def prove(self):
        self._update_status("PROVING")
        self.ezkl.prove(self.witness_path, self.compiled_circuit_path, self.pk_path, self.proof_path, "single")
        assert os.path.exists(self.proof_path)

    def run_all(self):
        timings = {}
        try:
            for name, fn in [
                ("calibrate_settings", self.calibrate_settings),
                ("compile_circuit", self.compile_circuit),
                ("get_srs", self.get_srs),
                ("gen_witness", self.gen_witness),
                ("setup", self.setup),
                ("prove", self.prove),
            ]:
                t = fn()
                timings[f"ezkl_{name}(s)"] = f"{t:.3f}s"
                self.logger.info(f"{self.job_name}: {name} took {t:.3f}s")
            self._update_status("REPORTING")
            self.logger.info(f"{self.job_name}: All stages completed..")
            return {"timings": timings, "error": None}
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.logger.error(f"{self.job_name}: Exception during proof: {e}\n{tb}")
            self._update_status("FAILED")
            return {"timings": timings, "error": f"{type(e).__name__}: {e}\n{tb}"}


# Usage in your worker code:
def run_zk_proof(
    parent_job_id,
    job_name,
    input_data_path,
    onnx_model_path,
    output_dir,
    logger_name="worker",
    status_file=None,
    overwrite=False
):
    proof = EZKLProofStages(
        job_name=job_name,
        input_data_path=input_data_path,
        onnx_model_path=onnx_model_path,
        output_dir=output_dir,
        overwrite=overwrite,
        logger_name=logger_name,
        status_file=status_file,
    )
    return proof.run_all()

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
        self.current_sub_job_id = None
        self.parent_job_id = None
        self.logger = logger

    def connect(self):
        if self.channel:
            self.channel.close()
        self.channel = grpc.insecure_channel(self.target)
        self.stub = pb_grpc.ZKJobServiceStub(self.channel)
        self.logger.info(f"✅ Connected to dispatcher at {self.target}")


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
                sys.executable, "zkInfer/system_watcher.py",
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

            # Run proof in a separate process
            # self.start_heartbeat()
            metrics = None
            status_file = os.path.join(model_dir, "status.txt")

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
                    status_file
                    # status_callback,
                )
                while not future.done():
                    try:
                        with open(status_file) as f:
                            stage = f.read().strip()
                            #send hearbeat here
                            self.stub.SendHeartbeat(pb.HeartbeatRequest(
                                worker_id=self.worker_id,
                                sub_job_id=response.sub_job_id,
                                job_id=response.job_id,
                                status="STARTED",
                                message=stage
                            ))
                    except FileNotFoundError:
                        pass
                    except grpc.RpcError as e:
                        self.logger.warning(f"⚠️ Heartbeat failed: {e.details()}")
                    time.sleep(15)
                result = future.result()
            
            if result.get("error"):
                self.logger.error(f"❌ Error during proof: {metrics['error']}")
                self.stub.SendHeartbeat(pb.HeartbeatRequest(
                    worker_id=self.worker_id,
                    sub_job_id=response.sub_job_id,
                    job_id=response.job_id,
                    status="FAILED",
                    message=metrics['error']
                ))
                return
            metrics = result.get("timings", {})


            # self.stop_heartbeat()

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
                response.output_dir,
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
                message="COMPLETED"
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
            if self.current_sub_job_id:
                self.stub.SendHeartbeat(pb.HeartbeatRequest(
                    worker_id=self.worker_id,
                    sub_job_id=self.current_sub_job_id,
                    job_id=self.parent_job_id,
                    status="FAILED",
                    message="FAILED"
                ))
 

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