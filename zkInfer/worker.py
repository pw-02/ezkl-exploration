import base64
import json
import os
import sys
import time
import uuid
import grpc
import hydra
from omegaconf import DictConfig
from uuid import uuid4
import subprocess
import shutil
import logging
import zkservice_pb2 as pb, zkservice_pb2_grpc as pb_grpc
import pandas as pd
from zkInfer.s3_utils import download_from_s3, upload_to_s3, file_exists_in_s3, upload_if_not_exists, download_if_exists_in_s3
from zkInfer.utils import get_fft_device, get_fft_summary, get_msm_device, get_msm_summary, parse_resource_usage_file, read_csv_into_dict, get_total_fft_duration, get_total_msm_duration

# def timed(fn):
#     def wrapper(self, *args, **kwargs):
#         start = time.perf_counter()
#         result = fn(self, *args, **kwargs)
#         return time.perf_counter() - start
#     return wrapper

def timed(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        if isinstance(result, tuple):
            return (elapsed, *result)
        else:
            return (elapsed, result)
    return wrapper


class EZKLProofStages:
    def __init__(
        self, 
        job_name, 
        input_data_path, 
        onnx_model_path, 
        overwrite=False, 
        logger = None, 
        status_file=None,
        s3_bucket=None,
        cache_dir=None,
        working_dir=None,
    ):
        import ezkl  # Only import here for multiprocess safety
        self.ezkl = ezkl
        self.job_name = job_name
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.logger = logger or logging.getLogger("worker")
        self.status_file = status_file
        self.overwrite = overwrite
        self.s3_bucket = s3_bucket
        self.cache_dir = cache_dir
        self.working_dir = working_dir
        self.settings_path = os.path.join(self.cache_dir, "settings.json")
        self.compiled_circuit_path = os.path.join(self.cache_dir, "network.compiled")
        self.pk_path = os.path.join(self.cache_dir, "pk.json")
        self.vk_path = os.path.join(self.cache_dir, "vk.json")
        self.witness_path = os.path.join(self.working_dir, "witness.json")
        self.proof_path = os.path.join(self.working_dir, "proof.pf")

    def _update_status(self, stage):
        if self.status_file:
            with open(self.status_file, "w") as f:
                f.write(stage)

    @timed
    def calibrate_settings(self):
        self._update_status("CALIBRATING")
        # If not overwrite, check if settings already exist locally or in S3
        if not self.overwrite:
            if os.path.exists(self.settings_path):
                return ("skipped",)
            if self.s3_bucket and download_if_exists_in_s3(self.s3_bucket, f"{self.cache_dir}/settings.json", self.settings_path):
                assert os.path.exists(self.settings_path)
                return ("downloaded",)

        # If we are here, either overwrite is True or settings file did not exist
        self.ezkl.gen_settings(self.onnx_model_path, self.settings_path)
        self.ezkl.calibrate_settings(self.input_data_path, self.onnx_model_path, self.settings_path, "resources")
        assert os.path.exists(self.settings_path)
        return ("created",)


    @timed
    def compile_circuit(self):
        self._update_status("COMPILING")
         # If not overwrite, check if compiled circuit exists locally or can be downloaded from S3
        if not self.overwrite:
            if os.path.exists(self.compiled_circuit_path):
                return ("skipped",)
            if self.s3_bucket and download_if_exists_in_s3(self.s3_bucket,f"{self.cache_dir}/network.compiled",self.compiled_circuit_path):
                assert os.path.exists(self.compiled_circuit_path)
                return ("downloaded",)
        # If here, we need to (re)compile
        self.ezkl.compile_circuit(self.onnx_model_path, self.compiled_circuit_path, self.settings_path)
        assert os.path.exists(self.compiled_circuit_path)
        return ("created",)
    @timed
    def get_srs(self):
        self._update_status("GETTING_SRS")
        self.ezkl.get_srs(self.settings_path)
        return ("created",)

    @timed
    def gen_witness(self):
        self._update_status("GENERATING_WITNESS")
        self.ezkl.gen_witness(self.input_data_path, self.compiled_circuit_path, self.witness_path)
        assert os.path.exists(self.witness_path)
        return ("created",)

    @timed
    def setup(self):
        self._update_status("SETTING_UP")
        if not self.overwrite:
            if os.path.exists(self.pk_path) and os.path.exists(self.vk_path):
                return ("skipped",)
        if (
            self.s3_bucket
            and file_exists_in_s3(self.s3_bucket, f"{self.cache_dir}/pk.json")
            and file_exists_in_s3(self.s3_bucket, f"{self.cache_dir}/vk.json")
        ):
            download_from_s3(self.s3_bucket, f"{self.cache_dir}/pk.json", self.pk_path)
            download_from_s3(self.s3_bucket, f"{self.cache_dir}/vk.json", self.vk_path)
            if os.path.exists(self.pk_path) and os.path.exists(self.vk_path):
                return ("downloaded",)
        # If here, either overwriting or files do not exist
        self.ezkl.setup(self.compiled_circuit_path, self.vk_path, self.pk_path)
        return ("created",)

    @timed
    def prove(self):
        self._update_status("PROVING")
        self.ezkl.prove(self.witness_path, self.compiled_circuit_path, self.pk_path, self.proof_path, "single")
        assert os.path.exists(self.proof_path)
        return ("created",)

    def run_all(self):
        timings = {}
        provenances = {}

        try:
            for name, fn in [
                ("calibrate_settings", self.calibrate_settings),
                ("compile_circuit", self.compile_circuit),
                ("get_srs", self.get_srs),
                ("gen_witness", self.gen_witness),
                ("setup", self.setup),
                ("prove", self.prove),
            ]:
                t, provenance = fn()
                timings[f"ezkl_{name}(s)"] = f"{t:.3f}"
                provenances[f"ezkl_{name}_provenance"] = provenance

                self.logger.info(f"{self.job_name}: {name} took {t:.3f}s")
            self._update_status("REPORTING")
            self.logger.info(f"{self.job_name}: All stages completed..")
            return {"timings": timings, "provenances": provenances, "error": None}
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.logger.error(f"{self.job_name}: Exception during proof: {e}\n{tb}")
            self._update_status("FAILED")
            return {"timings": timings, "provenances": provenances, "error": f"{type(e).__name__}: {e}\n{tb}"}


class ZKProofWorker:
    def __init__(self, cfg: DictConfig, logger=None):
        self.worker_id = f"worker_{base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('ascii')}"
        self.cfg = cfg
        self.target = f"{cfg.dispatcher.host}:{cfg.dispatcher.port}"
        self.channel = None
        self.stub = None
        self.logger = logger

    def connect(self):
        if self.channel:
            self.channel.close()
        self.channel = grpc.insecure_channel(self.target)
        self.stub = pb_grpc.ZKJobServiceStub(self.channel)
        self.logger.info(f"✅ Connected to dispatcher at {self.target}")
    
    
    def fetch_and_run_job(self):
        try:
            response = self.stub.GetNextSubJob(pb.WorkerIDRequest(worker_id=self.worker_id))
            if not response.job_available:
                self.logger.info("⏳ No jobs available. Sleeping...")
                time.sleep(5)
                return
            # self.global_job_id = response.job_id
            # self.current_sub_job_id = response.sub_job_id

            self.logger.info(f"📦 Got sub-job {response.sub_model_name} for job {response.job_id}")
            job_id = response.job_id
            sub_job_id = response.sub_job_id
            sub_model_name = response.sub_model_name
            model_path = response.model_path
            s3_bucket = response.s3_bucket
            cache_setup = response.cache_setup
            overwrite_setup = response.overwrite_setup
            input_json = json.loads(response.input_json)
            cache_prefix = os.path.dirname(model_path)
            worker_pid = str(os.getpid())

            #check local cache directory to see if the model already exists
            if not os.path.exists(model_path):
                #doesn't exist, so we crealte the local cache directory, and download the model
                os.makedirs(cache_prefix, exist_ok=True)
                local_model_path = os.path.join(cache_prefix, os.path.basename(model_path))
                download_from_s3(s3_bucket, model_path, local_model_path)
            else:
                #model already exists, so we just use the local path
                local_model_path = model_path

            local_working_dir = os.path.join('tmp', sub_job_id)
            os.makedirs(local_working_dir, exist_ok=True)

            #save the json input file to the local working directory
            local_input_path = os.path.join(local_working_dir, "input.json")
            with open(local_input_path, "w") as f:
                json.dump(input_json, f, indent=4)
            #save info about the job to a dict and write it to a file
            job_info = {
                "job_id": job_id,
                "sub_job_id": sub_job_id,
                "sub_model_name": sub_model_name,
                "model_path": local_model_path,
                "s3_bucket": s3_bucket,
                "cache_setup": cache_setup,
                "overwrite_setup": overwrite_setup,
                "worker_pid": worker_pid
            }
            job_info_file = os.path.join(local_working_dir, "job_info.json")
            with open(job_info_file, "w") as f:
                json.dump(job_info, f, indent=4)


            # Ensure local report directory exists
            status_file = os.path.join(local_working_dir, "status.txt")
            resource_usage_file = os.path.join(local_working_dir, 'resource_usage.log')
            resource_usage_proc = subprocess.Popen([
                sys.executable, "zkInfer/resource_logger.py",
                "--log_file", resource_usage_file,
                # "--interval", str(3),
                "--pid", worker_pid
            ])
          
            # ---- Start heartbeat process BEFORE running proof ----
            heartbeat_proc = subprocess.Popen([
                sys.executable, "zkInfer/heartbeat.py",
                self.target, self.worker_id, job_id, sub_job_id, status_file, worker_pid
                ])  
            
            os.environ["EZKL_LOG_DIR"] = local_working_dir  # This only affects this process and its children
            proof_stages = EZKLProofStages(
                job_name= sub_model_name,
                input_data_path=local_input_path,
                onnx_model_path=local_model_path,
                overwrite=overwrite_setup,
                logger = self.logger,
                status_file=status_file,
                s3_bucket=s3_bucket,
                cache_dir = cache_prefix,
                working_dir=local_working_dir
            )
            # ---- Run proof (blocking) ----
            result = proof_stages.run_all()
            # ---- Stop heartbeat process after proof is done ----
            for proc in [heartbeat_proc, resource_usage_proc]:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
            # ---- Upload reports to S3 if needed ----
            s3_upload_start = time.perf_counter()
            if cache_setup and s3_bucket:
                # Upload settings, compiled circuit, pk, vk, and proof files
                upload_if_not_exists(proof_stages.settings_path, s3_bucket, f"{cache_prefix}/settings.json")
                upload_if_not_exists(proof_stages.compiled_circuit_path, s3_bucket, f"{cache_prefix}/network.compiled")
                upload_if_not_exists(proof_stages.pk_path, s3_bucket, f"{cache_prefix}/pk.json")
                upload_if_not_exists(proof_stages.vk_path, s3_bucket, f"{cache_prefix}/vk.json")
                # upload_if_not_exists(proof_stages.proof_path, s3_bucket, f"{cache_prefix}/proof.pf")
            time_to_upload = time.perf_counter() - s3_upload_start


            if result.get("error"):
                self.logger.error(f"❌ Error during proof: {result['error']}")
                self.stub.SendHeartbeat(pb.HeartbeatRequest(
                    worker_id=self.worker_id,
                    sub_job_id=sub_job_id,
                    job_id=job_id,
                    status="FAILED",
                    message=result['error']
                ))
                return
            

            self.stub.SubmitSubJobResult(pb.SubJobResult(
                job_id=job_id,
                sub_job_id=sub_job_id,
                proof=open(proof_stages.proof_path, "rb").read(),
                success=True))
            
            timing_metrics = result.get("timings", {})
            provenance_metrics = result.get("provenances", {})
            timing_metrics.update({"s3_upload_time(s)": f"{time_to_upload:.3f}"})
            
            #generate and reports share with the job_manager
            max_process_mem, max_system_mem, avg_process_cpu, avg_system_cpu = parse_resource_usage_file(resource_usage_file)
            model_info = {"job_id": job_id, "sub_job_id": sub_job_id, 
                          "onnx_model_path": local_model_path, 
                          "input_data_path": local_input_path, 
                          "max_process_memory(GB)": max_process_mem, 
                          "max_system_memory(GB)": max_system_mem,
                          "avg_process_cpu(%)": avg_process_cpu,
                          "avg_system_cpu(%)": avg_system_cpu}
            
            circuit_info = read_csv_into_dict(os.path.join(local_working_dir, "halo2_circuit.csv"))
            prover_info_cpu = read_csv_into_dict(os.path.join(local_working_dir, "halo2_prover_cpu.csv"))
            fft_summary = get_fft_summary( os.path.join(local_working_dir, f"halo2_ffts.csv"))
            msm_summary = get_msm_summary(os.path.join(local_working_dir, f"halo2_msms.csv"))

            halo2_perf = {**model_info, **circuit_info, **prover_info_cpu, **fft_summary, **msm_summary}
            ezkl_perf = {**model_info, **timing_metrics, **provenance_metrics}

            self.stub.SendPerfReport(pb.PerfReport(
                job_id=job_id,
                sub_job_id=sub_job_id,
                worker_id=self.worker_id,
                ezkl_json=json.dumps(ezkl_perf),
                halo2_json=json.dumps(halo2_perf)))  # or whatever your gRPC call is

            #delete the local working directory
            shutil.rmtree(local_working_dir, ignore_errors=True)

            self.stub.SendHeartbeat(pb.HeartbeatRequest(
                worker_id=self.worker_id,
                sub_job_id=sub_job_id,
                job_id=job_id,
                status="DONE",
                message="COMPLETED"
            ))
      
            self.logger.info(f"✅ Completed sub-job {sub_job_id}")

        except grpc.RpcError as e:
            self.logger.error(f"❌ gRPC error: {e.details()} (code={e.code()})")
            self.reconnect_if_needed()
        except Exception as e:
            self.logger.error(f"❌ Error during job: {e}", exc_info=True)
            if sub_job_id and job_id:
                self.stub.SendHeartbeat(pb.HeartbeatRequest(
                    worker_id=self.worker_id,
                    sub_job_id=sub_job_id,
                    job_id=job_id,
                    status="FAILED",
                    message="FAILED"
                ))
                


    # def clean_local_files(self):
    #   # Gather additional report files to move or upload
    #     files_to_move = []
    #     files_to_remove = []
    #     files_to_copy = []
    #     for f in os.listdir("."):
    #         if (f.startswith("halo2_fft") and f.endswith(".csv")) or (f.startswith("halo2_msm") and f.endswith(".csv")):
    #             files_to_remove.append(f)
    #         elif f.startswith("halo2_") and f.endswith(".csv"):
    #             files_to_remove.append(f)
    #         # elif f.startswith("worker") and f.endswith(".log"):
    #         #     files_to_copy.append(f)

    #     # for f in files_to_move:
    #     #     shutil.move(f, os.path.join(model_dir, f))
    #     for f in files_to_remove:
    #         os.remove(f)
    #     # for f in files_to_copy:
    #     #     shutil.copy(f, os.path.join(model_dir, f))


    # def save_reports(self, local_working_dir, local_report_dir, sub_job_id, local_model_path, local_input_path, timings):
    #     model_info = {"name": sub_job_id, "onnx_model_path": local_model_path, "input_data_path": local_input_path}
    #     ezkl_file = os.path.join(local_report_dir, "ezkl_perf.csv")
    #     halo2_file = os.path.join(local_report_dir, "halo2_perf.csv")
    #     ezkl_perf = {**model_info, **timings}
    #     with open(ezkl_file, "a", newline="") as f:
    #         writer = csv.DictWriter(f, fieldnames=ezkl_perf.keys())
    #         if f.tell() == 0:
    #             writer.writeheader()
    #         writer.writerow(ezkl_perf)

    #     circuit_info = read_csv_into_dict("halo2_circuit.csv")
    #     prover_info = read_csv_into_dict("halo2_prover.csv")
    #     prover_info_cpu = read_csv_into_dict("halo2_prover_cpu.csv")

    #     fft_data = {}
    #     msm_data = {}
    #     for suffix in ["setup", "prover", "verifier"]:
    #         fft_file = f"halo2_ffts_{suffix}.csv"
    #         msm_file = f"halo2_msms_{suffix}.csv"
    #         if os.path.exists(fft_file):
    #             fft_data.update(get_fft_summary(fft_file, suffix))
    #         if os.path.exists(msm_file):
    #             msm_data.update(get_msm_summary(msm_file, suffix))
    #     full_metrics = {**model_info, **circuit_info, **prover_info, **prover_info_cpu}
    #     with open(halo2_file, "a", newline="") as f:
    #         writer = csv.DictWriter(f, fieldnames=full_metrics.keys())
    #         if f.tell() == 0:
    #             writer.writeheader()
    #         writer.writerow(full_metrics)

    #     # Gather additional report files to move or upload
    #     files_to_move = []
    #     files_to_remove = []
    #     files_to_copy = []
    #     for f in os.listdir("."):
    #         if (f.startswith("halo2_fft") and f.endswith(".csv")) or (f.startswith("halo2_msm") and f.endswith(".csv")):
    #             files_to_move.append(f)
    #         elif f.startswith("halo2_") and f.endswith(".csv"):
    #             files_to_remove.append(f)
    #         elif f.startswith("worker") and f.endswith(".log"):
    #             files_to_copy.append(f)

    #     for f in files_to_move:
    #         shutil.move(f, os.path.join(model_dir, f))
    #     for f in files_to_remove:
    #         os.remove(f)
    #     for f in files_to_copy:
    #         shutil.copy(f, os.path.join(model_dir, f))



    def reconnect_if_needed(self):
        self.logger.info("🔁 Attempting to reconnect gRPC channel...")
        self.connect()
        time.sleep(5)

    def run(self):
        self.connect()
        while True:
            self.fetch_and_run_job()

@hydra.main(config_path="../conf", config_name="config")
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