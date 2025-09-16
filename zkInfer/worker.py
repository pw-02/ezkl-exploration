import base64
import json
import os
import sys
import time
from typing import Any, Dict, Optional
import uuid
import grpc
import hydra
from omegaconf import DictConfig
from uuid import uuid4
import subprocess
import shutil
import logging

import psutil
import zkservice_pb2 as pb, zkservice_pb2_grpc as pb_grpc
import pandas as pd
# from zkInfer.s3_utils import download_from_s3, upload_to_s3, file_exists_in_s3, upload_if_not_exists, download_if_exists_in_s3
from zkInfer.utils import get_fft_summary, get_ip, get_msm_summary, parse_resource_usage_file, read_csv_into_dict
from zkInfer.storage_utils import (
    download_from_s3, upload_to_s3, file_exists_in_s3,
    download_if_exists_in_s3, s3_path
)

# def timed(fn):
#     def wrapper(self, *args, **kwargs):
#         start = time.perf_counter()
#         result = fn(self, *args, **kwargs)
#         return time.perf_counter() - start
#     return wrapper


class EZKLProofStages:
    def __init__(
        self, 
        input_data_path, 
        onnx_model_path, 
        cache_setup: bool,
        overwrite_cache: bool,
        cache_backend: Optional[str],
        s3_bucket: Optional[str] = None,
        status_file: Optional[str] = None,
        cache_dir=None,
        local_tmp_dir=None,
        share_data_via_s3: bool = False,
        # use_s3=None,
        logger = None, 


   
    ):
        import ezkl  # Only import here for multiprocess safety
        self.ezkl = ezkl
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.logger = logger or logging.getLogger("worker")
        self.status_file = status_file
        self.cache_setup = cache_setup
        self.overwrite_cache = overwrite_cache
        self.cache_backend = cache_backend
        self.cache_dir = cache_dir
        self.tmp_dir = local_tmp_dir
        self.s3_bucket = s3_bucket
        self.share_data_via_s3 = share_data_via_s3
        # self.cache_on_s3 = True if cache_backend == "s3" else False
      


        # self.use_s3 = use_s3

        # File paths
        if self.cache_setup and self.cache_backend == "local":
            self.settings_path = os.path.join(self.cache_dir, "settings.json")
            self.compiled_circuit_path = os.path.join(self.cache_dir, "network.compiled")
            self.pk_path = os.path.join(self.cache_dir, "pk.json")
            self.vk_path = os.path.join(self.cache_dir, "vk.json")
            self.witness_path = os.path.join(self.tmp_dir, "witness.json")
            self.proof_path = os.path.join(self.tmp_dir, "proof.pf")
        else:
            self.settings_path = os.path.join(self.tmp_dir, "settings.json")
            self.compiled_circuit_path = os.path.join(self.tmp_dir, "network.compiled")
            self.pk_path = os.path.join(self.tmp_dir, "pk.json")
            self.vk_path = os.path.join(self.tmp_dir, "vk.json")
            self.witness_path = os.path.join(self.tmp_dir, "witness.json")
            self.proof_path = os.path.join(self.tmp_dir, "proof.pf")
    
    # def get_run_args(self):
    #     run_args = self.ezkl.PyRunArgs()
    #     run_args.input_visibility = "public"
    #     run_args.param_visibility = "fixed"
    #     run_args.output_visibility = "public"
    #     return run_args
       
    def _update_status(self, stage):
        if self.status_file:
            with open(self.status_file, "w") as f:
                f.write(stage)

    def _try_load_from_cache(self, local_path, s3_key):
        """Helper to load from local disk or S3, returning (used_cache, s3_read_time)."""
        read_time = 0.0
        if os.path.exists(local_path):
            return True, read_time
        if self.share_data_via_s3:
            download_start = time.perf_counter()
            downloaded = download_if_exists_in_s3(self.s3_bucket, s3_key, local_path)
            read_time = time.perf_counter() - download_start if downloaded else 0.0
            if downloaded:
                assert os.path.exists(local_path)
                return True, read_time
        return False, read_time
    
    def get_pk_file_size_gb(self):
        """
        Get the size of the proving key file in GB.
        Returns:
            float: Size in GB, or 0.0 if file does not exist.
        """
        if os.path.exists(self.pk_path):
            return os.path.getsize(self.pk_path) / (1024 ** 3)
        return 0.0
    
    def get_vk_file_size_gb(self):
        """
        Get the size of the verifying key file in GB.
        Returns:
            float: Size in GB, or 0.0 if file does not exist.
        """
        if os.path.exists(self.vk_path):
            return os.path.getsize(self.vk_path) / (1024 ** 3)
        return 0.0

    def _try_load_keys_from_cache(self):
        """
        Helper to load PK/VK from local or S3.
        Returns:
            used_cache (bool)
            s3_read_time (float)
            s3_write_time (float) (always 0 here, as we don't upload in this function)
        """
        # Local cache
        if os.path.exists(self.pk_path) and os.path.exists(self.vk_path):
            return True, 0.0, 0.0

        # S3 cache
        s3_read_time = 0.0
        if self.share_data_via_s3:
            pk_key = f"{self.cache_dir}/pk.json"
            vk_key = f"{self.cache_dir}/vk.json"
            if file_exists_in_s3(self.s3_bucket, pk_key) and file_exists_in_s3(self.s3_bucket, vk_key):
                download_start = time.perf_counter()
                download_from_s3(self.s3_bucket, pk_key, self.pk_path)
                download_from_s3(self.s3_bucket, vk_key, self.vk_path)
                s3_read_time = time.perf_counter() - download_start
                if os.path.exists(self.pk_path) and os.path.exists(self.vk_path):
                    return True, s3_read_time, 0.0
        return False, 0.0, 0.0

    def calibrate_settings(self):
        self._update_status("CALIBRATING")
        s3_write_time = 0.0
        s3_read_time = 0.0
        used_cache = False

        if not self.overwrite_cache:
            used_cache, s3_read_time = self._try_load_from_cache(self.settings_path, f"{self.cache_dir}/settings.json")
            if used_cache:
                return used_cache, s3_read_time, s3_write_time

        # Generate and calibrate settings (no cache hit)
        self.ezkl.gen_settings(self.onnx_model_path, self.settings_path)
        self.ezkl.calibrate_settings(self.input_data_path, self.onnx_model_path, self.settings_path, "resources")
        assert os.path.exists(self.settings_path)

        # Optionally upload to S3
        if self.cache_setup and self.cache_backend == "s3":
            upload_start = time.perf_counter()
            upload_to_s3(self.settings_path, self.s3_bucket, f"{self.cache_dir}/settings.json")
            s3_write_time = time.perf_counter() - upload_start
            self.logger.debug(f"Uploaded settings to S3: {self.settings_path} -> {self.s3_bucket}/{self.cache_dir}/settings.json")
        return False, s3_read_time, s3_write_time

    def compile_circuit(self):
        self._update_status("COMPILING_")
        s3_write_time = 0.0
        s3_read_time = 0.0
        used_cache = False


        if not self.overwrite_cache:
            used_cache, s3_read_time = self._try_load_from_cache(self.compiled_circuit_path, f"{self.cache_dir}/network.compiled")
            if used_cache:
                return used_cache, s3_read_time, s3_write_time

        # Compile circuit (no cache hit)
        self.ezkl.compile_circuit(
            self.onnx_model_path, self.compiled_circuit_path, self.settings_path
        )
        assert os.path.exists(self.compiled_circuit_path)

        # Optionally upload to S3
        if self.cache_setup and self.cache_backend == "s3":
            upload_start = time.perf_counter()
            upload_to_s3(self.compiled_circuit_path, self.s3_bucket, f"{self.cache_dir}/network.compiled")
            s3_write_time = time.perf_counter() - upload_start
            self.logger.debug(f"Uploaded compiled circuit to S3: {self.compiled_circuit_path} -> {self.s3_bucket}/{self.cache_dir}/network.compiled")
        return False, s3_read_time, s3_write_time

    def get_srs(self):
        self._update_status("GETTING_SRS")
        self.ezkl.get_srs(self.settings_path)

    def gen_witness(self):
        self._update_status("GENERATING_WITNESS")
        self.ezkl.gen_witness(self.input_data_path, self.compiled_circuit_path, self.witness_path)
        assert os.path.exists(self.witness_path)

    def gen_keys(self):
        """
        Ensure proving and verifying keys exist locally, fetching from S3 or generating if needed.
        Returns:
            used_cache (bool): True if loaded from local/S3 cache, False if generated
            s3_read_time (float): Time spent downloading from S3 (seconds)
            s3_write_time (float): Time spent uploading to S3 (seconds)
        """
        self._update_status("KEY_GEN")
        s3_write_time = 0.0
        s3_read_time = 0.0
        used_cache = False

        if not self.overwrite_cache:
            used_cache, s3_read_time, _ = self._try_load_keys_from_cache()
            if used_cache:
                return used_cache, s3_read_time, s3_write_time

        # If here, need to generate new keys
        self.ezkl.setup(self.compiled_circuit_path, self.vk_path, self.pk_path)

        # Optionally upload to S3
        if self.cache_setup and self.cache_backend == "s3":
            pk_key = f"{self.cache_dir}/pk.json"
            vk_key = f"{self.cache_dir}/vk.json"
            upload_start = time.perf_counter()
            upload_to_s3(self.pk_path, self.s3_bucket, pk_key)
            upload_to_s3(self.vk_path, self.s3_bucket, vk_key)
            s3_write_time = time.perf_counter() - upload_start
            self.logger.debug(f"Uploaded keys to S3: {self.pk_path}, {self.vk_path} -> {self.s3_bucket}/{pk_key}, {self.s3_bucket}/{vk_key}")

        return False, s3_read_time, s3_write_time
    
    def compute_proof(self):
        self._update_status("PROVING")
        self.ezkl.prove(self.witness_path, self.compiled_circuit_path, self.pk_path, self.proof_path, "single")
        assert os.path.exists(self.proof_path)

    def run_all(self, setup_only: bool = False) -> Dict[str, Any]:
        """
        Run all proof pipeline stages in sequence.
        Returns: perf_measurements dict for each stage.
        """
        total_setup_time = 0.0
        total_s3_read_time = 0.0
        total_s3_write_time = 0.0
        perf_measurements: Dict[str, Any] = {}

        calibrate_settings_start = time.perf_counter()
        used_cache, s3_read_time, s3_write_time = self.calibrate_settings()
        calibrate_settings_time = time.perf_counter() - calibrate_settings_start
        perf_measurements["calibrate_settings_time(s)"] = calibrate_settings_time
        perf_measurements["calibrate_settings_used_cache"] = used_cache
        perf_measurements["calibrate_settings_s3_read_time(s)"] = s3_read_time
        perf_measurements["calibrate_settings_s3_write_time(s)"] = s3_write_time
        total_setup_time += calibrate_settings_time
        total_s3_read_time += s3_read_time
        total_s3_write_time += s3_write_time

        compile_circuit_start = time.perf_counter()
        used_cache, s3_read_time, s3_write_time = self.compile_circuit()
        compile_circuit_time = time.perf_counter() - compile_circuit_start
        perf_measurements["ezkl_compile_circuit_time(s)"] = compile_circuit_time
        perf_measurements["ezkl_compile_circuit_used_cache"] = used_cache
        perf_measurements["ezkl_compile_circuit_s3_read_time(s)"] = s3_read_time
        perf_measurements["ezkl_compile_circuit_s3_write_time(s)"] = s3_write_time
        total_setup_time += compile_circuit_time
        total_s3_read_time += s3_read_time
        total_s3_write_time += s3_write_time

        get_srs_start = time.perf_counter()
        self.get_srs()
        get_srs_time = time.perf_counter() - get_srs_start
        perf_measurements["ezkl_get_srs_time(s)"] = get_srs_time
        total_setup_time += get_srs_time

        gen_witness_start = time.perf_counter()
        self.gen_witness()
        gen_witness_time = time.perf_counter() - gen_witness_start
        perf_measurements["ezkl_gen_witness_time(s)"] = gen_witness_time
        total_setup_time += gen_witness_time

        key_gen_start = time.perf_counter()
        used_cache, s3_read_time, s3_write_time = self.gen_keys()
        key_gen_time = time.perf_counter() - key_gen_start
        perf_measurements["ezkl_key_gen_time(s)"] = key_gen_time
        perf_measurements["ezkl_key_gen_used_cache"] = used_cache
        perf_measurements["ezkl_key_gen_s3_read_time(s)"] = s3_read_time
        perf_measurements["ezkl_key_gen_s3_write_time(s)"] = s3_write_time

        total_setup_time += key_gen_time
        total_s3_read_time += s3_read_time
        total_s3_write_time += s3_write_time
        perf_measurements["ezkl_setup_time(s)"] = total_setup_time
        perf_measurements["ezkl_setup_s3_read_time(s)"] = total_s3_read_time
        perf_measurements["ezkl_setup_s3_write_time(s)"] = total_s3_write_time

        if not setup_only:
            compute_proof_start = time.perf_counter()
            self.compute_proof()
            compute_proof_time = time.perf_counter() - compute_proof_start
            perf_measurements["ezkl_proof_time(s)"] = compute_proof_time
       
        return perf_measurements


class ZKProofWorker:
    def __init__(self, cfg: DictConfig, logger=None):

        # if cfg.worker.worker_id is None:
        #     self.worker_id = f"{base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('ascii')}"
        # else:
        #     # Use the provided worker ID from the config
        #     self.worker_id = cfg.worker.worker_id
        self.worker_id  = get_ip()   # or f"worker-{get_ip()}"


        self.cfg = cfg
        self.target = f"{cfg.dispatcher.host}:{cfg.dispatcher.port}"
        self.channel = None
        self.stub = None
        self.logger: logging.Logger = logger
         # Get system resources
        self.num_cpus_pysical = psutil.cpu_count(logical=False)
        self.num_cpus_logical = psutil.cpu_count(logical=True)
        self.mem_bytes = psutil.virtual_memory().total
        self.mem_gb = round(self.mem_bytes / 1e9, 2)  # or use 1024**3 for GiB
        pass

    def connect(self):
        if self.channel:
            self.channel.close()
        self.channel = grpc.insecure_channel(self.target,options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),    # 64 MiB
            ("grpc.max_receive_message_length", 64 * 1024 * 1024), # 64 MiB
        ])
        self.stub = pb_grpc.ZKJobServiceStub(self.channel)
        self.logger.info(f"✅ Connected to dispatcher at {self.target}")
    
    def safe_grpc_call(self, call_fn, action="gRPC call", max_retries=2):
            for attempt in range(1, max_retries+1):
                try:
                    return call_fn()
                except grpc.RpcError as e:
                    self.logger.error(f"❌ {action} failed (attempt {attempt}): {e} (code={e.code()})")
                    self.reconnect_if_needed()
                    if attempt == max_retries:
                        raise
                except Exception as e:
                    self.logger.error(f"❌ {action} encountered an exception: {e}", exc_info=True)
                    if attempt == max_retries:
                        raise

    # def safe_send_heartbeat(self, job_id, status, message):
    #     try:
    #         self.stub.SendHeartbeat(pb.HeartbeatRequest(
    #             worker_id=self.worker_id,
    #             job_id=job_id,
    #             status=status,
    #             message=message
    #         ))
    #     except grpc.RpcError as e:
    #         self.logger.error(f"Failed to send heartbeat: {e} (code={e.code()})")
    #         self.reconnect_if_needed()
    #     except Exception as e:
    #         self.logger.error(f"Exception sending heartbeat: {e}", exc_info=True)
    
    def send_final_job_result(self, job_id, status, proof=None, message=None, perf_metrics=None):
        def _call():
            return self.stub.SubmitJobResult(pb.JobResult(
                job_id=job_id,
                proof=proof or b"",
                status=status,
                perf_metrics=json.dumps(perf_metrics) if perf_metrics else "",
                message=message or "",
            ))
        try:
            self.safe_grpc_call(_call, action=f"SubmitJobResult({status})", max_retries=3)
            self.logger.info(f"Sent final SubmitJobResult: {job_id} ({status})")
        except grpc.RpcError as e:
            self.logger.error(f"❌ gRPC error sending SubmitJobResult after retries: {e} (code={e.code()})")
    
    def cleanup(self, working_dir=None):
        """Cleanup resources and close gRPC channel."""
        if working_dir and os.path.exists(working_dir):
            shutil.rmtree(working_dir, ignore_errors=True)
    
    def fetch_and_run_job(self):
        job_id = None
        local_tmp_dir = None
        heartbeat_proc = None
        resource_usage_proc = None
        ezkl_perf_measurements = {}
        try:
            # --- 1. Pull a new job from the dispatcher ---
            try:
                response = self.stub.GetNextJob(pb.WorkerIdRequest(worker_id=self.worker_id))
            except grpc.RpcError as e:
                self.logger.error(f"❌ gRPC error while fetching subjob: {e} (code={e.code()})")
                self.reconnect_if_needed()
                return

            if not response.job_available:
                self.logger.info("⏳ No jobs available. Sleeping...")
                time.sleep(5)
                return

            # Get all IDs and job info
            job_id = response.job_id
            model_path = response.model_path
            input_path = response.input_path
            s3_bucket = response.s3_bucket
            cache_setup = response.cache_setup
            overwrite_cache = response.overwrite_cache
            cache_backend = response.cache_backend
            share_data_via_s3 = response.share_data_via_s3

            use_s3_for_cache = True if cache_backend == "s3" else False
            cache_prefix = os.path.dirname(model_path)
            worker_pid = str(os.getpid())
            s3_read_time = 0
            local_tmp_dir = os.path.join('tmp', job_id)
            os.makedirs(local_tmp_dir, exist_ok=True)

            # --- 2. Model download/setup ---
            try:
                if share_data_via_s3:
                    #try downloading model and input from S3
                    if cache_setup and not use_s3_for_cache:
                        local_model_path = os.path.join(cache_prefix, os.path.basename(model_path))
                        os.makedirs(cache_prefix, exist_ok=True)
                    else:
                        # download to tmp dir that will be cleaned up later
                        local_model_path = os.path.join(local_tmp_dir, os.path.basename(model_path))
                    download_onnx_model_start = time.perf_counter()
                    download_from_s3(s3_bucket, model_path, local_model_path)
                    s3_read_time += time.perf_counter() - download_onnx_model_start

                     # must be in S3, download to tmp dir
                    local_input_path = os.path.join(local_tmp_dir, "input.json")
                    download_input_start = time.perf_counter()
                    download_from_s3(s3_bucket, input_path, local_input_path)
                    s3_read_time += time.perf_counter() - download_input_start

                else:
                    if os.path.exists(model_path):
                        # model_path is already local so we can use it directly
                        local_model_path = model_path
                    else:
                        #raise an error if model is not found
                        raise FileNotFoundError(f"Model path {model_path} does not exist locally or in S3.")
                    if os.path.exists(input_path):
                        # input_path is already local so we can use it directly
                        local_input_path = input_path
                    else:
                        #raise an error if input is not found
                        raise FileNotFoundError(f"Input path {input_path} does not exist locally or in S3.")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to prepare model/input: {e}", exc_info=True)
                self.send_final_job_result(job_id, status="FAILED", message=f"Model prep failed: {e}")
                return    

            # --- 4. Start resource logging/heartbeat processes ---
            status_file = os.path.join(local_tmp_dir, "status.txt")
            resource_usage_file = os.path.join(local_tmp_dir, 'resource_usage.log')
            resource_usage_proc = subprocess.Popen([
                sys.executable, 
                "zkInfer/resource_logger.py",
                "--log_file", resource_usage_file,
                "--pid", worker_pid
            ])

            heartbeat_proc = subprocess.Popen([
                sys.executable, "zkInfer/heartbeat.py",
                "--target", self.target,
                "--worker_id", self.worker_id,
                "--job_id", job_id,
                "--status_file", status_file,
                "--parent_pid", worker_pid
            ])

            os.environ["EZKL_LOG_DIR"] = local_tmp_dir

            # --- 5. Proof computation ---
            proof_stages = EZKLProofStages(
                input_data_path=local_input_path,
                onnx_model_path=local_model_path,
                cache_setup=cache_setup,
                overwrite_cache=overwrite_cache,
                cache_backend=cache_backend,
                s3_bucket=s3_bucket,
                status_file=status_file,
                cache_dir=cache_prefix,
                local_tmp_dir=local_tmp_dir,
                share_data_via_s3= share_data_via_s3,
                logger=self.logger,
                # use_s3=use_s3_for_cache,
            )
            try:
                ezkl_perf_measurements = proof_stages.run_all(setup_only=False)
            except Exception as e:
                self.logger.error(f"❌ Error during proof stages: {e}", exc_info=True)
                self.send_final_job_result(job_id, status="FAILED", message=f"Proof stages failed: {e}")
                return

            # --- 6. Post-processing and reporting ---
            try:
                max_process_mem, max_system_mem, avg_process_cpu, avg_system_cpu = parse_resource_usage_file(resource_usage_file)
            except Exception as e:
                self.logger.error(f"❌ Failed to parse resource usage file: {e}", exc_info=True)
                max_process_mem = max_system_mem = avg_process_cpu = avg_system_cpu = None

            reporting_metrics = {
                "worker_id": self.worker_id,
                "max_process_memory(GB)": max_process_mem,
                "max_system_memory(GB)": max_system_mem,
                "avg_process_cpu(%)": avg_process_cpu,
                "avg_system_cpu(%)": avg_system_cpu,
                "pk_file_size(GB)": proof_stages.get_pk_file_size_gb(),
                "vk_file_size(GB)": proof_stages.get_vk_file_size_gb(),
            }
            circuit_info = read_csv_into_dict(os.path.join(local_tmp_dir, "halo2_circuit.csv"))
            prover_info_cpu = read_csv_into_dict(os.path.join(local_tmp_dir, "halo2_prover_cpu.csv"))
            fft_summary = get_fft_summary(os.path.join(local_tmp_dir, "halo2_ffts.csv"))
            msm_summary = get_msm_summary(os.path.join(local_tmp_dir, "halo2_msms.csv"))
            reporting_metrics = {**reporting_metrics, **circuit_info, **prover_info_cpu, **fft_summary, **msm_summary, **ezkl_perf_measurements}

            try:
                empty_proof_for_testing = b""
                self.send_final_job_result(
                    job_id,
                    status="COMPLETED",
                    proof=empty_proof_for_testing,
                    perf_metrics=reporting_metrics,
                    message="Proof computation completed successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to send proof: {e}", exc_info=True)
                self.send_final_job_result(
                    job_id, 
                    status="FAILED",
                    message=f"Send proof failed: {e}",
                    perf_metrics=reporting_metrics
                )
                return

            self.logger.info(f"✅ Completed job {job_id}")

        except Exception as e:
            self.logger.error(f"❌ Error during job: {e}", exc_info=True)
            self.send_final_job_result(
                    job_id,
                    status="FAILED",
                    message=str(e)
                ) 
        finally:
            # --- Cleanup heartbeat/resource usage and working dir ---
            for proc in [heartbeat_proc, resource_usage_proc]:
                if proc:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        proc.kill()
            if local_tmp_dir and os.path.exists(local_tmp_dir):
                shutil.rmtree(local_tmp_dir, ignore_errors=True)

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
        level=logging.DEBUG,
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

    # import uuid, base64
    # wid = f"worker_{base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('ascii')}"
    # print(wid)

    main()