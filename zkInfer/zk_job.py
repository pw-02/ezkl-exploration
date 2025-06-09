import os
import csv
import shutil

import ezkl
from zkInfer.job_manager import JobStatus
from zkInfer.metrics import get_fft_summary, get_msm_summary, read_csv_into_dict

from grpc_api.log_utils import setup_logger
logger = setup_logger('worker', log_file="worker.log")

import time
from functools import wraps

def timed(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = fn(self, *args, **kwargs)
        return time.perf_counter() - start
    return wrapper

def timed_with_result(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = fn(self, *args, **kwargs)
        duration = time.perf_counter() - start
        return result, duration
    return wrapper

class OnnxModelToProve:
    def __init__(self, parent_job_id, job_name, input_data_path, onnx_model_path, output_dir):
        self.parent_job_id = parent_job_id
        self.model_name = job_name
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.data_dir = os.path.dirname(onnx_model_path)
        # self.report_dir = os.path.join(output_dir, job_name)
        self.report_dir = output_dir
        self.status = JobStatus.IN_PROGRESS
        self.overwrite = False
        self.vk_path = os.path.join(self.data_dir, 'vk.json')
        self.settings_path = os.path.join(self.data_dir, 'settings.json')
        self.compiled_circuit_path = os.path.join(self.data_dir, 'network.compiled')
        self.pk_path = os.path.join(self.data_dir, 'pk.json')
        self.witness_path = os.path.join(self.data_dir, 'witness.json')
        self.proof_path = os.path.join(self.data_dir, 'proof.pf')
        self.model_info= {'name': job_name, 'onnx_model_path': onnx_model_path, 'input_data_path': input_data_path}
        self.model_dir = os.path.join(self.report_dir, self.model_name)
        os.makedirs( self.model_dir, exist_ok=True)

    @timed
    def _gen_settings(self):
        if not self.overwrite and os.path.exists(self.settings_path):
            return 0.0
        ezkl.gen_settings(self.onnx_model_path, self.settings_path)

    @timed
    def _calibrate_settings(self):
        ezkl.calibrate_settings(self.input_data_path, self.onnx_model_path, self.settings_path, "resources")

    @timed
    def _compile_circuit(self):
        if not self.overwrite and os.path.exists(self.compiled_circuit_path):
            return 0.0
        ezkl.compile_circuit(self.onnx_model_path, self.compiled_circuit_path, self.settings_path)

    @timed
    def _get_srs(self):
        ezkl.get_srs(self.settings_path)

    @timed
    def _gen_witness(self):
        ezkl.gen_witness(self.input_data_path, self.compiled_circuit_path, self.witness_path)
        assert os.path.exists(self.witness_path)

    @timed
    def _setup(self):
        if not self.overwrite and os.path.exists(self.pk_path):
            return 0.0
        ezkl.setup(self.compiled_circuit_path, self.vk_path, self.pk_path)
        for file in ['halo2_ffts.csv', 'halo2_msms.csv']:
            if os.path.exists(file):
                shutil.move(file, file.replace('.', '_setup.'))

    @timed
    def _prove(self):
        ezkl.prove(self.witness_path, self.compiled_circuit_path, self.pk_path, self.proof_path, "single")
        for file in ['halo2_ffts.csv', 'halo2_msms.csv']:
            if os.path.exists(file):
                shutil.move(file, file.replace('.', '_prover.'))
        assert os.path.exists(self.proof_path)

    @timed
    def _verify(self):
        try:
            ezkl.verify(self.proof_path, self.settings_path, self.vk_path)
            for file in ['halo2_ffts.csv', 'halo2_msms.csv']:
                if os.path.exists(file):
                    shutil.move(file, file.replace('.', '_verifier.'))
        except Exception:
            return 0.0

    def generate_zk_proof(self):
        # logger = logging.getLogger("worker")
        stages = [
            ('gen_settings', self._gen_settings),
            ('calibrate_settings', self._calibrate_settings),
            ('compile_circuit', self._compile_circuit),
            ('get_srs', self._get_srs),
            ('witness_gen', self._gen_witness),
            ('setup', self._setup),
            ('prove', self._prove),
            ('verify', self._verify),
        ]
        timings = {}
        for name, fn in stages:
            time_taken = fn()
            timings[f"ezkl_{name}(s)"] = f"{time_taken:.3f}"
            logger.info(f"{self.model_name}: {name} took {time_taken:.3f}s")
        
        logger.info(f"{self.model_name}: All stages completed.. Saving reports")
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
        # self.save_reports(timings)
        return timings
    
    def save_reports(self, timings):
        ezkl_file = os.path.join(self.report_dir, 'ezkl_perf.csv')
        halo2_file = os.path.join(self.report_dir, 'halo2_perf.csv')
        
        ezkl_perf = {**self.model_info, **timings}
        with open(ezkl_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ezkl_perf.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(ezkl_perf)

        circuit_info = read_csv_into_dict('halo2_circuit.csv')
        prover_info = read_csv_into_dict('halo2_prover.csv')

        fft_data = {}
        msm_data = {}

        for suffix in ['setup', 'prover', 'verifier']:
            fft_file = f'halo2_ffts_{suffix}.csv'
            msm_file = f'halo2_msms_{suffix}.csv'
            if os.path.exists(fft_file):
                fft_data.update(get_fft_summary(fft_file, suffix))
            if os.path.exists(msm_file):
                msm_data.update(get_msm_summary(msm_file, suffix))

        full_metrics = {**self.model_info, **circuit_info, **prover_info, **fft_data, **msm_data}
        
        with open(halo2_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=full_metrics.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(full_metrics)


        for f in os.listdir('.'):
            if f.startswith('halo2_fft') and f.endswith('.csv') or f.startswith('halo2_msm') and f.endswith('.csv'):
                shutil.move(f, os.path.join(self.model_dir, f))
            elif f.startswith('halo2_') and f.endswith('.csv'):
                #delete the file
                # shutil.move(f, os.path.join(self.report_dir, f))

                os.remove(f)

        
