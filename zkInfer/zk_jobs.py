import os
import json
import csv
import shutil
import logging
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict

import onnx
import ezkl

from zkInfer.inference_utils import run_model_inference, load_json_input
from zkInfer.onnx_splitter import split_model, collect_intermediate_inference_outputs
from zkInfer.metrics import get_fft_summary, get_msm_summary, read_csv_into_dict


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


class JobStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class OnnxModelToProve:
    def __init__(self, job_id, job_name, input_data_path, onnx_model_path, num_model_ops=None, num_model_params=None):
        self.job_id = job_id
        self.model_name = job_name
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.data_dir = os.path.dirname(onnx_model_path)

        self.status = JobStatus.PENDING
        self.num_model_ops = num_model_ops
        self.num_model_params = num_model_params
        self.overwrite = False

        self.vk_path = os.path.join(self.data_dir, 'vk.json')
        self.settings_path = os.path.join(self.data_dir, 'settings.json')
        self.compiled_circuit_path = os.path.join(self.data_dir, 'network.compiled')
        self.pk_path = os.path.join(self.data_dir, 'pk.json')
        self.witness_path = os.path.join(self.data_dir, 'witness.json')
        self.proof_path = os.path.join(self.data_dir, 'proof.pf')

    def generate_model_info(self):
        onnx_model = onnx.load(self.onnx_model_path)
        if self.num_model_ops is None or self.num_model_params is None:
            self.num_model_ops = len(onnx_model.graph.node)
            self.num_model_params = sum(
                onnx.numpy_helper.to_array(i).size for i in onnx_model.graph.initializer
            )

        model_ops = [node.op_type for node in onnx_model.graph.node]

        info = {
            "name": self.model_name,
            "onnx_model_path": self.onnx_model_path,
            "input_data_path": self.input_data_path,
            "model_ops": model_ops,
            "num_model_ops": self.num_model_ops,
            "num_model_params": self.num_model_params,
        }

        if os.path.exists(self.settings_path):
            with open(self.settings_path, 'r') as f:
                ezkl_settings = json.load(f)
            info.update(ezkl_settings)

        return info

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
        logger = logging.getLogger("zk")
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
        return timings


class GlobalProvingJob:
    def __init__(self, job_name, onnx_model_path, input_data_path,
                 split_mode="auto", ops_per_chunk=1, cache_setup_files=True):
        self.model_name = job_name
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.split_mode = split_mode
        self.ops_per_chunk = ops_per_chunk
        self.cache_setup_files = cache_setup_files

        self.inference_results = {}
        self.status = JobStatus.PENDING
        self.models_to_prove: List[OnnxModelToProve] = []

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        self.cache_directory = os.path.join('cache', self.model_name)
        self.report_directory = os.path.join('reports', self.model_name, timestamp)
        os.makedirs(self.report_directory, exist_ok=True)

    def prepare_for_processing(self, save_ezkl_settings=True):
        logger = logging.getLogger("zk")
        logger.info(f"Preparing job: {self.model_name}")

        # Run and save baseline inference
        try:
            result = run_model_inference(self.onnx_model_path, self.input_data_path)
            self.inference_results['non_zk_inference'] = result
            with open(os.path.join(self.report_directory, 'inference_results_nonzk.json'), 'w') as f:
                json.dump(result, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise

        # Determine splitting behavior
        if self.split_mode == "none":
            logger.info("Split mode is 'none'. Proving full model without splitting.")

            os.makedirs(self.cache_directory, exist_ok=True)
            model_path = os.path.join(self.cache_directory, 'model.onnx')
            input_path = os.path.join(self.cache_directory, 'input.json')

            shutil.copyfile(self.onnx_model_path, model_path)
            shutil.copyfile(self.input_data_path, input_path)

            self.models_to_prove.append(OnnxModelToProve(
                job_id=1,
                job_name=self.model_name,
                input_data_path=input_path,
                onnx_model_path=model_path,
            ))

        else:
            logger.info(f"Collecting intermediate outputs for split_mode='{self.split_mode}'.")
            intermediate_outputs = collect_intermediate_inference_outputs(self.onnx_model_path, self.input_data_path)

            if self.split_mode == "auto":
                split_group_size = 1
            elif self.split_mode == "fixed":
                split_group_size = self.ops_per_chunk
            else:
                raise ValueError(f"Invalid split_mode: {self.split_mode}")

            logger.info(f"Splitting model with group size: {split_group_size}")
            submodels = split_model(
                self.onnx_model_path,
                intermediate_outputs=intermediate_outputs,
                split_group_size=split_group_size,
                cache_dir=self.cache_directory,
            )

            logger.info(f"Split into {len(submodels)} sub-models.")
            for idx, (submodel_name, (input_path, model_path)) in enumerate(submodels.items(), 1):
                full_name = f"{self.model_name}_{submodel_name}"
                self.models_to_prove.append(OnnxModelToProve(
                    job_id=idx,
                    job_name=full_name,
                    input_data_path=input_path,
                    onnx_model_path=model_path,
                ))

        # Generate ezkl settings and write metadata
        if save_ezkl_settings:
            report_path = os.path.join(self.report_directory, 'ezkl_settings.csv')
            for model in self.models_to_prove:
                model._gen_settings()
                info = model.generate_model_info()
                file_exists = os.path.exists(report_path)
                with open(report_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=info.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(info)

        logger.info("Finished preparing job.")


    def gen_proof_for_sub_models(self):
        logger = logging.getLogger("zk")

        ezkl_file = os.path.join(self.report_directory, 'ezkl_perf.csv')
        halo2_file = os.path.join(self.report_directory, 'halo2_perf.csv')

        for idx, model in enumerate(self.models_to_prove, 1):
            logger.info(f"Proving sub-model {idx}/{len(self.models_to_prove)}: {model.model_name}")
            model_info = {
                'name': model.model_name,
                'onnx_model_path': model.onnx_model_path,
                'num_ops': model.num_model_ops,
                'num_params': model.num_model_params
            }

            ezkl_perf = {**model_info, **model.generate_zk_proof()}
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

            full_metrics = {**model_info, **circuit_info, **prover_info, **fft_data, **msm_data}
            with open(halo2_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=full_metrics.keys())
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(full_metrics)

            model_dir = os.path.join(self.report_directory, model.model_name)
            os.makedirs(model_dir, exist_ok=True)
            for f in os.listdir('.'):
                if f.startswith('halo2_') and f.endswith('.csv'):
                    shutil.move(f, os.path.join(model_dir, f))

        logger.info("All sub-models proved.")
