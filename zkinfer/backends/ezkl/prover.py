import logging
import os
import time
from typing import Any, Dict, Optional, Tuple
import json
import math

from zkinfer.storage.s3 import (
    download_file,
    upload_file,
    exists,
    download_if_exists,
)

class EZKLProofStages:
    def __init__(
        self,
        input_data_path: str,
        onnx_model_path: str,
        proving_cache_enabled: bool,
        proving_cache_overwrite: bool,
        proving_cache_type: str,
        proving_cache_root_dir: Optional[str] = None,
        proving_cache_s3_bucket: Optional[str] = None,
        proving_cache_s3_prefix: Optional[str] = None,
        status_file: Optional[str] = None,
        local_tmp_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        import ezkl

        self.ezkl = ezkl

        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.logger = logger or logging.getLogger("ezkl")
        self.status_file = status_file

        self.proving_cache_enabled = proving_cache_enabled
        self.proving_cache_overwrite = proving_cache_overwrite
        self.proving_cache_type = proving_cache_type
        self.proving_cache_root_dir = proving_cache_root_dir
        self.proving_cache_s3_bucket = proving_cache_s3_bucket
        self.proving_cache_s3_prefix = proving_cache_s3_prefix

        self.tmp_dir = local_tmp_dir or "tmp"

        # For filesystem cache, reusable EZKL setup outputs live under proving_cache_root_dir.
        # For S3/no cache, local files are created under tmp_dir and optionally uploaded/downloaded.
        if self.proving_cache_enabled and self.proving_cache_type == "filesystem":
            setup_dir = self.proving_cache_root_dir or self.tmp_dir
        else:
            setup_dir = self.tmp_dir

        os.makedirs(setup_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.settings_path = os.path.join(setup_dir, "settings.json")
        self.compiled_circuit_path = os.path.join(setup_dir, "network.compiled")
        self.pk_path = os.path.join(setup_dir, "pk.json")
        self.vk_path = os.path.join(setup_dir, "vk.json")

        self.witness_path = os.path.join(self.tmp_dir, "witness.json")
        self.proof_path = os.path.join(self.tmp_dir, "proof.pf")

        self.ezkl_stting_dict = {}

    @property
    def use_s3_cache(self) -> bool:
        return self.proving_cache_enabled and self.proving_cache_type == "s3"

    def _s3_key(self, filename: str) -> str:
        prefix = self.proving_cache_s3_prefix or ""
        return f"{prefix.rstrip('/')}/{filename}"

    def _update_status(self, stage: str) -> None:
        if not self.status_file:
            return

        with open(self.status_file, "w", encoding="utf-8") as f:
            f.write(stage)

    def _try_load_from_cache(self, local_path: str, filename: str) -> Tuple[bool, float]:
        if not self.proving_cache_enabled:
            return False, 0.0

        if os.path.exists(local_path) and not self.proving_cache_overwrite:
            return True, 0.0

        if not self.use_s3_cache:
            return False, 0.0

        start = time.perf_counter()
        downloaded = download_if_exists(
            self.proving_cache_s3_bucket,
            self._s3_key(filename),
            local_path,
        )
        read_time = time.perf_counter() - start if downloaded else 0.0

        return downloaded and os.path.exists(local_path), read_time

    def _upload_to_cache(self, local_path: str, filename: str) -> float:
        if not self.use_s3_cache:
            return 0.0

        start = time.perf_counter()
        upload_file(
            local_path,
            self.proving_cache_s3_bucket,
            self._s3_key(filename),
        )
        return time.perf_counter() - start

    def _try_load_keys_from_cache(self) -> Tuple[bool, float]:
        if not self.proving_cache_enabled:
            return False, 0.0

        if (
            os.path.exists(self.pk_path)
            and os.path.exists(self.vk_path)
            and not self.proving_cache_overwrite
        ):
            return True, 0.0

        if not self.use_s3_cache:
            return False, 0.0

        pk_key = self._s3_key("pk.json")
        vk_key = self._s3_key("vk.json")

        if not (
            exists(self.proving_cache_s3_bucket, pk_key)
            and exists(self.proving_cache_s3_bucket, vk_key)
        ):
            return False, 0.0

        start = time.perf_counter()
        download_file(self.proving_cache_s3_bucket, pk_key, self.pk_path)
        download_file(self.proving_cache_s3_bucket, vk_key, self.vk_path)
        return True, time.perf_counter() - start
    
    def _load_settings(self) -> Dict[str, Any]:
        with open(self.settings_path, "r", encoding="utf-8") as f:
            return json.load(f)


    def _save_settings(self, settings: Dict[str, Any]) -> None:
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)


    def _fix_logrows_after_calibration(self, margin: int = 1) -> None:
        if not os.path.exists(self.settings_path):
            return

        settings = self._load_settings()
        run_args = settings.setdefault("run_args", {})

        total_assignments = int(settings.get("total_assignments") or 0)
        current_logrows = int(run_args.get("logrows") or 0)

        if total_assignments <= 0:
            self.logger.warning(
                "Cannot fix logrows because total_assignments=%s",
                total_assignments,
            )
            return

        required_logrows = math.ceil(math.log2(total_assignments)) + margin

        if required_logrows > current_logrows:
            self.logger.warning(
                "Increasing EZKL logrows from %s to %s because total_assignments=%s",
                current_logrows,
                required_logrows,
                total_assignments,
            )
            run_args["logrows"] = required_logrows
            self._save_settings(settings)

    def get_pk_file_size_gb(self) -> float:
        return os.path.getsize(self.pk_path) / (1024 ** 3) if os.path.exists(self.pk_path) else 0.0

    def get_vk_file_size_gb(self) -> float:
        return os.path.getsize(self.vk_path) / (1024 ** 3) if os.path.exists(self.vk_path) else 0.0

    def calibrate_settings(self):
        self._update_status("CALIBRATING")

        used_cache, s3_read_time = self._try_load_from_cache(
            self.settings_path,
            "settings.json",
        )
        if used_cache:
            return True, s3_read_time, 0.0

        self.ezkl.gen_settings(self.onnx_model_path, self.settings_path)
        #accuracy, resources
        
        # self.ezkl.calibrate_settings(
        #     self.input_data_path,
        #     self.onnx_model_path,
        #     self.settings_path,
        #     "resources", 
        #     scales=[7],
        # )
        # self._fix_logrows_after_calibration(margin=1)

        s3_write_time = self._upload_to_cache(self.settings_path, "settings.json")

        self.ezkl_stting_dict = self._load_settings()

    

        return False, 0.0, s3_write_time

    def compile_circuit(self):
        self._update_status("COMPILING")

        used_cache, s3_read_time = self._try_load_from_cache(
            self.compiled_circuit_path,
            "network.compiled",
        )
        if used_cache:
            return True, s3_read_time, 0.0

        self.ezkl.compile_circuit(
            self.onnx_model_path,
            self.compiled_circuit_path,
            self.settings_path,
        )

        s3_write_time = self._upload_to_cache(
            self.compiled_circuit_path,
            "network.compiled",
        )
        return False, 0.0, s3_write_time

    def get_srs(self):
        self._update_status("GETTING_SRS")
        self.ezkl.get_srs(self.settings_path)

    def gen_witness(self):
        self._update_status("GENERATING_WITNESS")
        self.ezkl.gen_witness(
            self.input_data_path,
            self.compiled_circuit_path,
            self.witness_path,
        )

    def gen_keys(self):
        self._update_status("KEY_GEN")

        used_cache, s3_read_time = self._try_load_keys_from_cache()
        if used_cache:
            return True, s3_read_time, 0.0

        self.ezkl.setup(self.compiled_circuit_path, self.vk_path, self.pk_path)

        s3_write_time = 0.0
        s3_write_time += self._upload_to_cache(self.pk_path, "pk.json")
        s3_write_time += self._upload_to_cache(self.vk_path, "vk.json")

        return False, 0.0, s3_write_time

    def compute_proof(self):
        self._update_status("PROVING")
        self.ezkl.prove(
            self.witness_path,
            self.compiled_circuit_path,
            self.pk_path,
            self.proof_path,
            "single",
        )

    def run_all(self, setup_only: bool = False) -> Dict[str, Any]:
        total_setup_time = 0.0
        total_s3_read_time = 0.0
        total_s3_write_time = 0.0
        metrics: Dict[str, Any] = {}

        stages = [
            ("ezkl_calibrate_settings", self.calibrate_settings),
            ("ezkl_compile_circuit", self.compile_circuit),
        ]

        for name, fn in stages:
            start = time.perf_counter()
            used_cache, s3_read_time, s3_write_time = fn()
            elapsed = time.perf_counter() - start

            metrics[f"{name}_time(s)"] = elapsed
            metrics[f"{name}_used_cache"] = used_cache
            metrics[f"{name}_s3_read_time(s)"] = s3_read_time
            metrics[f"{name}_s3_write_time(s)"] = s3_write_time

            total_setup_time += elapsed
            total_s3_read_time += s3_read_time
            total_s3_write_time += s3_write_time

        start = time.perf_counter()
        self.get_srs()
        elapsed = time.perf_counter() - start
        metrics["ezkl_get_srs_time(s)"] = elapsed
        total_setup_time += elapsed

        start = time.perf_counter()
        self.gen_witness()
        elapsed = time.perf_counter() - start
        metrics["ezkl_gen_witness_time(s)"] = elapsed
        total_setup_time += elapsed

        start = time.perf_counter()
        used_cache, s3_read_time, s3_write_time = self.gen_keys()
        elapsed = time.perf_counter() - start

        metrics["ezkl_key_gen_time(s)"] = elapsed
        metrics["ezkl_key_gen_used_cache"] = used_cache
        metrics["ezkl_key_gen_s3_read_time(s)"] = s3_read_time
        metrics["ezkl_key_gen_s3_write_time(s)"] = s3_write_time

        total_setup_time += elapsed
        total_s3_read_time += s3_read_time
        total_s3_write_time += s3_write_time

        metrics["ezkl_setup_time(s)"] = total_setup_time
        metrics["ezkl_setup_s3_read_time(s)"] = total_s3_read_time
        metrics["ezkl_setup_s3_write_time(s)"] = total_s3_write_time

        if not setup_only:
            start = time.perf_counter()
            self.compute_proof()
            metrics["ezkl_proof_time(s)"] = time.perf_counter() - start

        self._update_status("DONE")
        return metrics, self.ezkl_stting_dict