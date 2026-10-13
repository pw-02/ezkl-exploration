import logging
import os
import time
from typing import Any, Dict, List

from zkinfer.config.runtime import SetupStoreConfig
from zkinfer.runtime.models import ProofJob
from zkinfer.runtime.resource_profiles import get_profile_estimates, max_stage_memory_gb
from zkinfer.storage.shared_store import SharedStorage
from zkinfer.storage.io import exists, load_json


# Captures setup-affecting choices currently hard-coded by EZKLProofStages.
# Bump setup_identity_version whenever those semantics or the custom EZKL setup
# implementation change in a way that can alter settings, compiled circuits,
# PKs, or VKs.
EZKL_SETUP_IDENTITY = {
    "setup_identity_version": 1,
    "backend": "ezkl",
    "settings_generation": "gen_settings_then_calibrate_with_fallback",
    "calibration_target": "resources",
    "proof_type": "single",
    "disable_selector_compression": False,
}


class RequestBuilder:
    """Materialize one inference request into independently executable proof jobs."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def build_jobs(
        self,
        request,
        shared_storage: SharedStorage,
        setup_store: SetupStoreConfig,
        max_retries: int,
    ) -> List[ProofJob]:
        materialized_partitions = self._load_or_split_model(request)
        jobs: List[ProofJob] = []

        total_model_store_time = 0.0
        total_input_store_time = 0.0

        for index, partition in enumerate(materialized_partitions):
            model_bytes = partition.model.SerializeToString()
            setup_key = shared_storage.setup_key_for_model(
                model_bytes,
                EZKL_SETUP_IDENTITY,
            )
            started = time.perf_counter()
            model_ref = shared_storage.put_model(model_bytes, setup_key=setup_key)
            setup_ref = shared_storage.setup_ref(setup_key)
            model_write_time = time.perf_counter() - started
            total_model_store_time += model_write_time

            resource_profile_path = (
                os.path.join(setup_ref.uri, "resource_profile.json")
                if setup_store.enabled else None
            )
            profiling_data = (
                self._load_resource_profile(partition.name, resource_profile_path, setup_store)
                if resource_profile_path else {}
            )
            profile_estimates = get_profile_estimates(profiling_data)
            predicted_duration_s = float(profile_estimates.get("job_runtime_s") or 0.0)
            predicted_peak_memory_gb = max_stage_memory_gb(profiling_data)
            partition_id = f"{index:05d}_{setup_key[:16]}"
            output_prefix_ref = shared_storage.proof_prefix(request.request_id, partition_id)

            started = time.perf_counter()
            input_ref = shared_storage.put_input(
                request.request_id, partition_id, partition.input_data,
            )
            input_write_time = time.perf_counter() - started
            total_input_store_time += input_write_time

            job = ProofJob(
                inference_request_name=request.name,
                model_name=partition.name,
                inference_request_id=request.request_id,
                model_ref=model_ref,
                input_ref=input_ref,
                setup_cache_ref=setup_ref,
                output_prefix_ref=output_prefix_ref,
                profiling_file_path=resource_profile_path,
                model_write_time=model_write_time,
                input_write_time=input_write_time,
                profiling_data=profiling_data,
                predicted_duration_s=predicted_duration_s,
                predicted_peak_memory_gb=predicted_peak_memory_gb,
                artifact_key=setup_key,
                max_retries=max_retries,
                parent_model_hash=partition.parent_hash,
                model_hash=partition.sub_hash,
            )
            jobs.append(job)

        request.preparation_metrics["artifact_model_store_time(s)"] = total_model_store_time
        request.preparation_metrics["artifact_input_store_time(s)"] = total_input_store_time
        request.preparation_metrics["input_upload_wall_time(s)"] = total_input_store_time
        request.preparation_metrics["input_upload_service_time(s)"] = total_input_store_time
        request.preparation_metrics["input_upload_workers"] = 1

        self.logger.info(
            "Materialized %d jobs into setup/request storage in %.3fs model + %.3fs input",
            len(jobs),
            total_model_store_time,
            total_input_store_time,
        )
        return jobs

    def _load_or_split_model(self, request):
        # Model execution is needed only for request preparation, not for
        # coordinator recovery, result handling, or control-plane tests.
        from zkinfer.graph.onnx_splitter import split_onnx_model_with_inputs

        split_mode = (request.split_mode or "none").lower()
        self.logger.info(
            "Preparing request %s using split_mode=%s operators_per_partition=%s simplify_model=%s",
            request.request_id,
            split_mode,
            request.ops_per_chunk,
            getattr(request, "simplify_model", False),
        )
        return split_onnx_model_with_inputs(
            model_path=request.onnx_model_path,
            input_data_path=request.input_data_path,
            split_mode=split_mode,
            split_group_size=request.ops_per_chunk,
            simplify_model=getattr(request, "simplify_model", False),
            simplified_model_path=getattr(request, "simplified_model_path", None),
            input_shapes=getattr(request, "input_shapes", None),
            model_name=request.name,
            metrics=request.preparation_metrics,
        )

    def _load_resource_profile(
        self,
        model_name: str,
        resource_profile_path: str,
        setup_store: SetupStoreConfig,
    ) -> Dict[str, Any]:
        if not exists(
            resource_profile_path,
            storage_type=setup_store.backend,
            s3_bucket=setup_store.s3_bucket,
        ):
            self.logger.info(
                "Resource profile not found for %s at %s; using conservative defaults",
                model_name,
                resource_profile_path,
            )
            return {}
        return load_json(
            resource_profile_path,
            storage_type=setup_store.backend,
            s3_bucket=setup_store.s3_bucket,
        )
