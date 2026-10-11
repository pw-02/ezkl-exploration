import logging
import os
import time
from typing import Any, Dict, List

from zkinfer.config.runtime import ArtifactStoreConfig, ProvingCacheConfig
from zkinfer.graph.onnx_splitter import split_onnx_model_with_inputs
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import ProofJob
from zkinfer.storage.artifact_store import ArtifactStore
from zkinfer.storage.io import exists, load_json


class RequestBuilder:
    """Materialize one inference request into independently executable proof jobs."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def build_jobs(
        self,
        request,
        artifact_store: ArtifactStore,
        artifact_store_config: ArtifactStoreConfig,
        proving_cache: ProvingCacheConfig,
        max_retries: int,
        transfer_only: bool = False,
    ) -> List[ProofJob]:
        materialized_partitions = self._load_or_split_model(request)
        jobs: List[ProofJob] = []

        model_store_time = 0.0
        input_store_time = 0.0

        for partition in materialized_partitions:
            setup_cache_uri = self._build_setup_cache_uri(
                request=request,
                proving_cache=proving_cache,
                model_name=partition.name,
                parent_model_hash=partition.parent_hash,
                model_hash=partition.sub_hash,
            )
            setup_cache_ref = self._artifact_prefix_ref(proving_cache, setup_cache_uri)

            output_prefix_uri = self._build_output_prefix_uri(
                request=request,
                artifact_store_config=artifact_store_config,
                model_name=partition.name,
                model_hash=partition.sub_hash,
            )
            output_prefix_ref = self._artifact_prefix_ref(artifact_store_config, output_prefix_uri)

            if transfer_only:
                profiling_file_path = None
                profiling_data = {}
                predicted_duration_s = 0.0
                predicted_peak_memory_gb = 0.0
                model_ref = ArtifactRef.empty()
            else:
                profiling_file_path = os.path.join(setup_cache_uri, "profiling.json")
                profiling_data = self._load_profiling_data(
                    model_name=partition.name,
                    profiling_file_path=profiling_file_path,
                    proving_cache=proving_cache,
                )
                predicted_duration_s = float(profiling_data.get("job_runtime(s)", 0.0) or 0.0)
                predicted_peak_memory_gb = float(
                    profiling_data.get(
                        "max_process_memory(GB)",
                        profiling_data.get("max_system_memory(GB)", 0.0),
                    )
                    or 0.0
                )

                start = time.perf_counter()
                model_ref = artifact_store.put_bytes(
                    partition.model.SerializeToString(),
                    kind="model",
                    filename=f"{partition.name}.onnx",
                )
                model_store_time += time.perf_counter() - start

            start = time.perf_counter()
            input_ref = artifact_store.put_json(
                partition.input_data,
                kind="input",
                filename=f"{partition.name}.json",
            )
            input_store_time += time.perf_counter() - start

            job = ProofJob(
                inference_request_name=request.name,
                model_name=partition.name,
                inference_request_id=request.request_id,
                model_ref=model_ref,
                input_ref=input_ref,
                setup_cache_ref=setup_cache_ref,
                output_prefix_ref=output_prefix_ref,
                profiling_file_path=profiling_file_path,
                model_write_time=model_store_time,
                input_write_time=0.0,
                profiling_data=profiling_data,
                predicted_duration_s=predicted_duration_s,
                predicted_peak_memory_gb=predicted_peak_memory_gb,
                artifact_key=partition.sub_hash,
                max_retries=max_retries,
                parent_model_hash=partition.parent_hash,
                model_hash=partition.sub_hash,
            )
            jobs.append(job)

        request.preparation_metrics["artifact_model_store_time(s)"] = model_store_time
        request.preparation_metrics["artifact_input_store_time(s)"] = input_store_time
        request.preparation_metrics["input_upload_wall_time(s)"] = input_store_time
        request.preparation_metrics["input_upload_service_time(s)"] = input_store_time
        request.preparation_metrics["input_upload_workers"] = 1

        self.logger.info(
            "Materialized %d jobs into content-addressed artifact store in %.3fs model + %.3fs input",
            len(jobs),
            model_store_time,
            input_store_time,
        )
        return jobs

    @staticmethod
    def _artifact_prefix_ref(config, uri: str) -> ArtifactRef:
        return ArtifactRef(
            backend=config.backend,
            uri=uri,
            s3_bucket=config.s3_bucket or "",
        )

    def _build_output_prefix_uri(
        self,
        request,
        artifact_store_config: ArtifactStoreConfig,
        model_name: str,
        model_hash: str,
    ) -> str:
        job_leaf = f"{model_name}_{model_hash[:8]}"
        return os.path.join(
            artifact_store_config.root_dir,
            "runs",
            request.request_id,
            "jobs",
            job_leaf,
        )

    def _load_or_split_model(self, request):
        split_mode = (request.split_mode or "none").lower()
        self.logger.info(
            "Preparing request %s using split_mode=%s ops_per_chunk=%s simplify_model=%s",
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

    def _build_setup_cache_uri(
        self,
        request,
        proving_cache: ProvingCacheConfig,
        model_name: str,
        parent_model_hash: str,
        model_hash: str,
    ) -> str:
        parent_cache_dir = os.path.join(proving_cache.root_dir, f"{request.name}_{parent_model_hash}")
        split_mode = (request.split_mode or "none").lower()
        if split_mode == "none":
            return os.path.join(parent_cache_dir, "full", parent_model_hash)
        split_key = f"{split_mode}_{request.ops_per_chunk}"
        return os.path.join(parent_cache_dir, "splits", split_key, f"{model_name}_{model_hash}")

    def _load_profiling_data(
        self,
        model_name: str,
        profiling_file_path: str,
        proving_cache: ProvingCacheConfig,
    ) -> Dict[str, Any]:
        if not exists(
            profiling_file_path,
            storage_type=proving_cache.backend,
            s3_bucket=proving_cache.s3_bucket,
        ):
            self.logger.warning(
                "Profiling data not found for %s at %s. Using conservative defaults.",
                model_name,
                profiling_file_path,
            )
            return {}
        return load_json(
            profiling_file_path,
            storage_type=proving_cache.backend,
            s3_bucket=proving_cache.s3_bucket,
        )
