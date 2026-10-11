import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from zkinfer.config.runtime import FileTransferConfig, ProvingCacheConfig
from zkinfer.graph.onnx_splitter import split_onnx_model_with_inputs
from zkinfer.runtime.models import ArtifactRef, ProofJob
from zkinfer.storage.io import exists, load_json, save_json, save_model_proto


class RequestBuilder:
    """Materialize one inference request into independently executable proof jobs."""

    def __init__(self, logger: logging.Logger, input_upload_workers: int = 16):
        self.logger = logger
        self.input_upload_workers = max(1, input_upload_workers)

    def build_jobs(
        self,
        request,
        file_transfer: FileTransferConfig,
        proving_cache: ProvingCacheConfig,
        max_retries: int,
        transfer_only: bool = False,
    ) -> List[ProofJob]:
        materialized_partitions = self._load_or_split_model(request)
        jobs: List[ProofJob] = []
        pending_input_uploads = []

        for partition in materialized_partitions:
            transfer_dir = os.path.join(
                file_transfer.root_dir,
                request.request_id,
                partition.parent_hash,
                partition.sub_hash,
            )
            model_uri = os.path.join(transfer_dir, "model.onnx")
            input_uri = os.path.join(transfer_dir, "input.json")
            setup_cache_uri = self._build_setup_cache_uri(
                request=request,
                proving_cache=proving_cache,
                model_name=partition.name,
                parent_model_hash=partition.parent_hash,
                model_hash=partition.sub_hash,
            )
            output_prefix_uri = self._build_output_prefix_uri(
                request=request,
                file_transfer=file_transfer,
                model_name=partition.name,
                model_hash=partition.sub_hash,
            )

            if transfer_only:
                model_write_time = 0.0
                profiling_file_path = None
                profiling_data = {}
                predicted_duration_s = 0.0
                predicted_peak_memory_gb = 0.0
            else:
                profiling_file_path = os.path.join(setup_cache_uri, "profiling.json")
                model_write_time = self._save_model_if_needed(
                    model_proto=partition.model,
                    model_uri=model_uri,
                    file_transfer=file_transfer,
                )
                profiling_data = self._load_profiling_data(
                    model_name=partition.name,
                    profiling_file_path=profiling_file_path,
                    file_transfer=file_transfer,
                )
                predicted_duration_s = float(profiling_data.get("job_runtime(s)", 0.0) or 0.0)
                predicted_peak_memory_gb = float(
                    profiling_data.get(
                        "max_process_memory(GB)",
                        profiling_data.get("max_system_memory(GB)", 0.0),
                    )
                    or 0.0
                )

            job = ProofJob(
                inference_request_name=request.name,
                model_name=partition.name,
                inference_request_id=request.request_id,
                model_ref=self._artifact_ref(file_transfer, model_uri),
                input_ref=self._artifact_ref(file_transfer, input_uri),
                setup_cache_ref=self._artifact_ref(proving_cache, setup_cache_uri),
                output_prefix_ref=self._artifact_ref(file_transfer, output_prefix_uri),
                profiling_file_path=profiling_file_path,
                model_write_time=model_write_time if file_transfer.backend == "s3" else 0.0,
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
            pending_input_uploads.append((job, partition.input_data, input_uri))

        self._save_inputs(
            pending_input_uploads=pending_input_uploads,
            file_transfer=file_transfer,
            preparation_metrics=request.preparation_metrics,
        )
        return jobs

    @staticmethod
    def _artifact_ref(config, uri: str) -> ArtifactRef:
        return ArtifactRef(backend=config.backend, uri=uri, s3_bucket=config.s3_bucket or "")

    def _build_output_prefix_uri(
        self,
        request,
        file_transfer: FileTransferConfig,
        model_name: str,
        model_hash: str,
    ) -> str:
        job_leaf = f"{model_name}_{model_hash[:8]}"
        if file_transfer.backend == "filesystem":
            base = request.run_dir or file_transfer.root_dir
            return os.path.join(base, "artifacts", "jobs", job_leaf)
        return os.path.join(
            file_transfer.root_dir,
            request.request_id,
            "artifacts",
            "jobs",
            job_leaf,
        )

    def _save_inputs(
        self,
        pending_input_uploads,
        file_transfer: FileTransferConfig,
        preparation_metrics: Dict[str, Any],
    ) -> None:
        if not pending_input_uploads:
            preparation_metrics["input_upload_wall_time(s)"] = 0.0
            preparation_metrics["input_upload_service_time(s)"] = 0.0
            preparation_metrics["input_upload_workers"] = 0
            return

        upload_workers = min(self.input_upload_workers, len(pending_input_uploads)) if file_transfer.backend == "s3" else 1
        upload_start = time.perf_counter()

        if upload_workers == 1:
            for job, input_data, input_uri in pending_input_uploads:
                duration = self._save_input(input_data, input_uri, file_transfer)
                if file_transfer.backend == "s3":
                    job.input_write_time = duration
        else:
            with ThreadPoolExecutor(max_workers=upload_workers, thread_name_prefix="input-upload") as executor:
                future_to_job = {
                    executor.submit(self._save_input, input_data, input_uri, file_transfer): job
                    for job, input_data, input_uri in pending_input_uploads
                }
                for future in as_completed(future_to_job):
                    future_to_job[future].input_write_time = future.result()

        input_upload_wall_time = time.perf_counter() - upload_start
        input_upload_service_time = sum(job.input_write_time for job, _, _ in pending_input_uploads)
        preparation_metrics["input_upload_wall_time(s)"] = input_upload_wall_time
        preparation_metrics["input_upload_service_time(s)"] = input_upload_service_time
        preparation_metrics["input_upload_workers"] = upload_workers

        self.logger.info(
            "Saved %d job inputs using %d upload workers in %.3fs (aggregate service time %.3fs)",
            len(pending_input_uploads),
            upload_workers,
            input_upload_wall_time,
            input_upload_service_time,
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

    def _save_model_if_needed(
        self,
        model_proto,
        model_uri: str,
        file_transfer: FileTransferConfig,
    ) -> float:
        if exists(model_uri, storage_type=file_transfer.backend, s3_bucket=file_transfer.s3_bucket):
            return 0.0
        start = time.perf_counter()
        save_model_proto(
            model_proto,
            model_uri,
            storage_type=file_transfer.backend,
            s3_bucket=file_transfer.s3_bucket,
        )
        return time.perf_counter() - start

    def _load_profiling_data(
        self,
        model_name: str,
        profiling_file_path: str,
        file_transfer: FileTransferConfig,
    ) -> Dict[str, Any]:
        if not exists(profiling_file_path, storage_type=file_transfer.backend, s3_bucket=file_transfer.s3_bucket):
            self.logger.warning(
                "Profiling data not found for %s at %s. Using conservative defaults.",
                model_name,
                profiling_file_path,
            )
            return {}
        return load_json(
            profiling_file_path,
            storage_type=file_transfer.backend,
            s3_bucket=file_transfer.s3_bucket,
        )

    def _save_input(self, input_data, input_uri: str, file_transfer: FileTransferConfig) -> float:
        start = time.perf_counter()
        save_json(
            input_data,
            input_uri,
            storage_type=file_transfer.backend,
            s3_bucket=file_transfer.s3_bucket,
        )
        return time.perf_counter() - start
