import logging
import os
import time
from typing import Any, Dict, List, Optional

from zkinfer.config.runtime import FileTransferConfig, ProvingCacheConfig
from zkinfer.graph.onnx_splitter import split_onnx_model_with_inputs
from zkinfer.runtime.models import ProofJob
from zkinfer.storage.io import (
    compute_bytes_md5_hex,
    exists,
    load_json,
    load_model_proto,
    save_json,
    save_model_proto,
)


class RequestBuilder:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def build_jobs(
        self,
        request,
        file_transfer: FileTransferConfig,
        proving_cache: ProvingCacheConfig,
        max_retries: int,
    ) -> List[ProofJob]:
        models_with_inputs = self._load_or_split_model(request)
        jobs: List[ProofJob] = []

        for model_name, model_hash, model_proto, input_data in models_with_inputs:
            job_dir = os.path.join(file_transfer.root_dir, model_hash)

            model_file_path = os.path.join(job_dir, "model.onnx")
            input_file_path = os.path.join(job_dir, "input.json")
            profiling_file_path = os.path.join(job_dir, "profiling.json")

            model_write_time = self._save_model_if_needed(
                model_proto=model_proto,
                model_file_path=model_file_path,
                file_transfer=file_transfer,
            )

            profiling_data = self._load_profiling_data(
                model_name=model_name,
                profiling_file_path=profiling_file_path,
                file_transfer=file_transfer,
            )
            predicted_duration = profiling_data.get("job_runtime(s)", 0.0)

            model_write_time += self._save_input(
                input_data=input_data,
                input_file_path=input_file_path,
                file_transfer=file_transfer,
            )

            jobs.append(
                ProofJob(
                    inference_request_name=request.name,
                    model_name=model_name,
                    inference_request_id=request.request_id,
                    model_path=model_file_path,
                    input_path=input_file_path,
                    profiling_file_path=profiling_file_path,
                    model_write_time=(
                        model_write_time if file_transfer.type == "s3" else 0.0
                    ),
                    profiling_data=profiling_data,
                    predicted_duration=predicted_duration,
                    max_retries=max_retries,
                )
            )

        return jobs

    def _load_or_split_model(self, request):
        split_mode = (request.split_mode or "none").lower()

        if split_mode == "none":
            model_proto = load_model_proto(request.onnx_model_path)
            input_data = load_json(request.input_data_path)
            model_hash = compute_bytes_md5_hex(model_proto.SerializeToString())

            return [(request.name, model_hash, model_proto, input_data)]

        self.logger.info(
            "Splitting request %s using split_mode=%s ops_per_chunk=%s",
            request.request_id,
            split_mode,
            request.ops_per_chunk,
        )

        return split_onnx_model_with_inputs(
            onnx_model_path=request.onnx_model_path,
            input_data_path=request.input_data_path,
            split_mode=split_mode,
            split_group_size=request.ops_per_chunk,
            simplify_model=getattr(request, "simplify_model", False),
            simplified_model_path=getattr(request, "simplified_model_path", None),
            simplify_input_shapes=getattr(request, "simplify_input_shapes", None),
        )

    def _save_model_if_needed(
        self,
        model_proto,
        model_file_path: str,
        file_transfer: FileTransferConfig,
    ) -> float:
        model_exists = exists(
            model_file_path,
            storage_type=file_transfer.type,
            s3_bucket=file_transfer.s3_bucket,
        )

        if model_exists:
            return 0.0

        start = time.perf_counter()

        save_model_proto(
            model_proto,
            model_file_path,
            storage_type=file_transfer.type,
            s3_bucket=file_transfer.s3_bucket,
        )

        return time.perf_counter() - start

    def _load_profiling_data(
        self,
        model_name: str,
        profiling_file_path: str,
        file_transfer: FileTransferConfig,
    ) -> Dict[str, Any]:
        profiling_exists = exists(
            profiling_file_path,
            storage_type=file_transfer.type,
            s3_bucket=file_transfer.s3_bucket,
        )

        if not profiling_exists:
            self.logger.warning(
                "Profiling data not found for %s at %s. Using predicted_duration=0.0",
                model_name,
                profiling_file_path,
            )
            return {}

        return load_json(
            profiling_file_path,
            storage_type=file_transfer.type,
            s3_bucket=file_transfer.s3_bucket,
        )

    def _save_input(
        self,
        input_data,
        input_file_path: str,
        file_transfer: FileTransferConfig,
    ) -> float:
        start = time.perf_counter()

        save_json(
            input_data,
            input_file_path,
            storage_type=file_transfer.type,
            s3_bucket=file_transfer.s3_bucket,
        )

        return time.perf_counter() - start