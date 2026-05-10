import logging
import os
import time
from typing import Dict, List

from zkinfer.config.runtime import FileTransferConfig, ProvingCacheConfig
from zkinfer.graph.onnx_splitter import split_onnx_model_with_inputs
from zkinfer.storage.storage_utils import (
    compute_bytes_md5_hex,
    file_exists,
    load_json_file,
    load_model_proto,
    save_json_file,
    save_model_proto_file,
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
    ):
        from zkinfer.runtime.coordinator import ProofJob

        models_with_inputs = self._load_or_split_model(request)
        jobs: List[ProofJob] = []

        for model_name, model_hash, model_proto, input_data in models_with_inputs:
            cache_dir = os.path.join(proving_cache.root_dir, model_hash)
            model_file_path = os.path.join(cache_dir, "model.onnx")
            input_file_path = os.path.join(cache_dir, "input.json")
            profiling_file_path = os.path.join(cache_dir, "profiling.json")

            model_write_time = self._save_model_if_needed(
                model_proto=model_proto,
                model_file_path=model_file_path,
                file_transfer=file_transfer,
                proving_cache=proving_cache,
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
                    model_write_time=model_write_time
                    if file_transfer.type == "s3"
                    else 0.0,
                    profiling_data=profiling_data,
                    predicted_duration=predicted_duration,
                    max_retries=max_retries,
                )
            )

        return jobs

    def _load_or_split_model(self, request):
        if request.split_mode == "none":
            model_proto = load_model_proto(request.onnx_model_path)
            input_data = load_json_file(request.input_data_path)
            model_hash = compute_bytes_md5_hex(model_proto.SerializeToString())
            return [(request.name, model_hash, model_proto, input_data)]

        return split_onnx_model_with_inputs(
            request.onnx_model_path,
            request.input_data_path,
            request.ops_per_chunk,
        )

    def _save_model_if_needed(
        self,
        model_proto,
        model_file_path: str,
        file_transfer: FileTransferConfig,
        proving_cache: ProvingCacheConfig,
    ) -> float:
        model_exists = file_exists(
            model_file_path,
            use_s3=file_transfer.type == "s3",
            s3_bucket=file_transfer.s3_bucket,
        )

        if not proving_cache.overwrite and model_exists:
            return 0.0

        start = time.perf_counter()
        save_model_proto_file(
            model_proto,
            model_file_path,
            use_s3=file_transfer.type == "s3",
            s3_bucket=file_transfer.s3_bucket,
        )
        return time.perf_counter() - start

    def _load_profiling_data(
        self,
        model_name: str,
        profiling_file_path: str,
        file_transfer: FileTransferConfig,
    ) -> Dict:
        profiling_exists = file_exists(
            profiling_file_path,
            use_s3=file_transfer.type == "s3",
            s3_bucket=file_transfer.s3_bucket,
        )

        if not profiling_exists:
            self.logger.warning(
                "Profiling data not found for %s at %s. Using predicted_duration=0.0",
                model_name,
                profiling_file_path,
            )
            return {}

        return load_json_file(
            profiling_file_path,
            use_s3=file_transfer.type == "s3",
            s3_bucket=file_transfer.s3_bucket,
        )

    def _save_input(
        self,
        input_data,
        input_file_path: str,
        file_transfer: FileTransferConfig,
    ) -> float:
        start = time.perf_counter()
        save_json_file(
            input_data,
            input_file_path,
            use_s3=file_transfer.type == "s3",
            s3_bucket=file_transfer.s3_bucket,
        )
        return time.perf_counter() - start