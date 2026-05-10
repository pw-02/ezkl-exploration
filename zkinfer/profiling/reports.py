import os
from typing import Dict, Optional


from zkinfer.storage.storage_utils import convert_csv_to_dict, write_dict_to_csv


def _values(data: Dict, key: str):
    value = data.get(key, [])
    if isinstance(value, list):
        return [v for v in value if isinstance(v, (int, float))]
    if isinstance(value, (int, float)):
        return [value]
    return []


def _safe_sum(data: Dict, key: str) -> float:
    return sum(_values(data, key))


def _safe_max(data: Dict, key: str) -> float:
    values = _values(data, key)
    return max(values) if values else 0.0


def _safe_avg(data: Dict, key: str) -> float:
    values = _values(data, key)
    return sum(values) / len(values) if values else 0.0


def write_job_report(job, perf_metrics: Optional[Dict] = None, out_dir: str = "reports") -> Dict:
    perf_metrics = perf_metrics or {}

    report_dir = os.path.join(out_dir, job.inference_request_id)
    os.makedirs(report_dir, exist_ok=True)

    queue_wait_time_s = (
        (job.started_time - job.queued_time).total_seconds()
        if job.queued_time and job.started_time
        else None
    )
    job_runtime_s = (
        (job.completed_time - job.started_time).total_seconds()
        if job.started_time and job.completed_time
        else None
    )
    total_elapsed_time_s = (
        (job.completed_time - job.queued_time).total_seconds()
        if job.queued_time and job.completed_time
        else None
    )

    total_s3_write_time = perf_metrics.get("ezkl_setup_s3_write_time(s)", 0) or 0
    total_s3_write_time += job.model_write_time or 0

    report = {
        "request_id": job.inference_request_id,
        "job_id": job.job_id,
        "job_name": job.job_name,
        "worker_id": perf_metrics.get("worker_id", "unknown"),
        "model_path": job.model_path,
        "input_path": job.input_path,
        "job_status": job.job_status.value,
        "queued_time": job.queued_time.isoformat() if job.queued_time else None,
        "started_time": job.started_time.isoformat() if job.started_time else None,
        "completed_time": job.completed_time.isoformat() if job.completed_time else None,
        "message": job.error_message,
        "model_write_time": job.model_write_time,
        "queue_wait_time(s)": queue_wait_time_s,
        "job_runtime(s)": job_runtime_s,
        "total_elapsed_time(s)": total_elapsed_time_s,
        "circuit_size(n)": perf_metrics.get("circuit_size(n)", 0),
        "create_vk_time(s)": perf_metrics.get("create_vk_time(s)", 0),
        "create_pk_time(s)": perf_metrics.get("create_pk_time(s)", 0),
        "vk_size_gb": perf_metrics.get("vk_file_size(GB)", 0),
        "pk_size_gb": perf_metrics.get("pk_file_size(GB)", 0),
        "setup_time(s)": perf_metrics.get("ezkl_setup_time(s)", 0),
        "prove_time(s)": perf_metrics.get("proof_time(s)", 0),
        "verify_time(s)": perf_metrics.get("verify_time(s)", 0),
        "read_pk_time(s)": perf_metrics.get("read_pk_time(s)", 0),
        "ezkl_proof_time(s)": perf_metrics.get("ezkl_proof_time(s)", 0),
        "max_system_memory(GB)": perf_metrics.get("max_system_memory(GB)", 0),
        "max_process_memory(GB)": perf_metrics.get("max_process_memory(GB)", 0),
        "avg_system_cpu(%)": perf_metrics.get("avg_system_cpu(%)", 0),
        "avg_process_cpu(%)": perf_metrics.get("avg_process_cpu(%)", 0),
        "total_fft_time(s)": perf_metrics.get("fft_total_time(s)", 0),
        "fft_device": perf_metrics.get("fft_device", "unknown"),
        "total_msm_time(s)": perf_metrics.get("msm_total_time(s)", 0),
        "msm_device": perf_metrics.get("msm_device", "unknown"),
        "total_s3_read_time(s)": perf_metrics.get("ezkl_setup_s3_read_time(s)", 0),
        "total_s3_write_time(s)": total_s3_write_time,
    }

    write_dict_to_csv(report, os.path.join(report_dir, "job_report.csv"))

    if perf_metrics:
        write_dict_to_csv(
            {
                "request_id": job.inference_request_id,
                "job_id": job.job_id,
                **perf_metrics,
            },
            os.path.join(report_dir, "perf_metrics.csv"),
        )

    return report


def write_request_report(request, out_dir: str = "reports") -> Dict:
    report_dir = os.path.join(out_dir, request.request_id)
    os.makedirs(report_dir, exist_ok=True)

    job_report_file = os.path.join(report_dir, "job_report.csv")
    job_data = convert_csv_to_dict(job_report_file)

    queue_wait_time_s = (
        (request.started_time - request.queued_time).total_seconds()
        if request.queued_time and request.started_time
        else None
    )
    request_runtime_s = (
        (request.completed_time - request.started_time).total_seconds()
        if request.started_time and request.completed_time
        else None
    )
    total_elapsed_time_s = (
        (request.completed_time - request.created_time).total_seconds()
        if request.created_time and request.completed_time
        else None
    )

    report = {
        "request_id": request.request_id,
        "request_name": request.name,
        "onnx_model_path": request.onnx_model_path,
        "input_data_path": request.input_data_path,
        "data_exchange_backend": request.data_exchange_backend,
        "cache_setup": request.cache_setup,
        "overwrite_cache": request.overwrite_cache,
        "cache_backend": request.cache_backend,
        "split_mode": request.split_mode,
        "ops_per_chunk": request.ops_per_chunk,
        "num_proof_jobs": len(request.proof_jobs),
        "num_prover_workers": request.num_prover_workers,
        "s3_bucket": request.s3_bucket,
        "created_time": request.created_time.isoformat(),
        "queued_time": request.queued_time.isoformat() if request.queued_time else None,
        "started_time": request.started_time.isoformat() if request.started_time else None,
        "completed_time": request.completed_time.isoformat() if request.completed_time else None,
        "error_message": request.error_message,
        "request_status": request.request_status.value,
        "queue_wait_time(s)": queue_wait_time_s,
        "request_runtime(s)": request_runtime_s,
        "total_elapsed_time(s)": total_elapsed_time_s,
        "agg_setup_time(s)": _safe_sum(job_data, "setup_time(s)"),
        "agg_prove_time(s)": _safe_sum(job_data, "prove_time(s)"),
        "agg_verify_time(s)": _safe_sum(job_data, "verify_time(s)"),
        "agg_fft_time(s)": _safe_sum(job_data, "total_fft_time(s)"),
        "agg_msm_time(s)": _safe_sum(job_data, "total_msm_time(s)"),
        "agg_circuit_size(n)": _safe_sum(job_data, "circuit_size(n)"),
        "agg_job_runtime(s)": _safe_sum(job_data, "job_runtime(s)"),
        "max_system_memory(GB)": _safe_max(job_data, "max_system_memory(GB)"),
        "max_process_memory(GB)": _safe_max(job_data, "max_process_memory(GB)"),
        "avg_system_cpu(%)": _safe_avg(job_data, "avg_system_cpu(%)"),
        "avg_process_cpu(%)": _safe_avg(job_data, "avg_process_cpu(%)"),
        "max_pk_size_gb": _safe_max(job_data, "pk_size_gb"),
        "max_vk_size_gb": _safe_max(job_data, "vk_size_gb"),
        "agg_s3_read_time(s)": _safe_sum(job_data, "total_s3_read_time(s)"),
        "agg_s3_write_time(s)": _safe_sum(job_data, "total_s3_write_time(s)"),
    }

    write_dict_to_csv(report, os.path.join(report_dir, "request_report.csv"))
    return report