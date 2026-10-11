from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from zkinfer.backends.ezkl.ezkl_prover import EZKLProofStages
from zkinfer.profiling.msm_fft_parser import (
    read_csv_first_row,
    summarize_fft_report,
    summarize_msm_report,
)
from zkinfer.profiling.resource_parser import parse_resource_usage_file
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.node.phase_admission import ResourceAdmissionController
from zkinfer.storage.s3 import upload_file


@dataclass(frozen=True)
class LocalProverTask:
    node_id: str
    request_id: str
    job_id: str
    setup_key: str
    model_path: str
    input_path: str
    setup_cache_dir: str
    tmp_dir: str
    output_dir: str
    output_prefix: str
    output_backend: str
    output_s3_bucket: str = ""
    transfer_only: bool = False
    resource_monitor_interval_sec: int = 1
    model_materialize_time_s: float = 0.0
    input_materialize_time_s: float = 0.0
    transfer_metrics: Dict[str, Any] = field(default_factory=dict)
    phase_admission_dir: str = ""
    phase_admission_enabled: bool = True
    phase_memory_reserve_gb: float = 8.0
    phase_cpu_budget_percent: float = 100.0
    phase_profile_percentile: float = 0.90
    phase_fallback_cpu_percent: float = 100.0
    predicted_peak_memory_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalProverTask":
        return cls(**data)


@dataclass(frozen=True)
class ProverTaskResult:
    artifact_manifest_ref: Optional[ArtifactRef]
    metrics: Dict[str, Any]
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        ref = self.artifact_manifest_ref
        return {
            "artifact_manifest_ref": ref.to_dict() if ref else None,
            "metrics": self.metrics,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProverTaskResult":
        ref_data = data.get("artifact_manifest_ref")
        ref = ArtifactRef.from_dict(ref_data) if ref_data else None
        if ref is not None and ref.is_empty:
            ref = None
        return cls(
            artifact_manifest_ref=ref,
            metrics=data.get("metrics") or {},
            message=data.get("message") or "",
        )


class ResourceAwareEZKLProofStages(EZKLProofStages):
    """EZKL stages guarded by node-local resource reservations.

    The controller predicts each stage's memory/CPU demand from prior runs of
    the same setup key. There are no configured job-count or heavy-phase-count
    limits in the normal admission path.
    """

    def __init__(
        self,
        *args,
        admission: ResourceAdmissionController,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.admission = admission
        self.admission_metrics: Dict[str, Any] = {
            "phase_admission_enabled": admission.enabled,
            "phase_admission_wait_time(s)": 0.0,
        }

    @staticmethod
    def _metric_stage(stage: str) -> str:
        return stage.lower().replace("-", "_")

    @contextmanager
    def _admit(self, stage: str) -> Iterator[Dict[str, Any]]:
        # Waiting is explicit in the resource trace instead of being attributed
        # to the previous EZKL phase.
        self._update_status(f"WAITING_FOR_{stage}")
        with self.admission.hold(stage) as info:
            metric_stage = self._metric_stage(stage)
            wait_time = float(info.get("wait_time_s") or 0.0)
            self.admission_metrics["phase_admission_wait_time(s)"] += wait_time
            self.admission_metrics[
                f"phase_admission_{metric_stage}_wait_time(s)"
            ] = self.admission_metrics.get(
                f"phase_admission_{metric_stage}_wait_time(s)", 0.0
            ) + wait_time
            self.admission_metrics[
                f"phase_admission_{metric_stage}_predicted_memory(GB)"
            ] = float(info.get("memory_gb") or 0.0)
            self.admission_metrics[
                f"phase_admission_{metric_stage}_predicted_cpu_machine(%)"
            ] = float(info.get("cpu_percent") or 0.0)
            self.admission_metrics[
                f"phase_admission_{metric_stage}_estimate_samples"
            ] = int(info.get("samples") or 0)
            self.admission_metrics[
                f"phase_admission_{metric_stage}_estimate_source"
            ] = str(info.get("source") or "fallback")
            yield info

    def calibrate_settings_with_fallback(self):
        with self._admit("CALIBRATING"):
            return super().calibrate_settings_with_fallback()

    def compile_circuit(self):
        with self._admit("COMPILING"):
            return super().compile_circuit()

    def get_srs(self):
        with self._admit("GETTING_SRS"):
            return super().get_srs()

    def gen_witness(self):
        with self._admit("GENERATING_WITNESS"):
            return super().gen_witness()

    def gen_keys(self):
        with self._admit("KEY_GEN"):
            return super().gen_keys()

    def compute_proof(self):
        with self._admit("PROVING"):
            return super().compute_proof()

    def setup_and_compute_proof(self) -> Dict[str, float]:
        # The fused Rust operation does not return to Python between keygen and
        # proving. Reserve the larger of their historical resource envelopes for
        # the combined region while the Rust status file still records the exact
        # KEY_GEN -> PROVING transition.
        with self._admit("FUSED_KEYGEN_PROVE"):
            return super().setup_and_compute_proof()


class ProverTaskRunner:
    """Execute one proof using local paths only.

    This runner executes inside an isolated OS process. It has no coordinator or
    node-cache policy responsibilities.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def run(self, task: LocalProverTask) -> ProverTaskResult:
        os.makedirs(task.output_dir, exist_ok=True)
        os.makedirs(task.tmp_dir, exist_ok=True)

        if task.transfer_only:
            return self._run_transfer_only(task)

        status_file = os.path.join(task.output_dir, "status.txt")
        resource_proc, resource_usage_file = self._start_resource_logger(task, status_file)

        try:
            proof_stages = self._build_proof_stages(task, status_file)
            ezkl_metrics, _ = proof_stages.run_all(setup_only=False)
            ezkl_metrics.update(proof_stages.admission_metrics)

            metrics = self._collect_metrics(
                task=task,
                resource_usage_file=resource_usage_file,
                proof_stages=proof_stages,
                ezkl_metrics=ezkl_metrics,
            )
            manifest_ref, publish_metrics = self._publish_job_artifacts(task, proof_stages)
            metrics.update(self._flatten_transfer_metrics(task.transfer_metrics))
            metrics.update(self._flatten_output_metrics(publish_metrics))
            metrics["prover_process_pid"] = os.getpid()
            return ProverTaskResult(
                artifact_manifest_ref=manifest_ref,
                metrics=metrics,
                message="Proof computation completed successfully",
            )
        finally:
            self._stop_process(resource_proc)

    def _run_transfer_only(self, task: LocalProverTask) -> ProverTaskResult:
        parse_start = time.perf_counter()
        with open(task.input_path, "r", encoding="utf-8") as file:
            json.load(file)
        input_parse_time_s = time.perf_counter() - parse_start
        metrics = {
            "node_id": task.node_id,
            "transfer_only": True,
            "model_materialize_time(s)": task.model_materialize_time_s,
            "input_materialize_time(s)": task.input_materialize_time_s,
            "input_parse_time(s)": input_parse_time_s,
            "input_transfer_bytes": os.path.getsize(task.input_path),
            "prover_process_pid": os.getpid(),
            "phase_admission_enabled": False,
        }
        metrics.update(self._flatten_transfer_metrics(task.transfer_metrics))
        return ProverTaskResult(None, metrics, "Transfer-only benchmark completed")

    def _start_resource_logger(self, task: LocalProverTask, status_file: str):
        resource_usage_file = os.path.join(task.output_dir, "resource_usage.csv")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "zkinfer.profiling.resource_logger",
                "--output_file",
                resource_usage_file,
                "--pid",
                str(os.getpid()),
                "--interval",
                str(task.resource_monitor_interval_sec),
                "--status_file",
                status_file,
                "--include_children",
                "--exclude_pid",
                str(os.getpid()),
            ]
        )
        return process, resource_usage_file

    @staticmethod
    def _stop_process(process) -> None:
        if not process:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except Exception:
            process.kill()

    def _build_proof_stages(
        self,
        task: LocalProverTask,
        status_file: str,
    ) -> ResourceAwareEZKLProofStages:
        admission_dir = task.phase_admission_dir or os.path.join(task.tmp_dir, "phase-admission")
        admission = ResourceAdmissionController(
            root_dir=admission_dir,
            setup_key=task.setup_key,
            enabled=task.phase_admission_enabled,
            memory_reserve_gb=task.phase_memory_reserve_gb,
            cpu_budget_percent=task.phase_cpu_budget_percent,
            fallback_memory_gb=task.predicted_peak_memory_gb,
            fallback_cpu_percent=task.phase_fallback_cpu_percent,
            profile_percentile=task.phase_profile_percentile,
        )
        return ResourceAwareEZKLProofStages(
            input_data_path=task.input_path,
            onnx_model_path=task.model_path,
            artifact_dir=task.output_dir,
            proving_cache_enabled=True,
            proving_cache_overwrite=False,
            proving_cache_type="filesystem",
            proving_cache_root_dir=task.setup_cache_dir,
            proving_cache_s3_bucket=None,
            proving_cache_s3_prefix=None,
            status_file=status_file,
            local_tmp_dir=task.tmp_dir,
            logger=self.logger,
            admission=admission,
        )

    def _collect_metrics(
        self,
        task: LocalProverTask,
        resource_usage_file: str,
        proof_stages: EZKLProofStages,
        ezkl_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            resource_metrics = parse_resource_usage_file(resource_usage_file)
        except Exception as exc:
            self.logger.error("Failed to parse resource usage file: %s", exc, exc_info=True)
            resource_metrics = {
                "max_process_memory(GB)": None,
                "max_system_memory(GB)": None,
                "avg_process_cpu_raw(%)": None,
                "avg_process_cpu_machine(%)": None,
                "avg_system_cpu(%)": None,
            }

        return {
            "node_id": task.node_id,
            "predicted_peak_memory(GB)": task.predicted_peak_memory_gb,
            "model_materialize_time(s)": task.model_materialize_time_s,
            "input_materialize_time(s)": task.input_materialize_time_s,
            **resource_metrics,
            "pk_file_size(GB)": proof_stages.get_pk_file_size_gb(),
            "vk_file_size(GB)": proof_stages.get_vk_file_size_gb(),
            **read_csv_first_row(os.path.join(task.output_dir, "halo2_circuit.csv")),
            **read_csv_first_row(os.path.join(task.output_dir, "halo2_prover_cpu.csv")),
            **summarize_fft_report(os.path.join(task.output_dir, "halo2_ffts.csv")),
            **summarize_msm_report(os.path.join(task.output_dir, "halo2_msms.csv")),
            **ezkl_metrics,
        }

    @staticmethod
    def _same_path(a: str | Path, b: str | Path) -> bool:
        try:
            return Path(a).resolve() == Path(b).resolve()
        except OSError:
            return os.path.abspath(str(a)) == os.path.abspath(str(b))

    def _publish_file(
        self,
        local_path: str,
        task: LocalProverTask,
        artifact_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        started = time.perf_counter()
        size_bytes = os.path.getsize(local_path)
        if task.output_backend == "filesystem":
            output_path = os.path.join(task.output_prefix, artifact_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if not self._same_path(local_path, output_path):
                shutil.copy2(local_path, output_path)
            return output_path, {
                "artifact": artifact_name,
                "backend": "filesystem",
                "bytes_uploaded": 0,
                "bytes_written": size_bytes,
                "publish_time(s)": time.perf_counter() - started,
            }

        if task.output_backend == "s3":
            if not task.output_s3_bucket:
                raise ValueError("S3 output artifact requires a bucket")
            key = f"{task.output_prefix.rstrip('/')}/{artifact_name}"
            upload_file(local_path, task.output_s3_bucket, key)
            return key, {
                "artifact": artifact_name,
                "backend": "s3",
                "bytes_uploaded": size_bytes,
                "bytes_written": 0,
                "publish_time(s)": time.perf_counter() - started,
            }

        raise ValueError(f"Unsupported output backend: {task.output_backend}")

    def _publish_job_artifacts(
        self,
        task: LocalProverTask,
        proof_stages: EZKLProofStages,
    ) -> Tuple[ArtifactRef, Dict[str, Any]]:
        artifact_sources = {
            "proof.pf": proof_stages.proof_path,
            "witness.json": proof_stages.witness_path,
            "vk.json": proof_stages.vk_path,
            "settings.json": proof_stages.settings_path,
        }
        for artifact_name, source_path in artifact_sources.items():
            if not source_path or not os.path.exists(source_path):
                raise FileNotFoundError(
                    f"Required artifact {artifact_name} was not generated: {source_path}"
                )

        artifact_metadata: Dict[str, Dict[str, Any]] = {}
        publish_metrics: Dict[str, Any] = {
            "artifacts": {},
            "output_bytes_uploaded": 0,
            "output_bytes_written": 0,
            "output_publish_time(s)": 0.0,
        }
        tmp_dir = Path(task.tmp_dir).resolve()

        for artifact_name, source_path in artifact_sources.items():
            output_path = os.path.join(task.output_dir, artifact_name)
            source = Path(source_path).resolve()
            try:
                source.relative_to(tmp_dir)
                source_is_temporary = True
            except ValueError:
                source_is_temporary = False

            if not self._same_path(source, output_path):
                if source_is_temporary:
                    shutil.move(str(source), output_path)
                else:
                    shutil.copy2(str(source), output_path)

            published_uri, file_metrics = self._publish_file(
                output_path,
                task,
                artifact_name,
            )
            publish_metrics["artifacts"][artifact_name] = file_metrics
            publish_metrics["output_bytes_uploaded"] += int(
                file_metrics.get("bytes_uploaded", 0) or 0
            )
            publish_metrics["output_bytes_written"] += int(
                file_metrics.get("bytes_written", 0) or 0
            )
            publish_metrics["output_publish_time(s)"] += float(
                file_metrics.get("publish_time(s)", 0.0) or 0.0
            )
            artifact_metadata[artifact_name] = {
                "file": artifact_name,
                "local_path": output_path,
                "published_uri": published_uri,
                "size_bytes": os.path.getsize(output_path),
            }

        manifest = {
            "request_id": task.request_id,
            "job_id": task.job_id,
            "node_id": task.node_id,
            "setup_key": task.setup_key,
            "setup_cache_dir": task.setup_cache_dir,
            "storage_backend": task.output_backend,
            "s3_bucket": task.output_s3_bucket,
            "srs_path": proof_stages.srs_path,
            "artifacts": artifact_metadata,
            "transfer_metrics": task.transfer_metrics,
            "output_publish_metrics": publish_metrics,
        }
        manifest_path = os.path.join(task.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=4)

        published_manifest, manifest_metrics = self._publish_file(
            manifest_path,
            task,
            "manifest.json",
        )
        publish_metrics["artifacts"]["manifest.json"] = manifest_metrics
        publish_metrics["output_bytes_uploaded"] += int(
            manifest_metrics.get("bytes_uploaded", 0) or 0
        )
        publish_metrics["output_bytes_written"] += int(
            manifest_metrics.get("bytes_written", 0) or 0
        )
        publish_metrics["output_publish_time(s)"] += float(
            manifest_metrics.get("publish_time(s)", 0.0) or 0.0
        )

        return ArtifactRef(
            backend=task.output_backend,
            uri=published_manifest,
            s3_bucket=task.output_s3_bucket,
            size_bytes=os.path.getsize(manifest_path),
        ), publish_metrics

    @staticmethod
    def _flatten_transfer_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        model = metrics.get("model", {}) or {}
        input_metrics = metrics.get("input", {}) or {}
        setup = metrics.get("setup", {}) or {}
        return {
            "transfer_model_cache_hit": bool(model.get("cache_hit", False)),
            "transfer_input_cache_hit": bool(input_metrics.get("cache_hit", False)),
            "transfer_model_bytes_downloaded": int(model.get("bytes_downloaded", 0) or 0),
            "transfer_input_bytes_downloaded": int(input_metrics.get("bytes_downloaded", 0) or 0),
            "transfer_model_bytes_from_local_cache": int(
                model.get("bytes_from_local_cache", 0) or 0
            ),
            "transfer_input_bytes_from_local_cache": int(
                input_metrics.get("bytes_from_local_cache", 0) or 0
            ),
            "transfer_setup_bytes_downloaded": int(setup.get("bytes_downloaded", 0) or 0),
            "transfer_setup_bytes_from_local_cache": int(
                setup.get("bytes_from_local_cache", 0) or 0
            ),
            "transfer_setup_restore_time(s)": float(
                setup.get("restore_time(s)", 0.0) or 0.0
            ),
            "transfer_pk_decompress_time(s)": float(
                setup.get("pk_decompress_time(s)", 0.0) or 0.0
            ),
        }

    @staticmethod
    def _flatten_output_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "output_bytes_uploaded": int(metrics.get("output_bytes_uploaded", 0) or 0),
            "output_bytes_written": int(metrics.get("output_bytes_written", 0) or 0),
            "output_publish_time(s)": float(
                metrics.get("output_publish_time(s)", 0.0) or 0.0
            ),
        }
