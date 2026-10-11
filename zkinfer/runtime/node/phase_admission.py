from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import psutil

try:
    import fcntl
except ImportError as exc:  # pragma: no cover - zkInfer nodes are Linux targets
    raise RuntimeError("phase-aware admission requires POSIX file locking") from exc


_STAGE_METRIC_RE = re.compile(
    r"^stage_(?P<stage>.+)_(?P<metric>max_memory\(GB\)|avg_cpu_machine\(%\))$"
)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    q = min(1.0, max(0.0, float(q)))
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_stage(stage: str) -> str:
    stage = (stage or "UNKNOWN").strip().lower()
    stage = re.sub(r"[^a-z0-9]+", "_", stage).strip("_")
    return stage or "unknown"


class HistoricalResourceProfiles:
    """Persistent per-setup-key stage resource observations.

    Profiles are intentionally node-local. They describe how a particular
    circuit/setup key behaved on this machine class and are updated after each
    successful proof. The admission controller uses the configured percentile
    of recent observations rather than a manually chosen process count.
    """

    def __init__(self, root_dir: str | Path, max_observations: int = 32):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_observations = max(4, int(max_observations))
        self.lock_path = self.root_dir / "profiles.lock"

    def _profile_path(self, setup_key: str) -> Path:
        digest = hashlib.sha256(setup_key.encode("utf-8")).hexdigest()
        return self.root_dir / f"{digest}.json"

    def _load_unlocked(self, setup_key: str) -> Dict[str, Any]:
        path = self._profile_path(setup_key)
        if not path.exists():
            return {"setup_key": setup_key, "stages": {}}
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError("profile is not a JSON object")
            data.setdefault("setup_key", setup_key)
            data.setdefault("stages", {})
            return data
        except (OSError, ValueError, json.JSONDecodeError):
            return {"setup_key": setup_key, "stages": {}}

    def load(self, setup_key: str) -> Dict[str, Any]:
        with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                return self._load_unlocked(setup_key)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def update_from_metrics(self, setup_key: str, metrics: Dict[str, Any]) -> None:
        if not setup_key or not metrics:
            return

        observed: Dict[str, Dict[str, float]] = {}
        for key, raw_value in metrics.items():
            match = _STAGE_METRIC_RE.match(str(key))
            if not match:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            stage = match.group("stage")
            metric = match.group("metric")
            observed.setdefault(stage, {})[metric] = value

        if not observed:
            return

        with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                profile = self._load_unlocked(setup_key)
                stages = profile.setdefault("stages", {})
                for stage, values in observed.items():
                    entry = stages.setdefault(stage, {"observations": []})
                    observations = list(entry.get("observations") or [])
                    observations.append(
                        {
                            "memory_gb": float(values.get("max_memory(GB)", 0.0)),
                            "cpu_machine_percent": float(
                                values.get("avg_cpu_machine(%)", 0.0)
                            ),
                            "recorded_at": time.time(),
                        }
                    )
                    entry["observations"] = observations[-self.max_observations :]
                profile["updated_at"] = time.time()

                path = self._profile_path(setup_key)
                tmp_path = path.with_suffix(".json.tmp")
                with open(tmp_path, "w", encoding="utf-8") as file:
                    json.dump(profile, file, indent=2, sort_keys=True)
                os.replace(tmp_path, path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def estimate(
        self,
        setup_key: str,
        stage: str,
        fallback_memory_gb: float,
        fallback_cpu_percent: float,
        percentile: float,
    ) -> Dict[str, Any]:
        profile = self.load(setup_key)
        stages = profile.get("stages") or {}

        def estimate_one(stage_name: str) -> Optional[Dict[str, Any]]:
            entry = stages.get(_metric_stage(stage_name)) or {}
            observations = entry.get("observations") or []
            memory_values = [
                float(item.get("memory_gb") or 0.0)
                for item in observations
                if float(item.get("memory_gb") or 0.0) > 0
            ]
            cpu_values = [
                float(item.get("cpu_machine_percent") or 0.0)
                for item in observations
                if float(item.get("cpu_machine_percent") or 0.0) > 0
            ]
            if not memory_values and not cpu_values:
                return None
            memory_gb = (
                max(memory_values)
                if len(memory_values) < 5
                else _percentile(memory_values, percentile)
            )
            cpu_percent = (
                max(cpu_values)
                if len(cpu_values) < 5
                else _percentile(cpu_values, percentile)
            )
            return {
                "memory_gb": memory_gb,
                "cpu_percent": cpu_percent,
                "samples": max(len(memory_values), len(cpu_values)),
                "source": "historical",
            }

        if stage == "FUSED_KEYGEN_PROVE":
            parts = [estimate_one("KEY_GEN"), estimate_one("PROVING")]
            parts = [item for item in parts if item is not None]
            if parts:
                return {
                    "memory_gb": max(item["memory_gb"] for item in parts),
                    "cpu_percent": max(item["cpu_percent"] for item in parts),
                    "samples": min(item["samples"] for item in parts),
                    "source": "historical_fused",
                }
        else:
            direct = estimate_one(stage)
            if direct is not None:
                return direct

        return {
            "memory_gb": max(0.0, float(fallback_memory_gb or 0.0)),
            "cpu_percent": max(0.0, float(fallback_cpu_percent or 0.0)),
            "samples": 0,
            "source": "fallback",
        }


class ResourceAdmissionController:
    """Cross-process phase admission based on predicted memory and CPU demand.

    Every prover subprocess on a node shares the same reservation directory.
    Before entering a stage, a process reserves the resources predicted for that
    stage from historical runs of the same setup key. If no stage history exists,
    the job's whole-job peak-memory estimate is used conservatively.

    No configured job count or heavy-phase count is part of the policy.
    """

    def __init__(
        self,
        root_dir: str,
        setup_key: str,
        enabled: bool,
        memory_reserve_gb: float,
        cpu_budget_percent: float,
        fallback_memory_gb: float,
        fallback_cpu_percent: float = 100.0,
        profile_percentile: float = 0.90,
        poll_interval_s: float = 0.25,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.setup_key = setup_key
        self.enabled = bool(enabled)
        self.memory_reserve_gb = max(0.0, float(memory_reserve_gb or 0.0))
        self.cpu_budget_percent = min(100.0, max(1.0, float(cpu_budget_percent or 100.0)))
        self.fallback_memory_gb = max(0.0, float(fallback_memory_gb or 0.0))
        self.fallback_cpu_percent = min(
            100.0, max(0.0, float(fallback_cpu_percent or 0.0))
        )
        self.profile_percentile = min(1.0, max(0.5, float(profile_percentile)))
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.lock_path = self.root_dir / "admission.lock"
        self.reservations_dir = self.root_dir / "reservations"
        self.reservations_dir.mkdir(parents=True, exist_ok=True)
        self.holder_path = self.reservations_dir / f"{os.getpid()}.json"
        self.profiles = HistoricalResourceProfiles(
            self.root_dir / "resource_profiles"
        )

    def _clean_stale_reservations(self) -> None:
        for path in self.reservations_dir.glob("*.json"):
            try:
                pid = int(path.stem)
            except ValueError:
                path.unlink(missing_ok=True)
                continue
            if not psutil.pid_exists(pid):
                path.unlink(missing_ok=True)

    def _read_reservations(self) -> list[Dict[str, Any]]:
        reservations: list[Dict[str, Any]] = []
        for path in self.reservations_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                if isinstance(data, dict):
                    reservations.append(data)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return reservations

    def estimate(self, stage: str) -> Dict[str, Any]:
        memory = psutil.virtual_memory()
        usable_memory_gb = max(
            0.0,
            float(memory.total) / (1024**3) - self.memory_reserve_gb,
        )
        fallback_memory = self.fallback_memory_gb or usable_memory_gb
        estimate = self.profiles.estimate(
            setup_key=self.setup_key,
            stage=stage,
            fallback_memory_gb=fallback_memory,
            fallback_cpu_percent=self.fallback_cpu_percent,
            percentile=self.profile_percentile,
        )
        estimate["memory_gb"] = max(0.0, float(estimate.get("memory_gb") or 0.0))
        estimate["cpu_percent"] = min(
            100.0, max(0.0, float(estimate.get("cpu_percent") or 0.0))
        )
        return estimate

    def acquire(self, stage: str) -> Dict[str, Any]:
        estimate = self.estimate(stage)
        started = time.perf_counter()

        if not self.enabled:
            return {
                **estimate,
                "wait_time_s": 0.0,
                "enabled": False,
                "stage": stage,
            }

        memory = psutil.virtual_memory()
        total_memory_gb = float(memory.total) / (1024**3)
        memory_budget_gb = max(0.0, total_memory_gb - self.memory_reserve_gb)
        requested_memory_gb = float(estimate["memory_gb"])
        requested_cpu_percent = float(estimate["cpu_percent"])

        if requested_memory_gb > memory_budget_gb:
            raise RuntimeError(
                f"Predicted {stage} memory {requested_memory_gb:.2f} GB exceeds "
                f"node admission budget {memory_budget_gb:.2f} GB"
            )

        while True:
            with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    self._clean_stale_reservations()
                    reservations = [
                        item
                        for item in self._read_reservations()
                        if int(item.get("pid") or -1) != os.getpid()
                    ]
                    reserved_memory_gb = sum(
                        float(item.get("memory_gb") or 0.0) for item in reservations
                    )
                    reserved_cpu_percent = sum(
                        float(item.get("cpu_percent") or 0.0) for item in reservations
                    )

                    memory_fits = (
                        reserved_memory_gb + requested_memory_gb <= memory_budget_gb
                    )
                    cpu_fits = (
                        reserved_cpu_percent + requested_cpu_percent
                        <= self.cpu_budget_percent
                    )

                    if memory_fits and cpu_fits:
                        reservation = {
                            "pid": os.getpid(),
                            "setup_key": self.setup_key,
                            "stage": stage,
                            "memory_gb": requested_memory_gb,
                            "cpu_percent": requested_cpu_percent,
                            "estimate_source": estimate.get("source", "fallback"),
                            "estimate_samples": int(estimate.get("samples") or 0),
                            "acquired_at": time.time(),
                        }
                        with open(self.holder_path, "w", encoding="utf-8") as file:
                            json.dump(reservation, file, sort_keys=True)
                        return {
                            **estimate,
                            "wait_time_s": time.perf_counter() - started,
                            "enabled": True,
                            "stage": stage,
                        }
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            time.sleep(self.poll_interval_s)

    def release(self) -> None:
        if not self.enabled:
            return
        with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self.holder_path.unlink(missing_ok=True)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def current_reservations(self) -> Dict[str, float]:
        with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._clean_stale_reservations()
                reservations = self._read_reservations()
                return {
                    "count": float(len(reservations)),
                    "memory_gb": sum(
                        float(item.get("memory_gb") or 0.0) for item in reservations
                    ),
                    "cpu_percent": sum(
                        float(item.get("cpu_percent") or 0.0) for item in reservations
                    ),
                }
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def hold(self, stage: str) -> Iterator[Dict[str, Any]]:
        admission = self.acquire(stage)
        try:
            yield admission
        finally:
            self.release()
