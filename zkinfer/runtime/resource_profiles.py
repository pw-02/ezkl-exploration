from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, Iterable, Optional


DEFAULT_PROFILE_PERCENTILE = 0.90
DEFAULT_HISTORY_LIMIT = 32
_RESOURCE_HISTORY_KEY = "_resource_history"
_STAGE_METRIC_RE = re.compile(
    r"^stage_(?P<stage>.+)_(?P<metric>max_memory\(GB\)|avg_cpu_machine\(%\))$"
)


def normalize_stage_name(stage: str) -> str:
    """Return the canonical stage key used in persisted resource profiles."""
    normalized = re.sub(r"[^a-z0-9]+", "_", (stage or "unknown").strip().lower())
    return normalized.strip("_") or "unknown"


def _as_positive_float(value: Any) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0.0 else 0.0


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values if float(value) > 0.0)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]

    percentile = min(1.0, max(0.0, float(percentile)))
    position = percentile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _conservative_estimate(values: Iterable[float], percentile: float) -> float:
    samples = [float(value) for value in values if float(value) > 0.0]
    if not samples:
        return 0.0
    if len(samples) < 5:
        return max(samples)
    return _percentile(samples, percentile)


def _observation_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    stages: Dict[str, Dict[str, float]] = {}
    for key, raw_value in report.items():
        match = _STAGE_METRIC_RE.match(str(key))
        if not match:
            continue
        value = _as_positive_float(raw_value)
        if value <= 0.0:
            continue
        stage = normalize_stage_name(match.group("stage"))
        metric = match.group("metric")
        stage_data = stages.setdefault(stage, {})
        if metric == "max_memory(GB)":
            stage_data["memory_gb"] = value
        elif metric == "avg_cpu_machine(%)":
            stage_data["cpu_machine_percent"] = value

    return {
        "recorded_at": time.time(),
        "job_runtime_s": _as_positive_float(report.get("job_runtime(s)")),
        "peak_memory_gb": _as_positive_float(
            report.get("max_process_memory(GB)")
            or report.get("max_system_memory(GB)")
        ),
        "stages": stages,
    }


def _estimate_history(
    observations: Iterable[Dict[str, Any]],
    percentile: float,
) -> Dict[str, Any]:
    observations = list(observations)
    duration_values = [
        _as_positive_float(item.get("job_runtime_s")) for item in observations
    ]
    peak_memory_values = [
        _as_positive_float(item.get("peak_memory_gb")) for item in observations
    ]

    stages: Dict[str, Dict[str, list[float]]] = {}
    for observation in observations:
        for stage, stage_data in (observation.get("stages") or {}).items():
            entry = stages.setdefault(
                normalize_stage_name(stage),
                {"memory_gb": [], "cpu_machine_percent": []},
            )
            memory_gb = _as_positive_float(stage_data.get("memory_gb"))
            cpu_percent = _as_positive_float(stage_data.get("cpu_machine_percent"))
            if memory_gb > 0.0:
                entry["memory_gb"].append(memory_gb)
            if cpu_percent > 0.0:
                entry["cpu_machine_percent"].append(cpu_percent)

    stage_estimates: Dict[str, Dict[str, Any]] = {}
    for stage, values in stages.items():
        memory_values = values["memory_gb"]
        cpu_values = values["cpu_machine_percent"]
        stage_estimates[stage] = {
            "memory_gb": _conservative_estimate(memory_values, percentile),
            "cpu_machine_percent": _conservative_estimate(cpu_values, percentile),
            "samples": max(len(memory_values), len(cpu_values)),
        }

    return {
        "job_runtime_s": _conservative_estimate(duration_values, percentile),
        "peak_memory_gb": _conservative_estimate(peak_memory_values, percentile),
        "samples": len(observations),
        "stages": stage_estimates,
    }


def update_resource_profile(
    existing_profile: Optional[Dict[str, Any]],
    latest_report: Dict[str, Any],
    setup_key: str,
    *,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    percentile: float = DEFAULT_PROFILE_PERCENTILE,
) -> Dict[str, Any]:
    """Merge one completed run into the persistent scheduling profile.

    The top level remains the latest job report so existing analysis code can
    continue to read familiar fields. Rolling observations and estimates used by
    the scheduler are stored under ``_resource_history``.
    """
    history_limit = max(4, int(history_limit))
    previous = existing_profile or {}
    previous_history = previous.get(_RESOURCE_HISTORY_KEY) or {}
    observations = list(previous_history.get("observations") or [])

    # Seed a legacy profiling.json once if it predates rolling history.
    if not observations and previous and _RESOURCE_HISTORY_KEY not in previous:
        legacy_observation = _observation_from_report(previous)
        if legacy_observation["peak_memory_gb"] > 0.0 or legacy_observation["stages"]:
            observations.append(legacy_observation)

    observations.append(_observation_from_report(latest_report))
    observations = observations[-history_limit:]

    result = dict(latest_report)
    result[_RESOURCE_HISTORY_KEY] = {
        "version": 1,
        "setup_key": setup_key,
        "updated_at": time.time(),
        "percentile": float(percentile),
        "observations": observations,
        "estimates": _estimate_history(observations, percentile),
    }
    return result


def get_profile_estimates(
    profile: Optional[Dict[str, Any]],
    *,
    percentile: float = DEFAULT_PROFILE_PERCENTILE,
) -> Dict[str, Any]:
    """Return conservative job and stage estimates from a persisted profile."""
    profile = profile or {}
    history = profile.get(_RESOURCE_HISTORY_KEY) or {}
    observations = history.get("observations") or []
    if observations:
        return _estimate_history(observations, percentile)

    # Legacy profiling.json files contain only the most recent job report.
    observation = _observation_from_report(profile)
    if observation["peak_memory_gb"] <= 0.0 and not observation["stages"]:
        return {"job_runtime_s": 0.0, "peak_memory_gb": 0.0, "samples": 0, "stages": {}}
    return _estimate_history([observation], percentile)


def get_stage_estimate(
    profile: Optional[Dict[str, Any]],
    stage: str,
    *,
    percentile: float = DEFAULT_PROFILE_PERCENTILE,
) -> Dict[str, Any]:
    estimates = get_profile_estimates(profile, percentile=percentile)
    return dict((estimates.get("stages") or {}).get(normalize_stage_name(stage)) or {})


def max_stage_memory_gb(
    profile: Optional[Dict[str, Any]],
    *,
    percentile: float = DEFAULT_PROFILE_PERCENTILE,
) -> float:
    estimates = get_profile_estimates(profile, percentile=percentile)
    stage_memory = [
        _as_positive_float(stage.get("memory_gb"))
        for stage in (estimates.get("stages") or {}).values()
    ]
    stage_peak = max(stage_memory, default=0.0)
    return max(stage_peak, _as_positive_float(estimates.get("peak_memory_gb")))
