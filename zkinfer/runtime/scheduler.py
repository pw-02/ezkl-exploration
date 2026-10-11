from collections import defaultdict
from typing import Iterable, List, Optional

from zkinfer.runtime.models import NodeResources, ProofJob
from zkinfer.runtime.resource_profiles import (
    get_stage_estimate,
    max_stage_memory_gb,
)


class JobScheduler:
    """Request-fair scheduler with selectable placement policies.

    The coordinator leases whole proof jobs to nodes. ``resource_aware`` uses
    historical stage resource profiles for placement. Other policies retain the
    conservative whole-job memory check so they remain useful evaluation
    baselines. Node-local admission controls later stage transitions.
    """

    def __init__(self, policy: str = "fifo"):
        self.policy = (policy or "fifo").lower()
        self.policy = {"artifact_aware": "cache_aware"}.get(self.policy, self.policy)
        if self.policy not in {"fifo", "lpt", "resource_aware", "cache_aware"}:
            raise ValueError(f"Unsupported scheduling policy: {self.policy}")
        self._queue: List[ProofJob] = []
        self._request_dispatch_count = defaultdict(int)
        self._dispatch_sequence = 0
        self._request_last_dispatch = defaultdict(lambda: -1)

    def add_jobs(self, jobs: Iterable[ProofJob]) -> None:
        for job in jobs:
            self.requeue(job)

    def next_job(self, node: Optional[NodeResources] = None) -> Optional[ProofJob]:
        if not self._queue:
            return None
        node = node or NodeResources(node_id="unknown")
        candidate_index = self._select_index(node)
        if candidate_index is None:
            return None
        job = self._queue.pop(candidate_index)
        request_id = job.inference_request_id
        self._request_dispatch_count[request_id] += 1
        self._dispatch_sequence += 1
        self._request_last_dispatch[request_id] = self._dispatch_sequence
        return job

    def requeue(self, job: ProofJob) -> None:
        self._queue.append(job)

    def _select_index(self, node: NodeResources) -> Optional[int]:
        feasible = [
            (idx, job)
            for idx, job in enumerate(self._queue)
            if self._fits_node(job, node)
        ]
        if not feasible:
            return None

        request_ids = {job.inference_request_id for _, job in feasible}
        chosen_request = min(
            request_ids,
            key=lambda request_id: (
                self._request_dispatch_count[request_id],
                self._request_last_dispatch[request_id],
            ),
        )
        candidates = [
            item for item in feasible if item[1].inference_request_id == chosen_request
        ]

        if self.policy == "fifo":
            return min(candidates, key=lambda item: item[0])[0]

        if self.policy == "lpt":
            return max(
                candidates,
                key=lambda item: (item[1].predicted_duration_s or 0.0, -item[0]),
            )[0]

        cached = set(node.cached_setup_keys or [])

        def score(item):
            idx, job = item
            cache_hit = 1 if job.artifact_key in cached else 0
            duration = job.predicted_duration_s or 0.0

            if self.policy == "cache_aware":
                return (cache_hit, duration, -idx)

            start_memory = self._starting_stage_memory_gb(job)
            if start_memory <= 0.0:
                start_memory = job.predicted_peak_memory_gb or 0.0
            memory_slack = max(node.memory_available_gb - start_memory, 0.0)

            # Prefer an existing setup copy, then choose the job whose starting
            # stage most closely fits currently available memory. Duration is a
            # final tie-breaker so long jobs start earlier when fit is similar.
            return (cache_hit, -memory_slack, duration, -idx)

        return max(candidates, key=score)[0]

    @staticmethod
    def _starting_stage(job: ProofJob) -> str:
        # A persisted profile implies that this setup key completed previously
        # and reusable setup artifacts should exist in shared storage. A node may
        # still restore them locally before witness generation begins.
        return "GENERATING_WITNESS" if job.profiling_data else "CALIBRATING"

    @classmethod
    def _starting_stage_memory_gb(cls, job: ProofJob) -> float:
        estimate = get_stage_estimate(job.profiling_data, cls._starting_stage(job))
        try:
            return max(0.0, float(estimate.get("memory_gb") or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _fits_node(self, job: ProofJob, node: NodeResources) -> bool:
        if self.policy != "resource_aware":
            required_memory = max(0.0, float(job.predicted_peak_memory_gb or 0.0))
            if required_memory > 0.0 and node.memory_available_gb > 0.0:
                return required_memory <= node.memory_available_gb
            return True

        predicted_peak = max_stage_memory_gb(job.profiling_data)
        if predicted_peak <= 0.0:
            predicted_peak = max(0.0, float(job.predicted_peak_memory_gb or 0.0))

        # A job should never be placed on a machine that cannot hold its
        # predicted peak even after all other work drains.
        if predicted_peak > 0.0 and node.memory_total_gb > 0.0:
            if predicted_peak > node.memory_total_gb:
                return False

        start_memory = self._starting_stage_memory_gb(job)
        if start_memory <= 0.0:
            # No stage history yet: use the whole-job estimate until the first
            # completed run provides stage-specific measurements.
            start_memory = predicted_peak

        if start_memory > 0.0 and node.memory_available_gb > 0.0:
            if start_memory > node.memory_available_gb:
                return False

        return True

    def clear(self) -> None:
        self._queue.clear()
        self._request_dispatch_count.clear()
        self._request_last_dispatch.clear()
        self._dispatch_sequence = 0

    def empty(self) -> bool:
        return not self._queue

    def __len__(self) -> int:
        return len(self._queue)
