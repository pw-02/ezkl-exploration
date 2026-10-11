from collections import defaultdict
from typing import Iterable, List, Optional

from zkinfer.runtime.models import NodeResources, ProofJob
from zkinfer.runtime.resource_profiles import (
    get_stage_estimate,
    max_stage_memory_gb,
)


class JobScheduler:
    """Request-fair scheduler with resource- and cache-aware placement.

    The coordinator leases whole proof jobs to nodes. Historical stage resource
    profiles are used to decide whether a job can start useful work on the node
    now. Node-local admission remains responsible for each later stage transition.
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

        # Fairness is applied before within-request ranking so a large request
        # cannot monopolize the cluster simply because it has more ready jobs.
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
            start_memory = self._starting_stage_memory_gb(job)
            if start_memory <= 0.0:
                start_memory = job.predicted_peak_memory_gb or 0.0
            memory_slack = max(node.memory_available_gb - start_memory, 0.0)

            if self.policy == "cache_aware":
                return (cache_hit, duration, -memory_slack, -idx)

            # resource_aware uses best-fit memory packing after cache locality.
            # A smaller positive slack means this job makes better use of the
            # memory the polling node reports as currently available.
            return (cache_hit, -memory_slack, duration, -idx)

        return max(candidates, key=score)[0]

    @staticmethod
    def _starting_stage(job: ProofJob) -> str:
        # A persisted profile means this setup key has completed before and its
        # reusable setup artifacts should exist in shared storage. The node may
        # still need to restore them locally, but proof computation can start at
        # witness generation once that restore completes.
        return "GENERATING_WITNESS" if job.profiling_data else "CALIBRATING"

    @classmethod
    def _starting_stage_memory_gb(cls, job: ProofJob) -> float:
        estimate = get_stage_estimate(job.profiling_data, cls._starting_stage(job))
        try:
            return max(0.0, float(estimate.get("memory_gb") or 0.0))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _fits_node(cls, job: ProofJob, node: NodeResources) -> bool:
        """Check permanent feasibility and whether the job can start now.

        Later high-memory stages are intentionally not required to fit the node's
        *current* free memory. The node-local admission controller can wait for
        those stages when other work releases resources. We only reject a job if
        its predicted peak cannot fit the machine at all, or if its first useful
        stage cannot fit the node's currently reported capacity.
        """
        predicted_peak = max_stage_memory_gb(job.profiling_data)
        if predicted_peak <= 0.0:
            predicted_peak = max(0.0, float(job.predicted_peak_memory_gb or 0.0))

        if predicted_peak > 0.0 and node.memory_total_gb > 0.0:
            if predicted_peak > node.memory_total_gb:
                return False

        start_memory = cls._starting_stage_memory_gb(job)
        if start_memory <= 0.0:
            # Without stage history, keep the conservative whole-job check.
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
