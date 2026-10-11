from collections import defaultdict
from typing import Iterable, List, Optional

from zkinfer.runtime.models import NodeResources, ProofJob


class JobScheduler:
    """Resource-aware, request-fair ready-queue scheduler.

    Nodes report fresh resources when polling. The coordinator chooses one
    feasible proof job; local concurrency remains a node-supervisor concern.
    Fairness is enforced across inference requests before applying the selected
    within-request ranking policy.
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
        # Queue order is intentionally simple. Fairness and policy are evaluated
        # against the fresh node state at dispatch time.
        self._queue.append(job)

    def _select_index(self, node: NodeResources) -> Optional[int]:
        feasible = [
            (idx, job)
            for idx, job in enumerate(self._queue)
            if self._fits_node(job, node)
        ]
        if not feasible:
            return None

        # First choose the least-served request. Last-dispatch breaks ties so
        # requests with equal service counts rotate rather than stick to one.
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
            memory = job.predicted_peak_memory_gb or 0.0
            memory_slack = max(node.memory_available_gb - memory, 0.0)
            if self.policy == "cache_aware":
                return (cache_hit, duration, -idx)
            return (cache_hit, -memory_slack, duration, -idx)

        return max(candidates, key=score)[0]

    @staticmethod
    def _fits_node(job: ProofJob, node: NodeResources) -> bool:
        required_memory = job.predicted_peak_memory_gb or 0.0
        if required_memory > 0 and node.memory_available_gb > 0:
            if required_memory > node.memory_available_gb:
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
