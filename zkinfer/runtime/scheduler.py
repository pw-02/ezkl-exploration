from typing import Iterable, List, Optional

from zkinfer.runtime.models import NodeResources, ProofJob


class JobScheduler:
    """Ready-queue scheduler for node-agent polling.

    The coordinator does not assign fixed worker slots. A node agent reports
    fresh memory, CPU, cache, and active-job state when it polls. The scheduler
    then chooses one feasible job for that node and returns a lease.
    """

    def __init__(self, policy: str = "fifo"):
        self.policy = (policy or "fifo").lower()
        aliases = {"artifact_aware": "cache_aware"}
        self.policy = aliases.get(self.policy, self.policy)

        if self.policy not in {"fifo", "lpt", "resource_aware", "cache_aware"}:
            raise ValueError(f"Unsupported scheduling policy: {self.policy}")

        self._queue: List[ProofJob] = []

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
        return self._queue.pop(candidate_index)

    def requeue(self, job: ProofJob) -> None:
        if self.policy == "fifo":
            self._queue.append(job)
            return

        duration = job.predicted_duration_s or 0.0
        for idx, queued_job in enumerate(self._queue):
            queued_duration = queued_job.predicted_duration_s or 0.0
            if duration > queued_duration:
                self._queue.insert(idx, job)
                return
        self._queue.append(job)

    def _select_index(self, node: NodeResources) -> Optional[int]:
        feasible = [
            (idx, job)
            for idx, job in enumerate(self._queue)
            if self._fits_node(job, node)
        ]
        if not feasible:
            return None

        if self.policy == "fifo":
            return feasible[0][0]

        if self.policy == "lpt":
            return max(feasible, key=lambda item: item[1].predicted_duration_s or 0.0)[0]

        cached = set(node.cached_setup_keys or [])

        def score(item):
            idx, job = item
            setup_cache_hit = 1 if job.artifact_key in cached else 0
            duration = job.predicted_duration_s or 0.0
            memory = job.predicted_peak_memory_gb or 0.0
            memory_slack = node.memory_available_gb - memory

            if self.policy == "cache_aware":
                return (setup_cache_hit, duration, -idx)

            # resource_aware: prefer cached jobs, then pack memory fairly tightly,
            # then start long jobs early.
            return (setup_cache_hit, -max(memory_slack, 0.0), duration, -idx)

        return max(feasible, key=score)[0]

    @staticmethod
    def _fits_node(job: ProofJob, node: NodeResources) -> bool:
        if not node.local_admission_available:
            return False

        required_memory = job.predicted_peak_memory_gb or 0.0
        if required_memory > 0 and node.memory_available_gb > 0:
            if required_memory > node.memory_available_gb:
                return False

        return True

    def clear(self) -> None:
        self._queue.clear()

    def empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)
