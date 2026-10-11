from typing import Iterable, List, Optional

from zkinfer.runtime.models import ProofJob, WorkerResources


class JobScheduler:
    """Ready-queue scheduler for proof jobs.

    gRPC remains a control plane: workers report capacity and cache state, then
    the coordinator returns references to data-plane artifacts.  The scheduler
    never moves bytes itself; it only chooses a feasible job.
    """

    def __init__(self, policy: str = "fifo"):
        self.policy = (policy or "fifo").lower()

        if self.policy not in {"fifo", "lpt", "resource_aware", "artifact_aware"}:
            raise ValueError(f"Unsupported scheduling policy: {self.policy}")

        self._queue: List[ProofJob] = []

    def add_jobs(self, jobs: Iterable[ProofJob]) -> None:
        for job in jobs:
            self.requeue(job)

    def next_job(self, worker: Optional[WorkerResources] = None) -> Optional[ProofJob]:
        if not self._queue:
            return None

        worker = worker or WorkerResources(worker_id="unknown")
        candidate_index = self._select_index(worker)

        if candidate_index is None:
            return None

        return self._queue.pop(candidate_index)

    def requeue(self, job: ProofJob) -> None:
        if self.policy == "fifo":
            self._queue.append(job)
            return

        # Keep LPT-like queues mostly ordered so retries do not fall to the end.
        duration = job.predicted_duration or 0.0
        for idx, queued_job in enumerate(self._queue):
            queued_duration = queued_job.predicted_duration or 0.0
            if duration > queued_duration:
                self._queue.insert(idx, job)
                return

        self._queue.append(job)

    def _select_index(self, worker: WorkerResources) -> Optional[int]:
        feasible = [
            (idx, job)
            for idx, job in enumerate(self._queue)
            if self._fits_worker(job, worker)
        ]

        if not feasible:
            return None

        if self.policy == "fifo":
            return feasible[0][0]

        if self.policy == "lpt":
            return max(
                feasible,
                key=lambda item: item[1].predicted_duration or 0.0,
            )[0]

        cached = set(worker.cached_artifact_keys or [])

        def score(item):
            idx, job = item
            locality = 1 if job.artifact_key in cached else 0
            duration = job.predicted_duration or 0.0
            memory = job.predicted_peak_memory_gb or 0.0

            if self.policy == "artifact_aware":
                return (locality, duration, -idx)

            # resource_aware: prefer locality, then jobs with larger memory and
            # longer duration so scarce/high-cost jobs start early.
            return (locality, memory, duration, -idx)

        return max(feasible, key=score)[0]

    @staticmethod
    def _fits_worker(job: ProofJob, worker: WorkerResources) -> bool:
        if worker.free_slots <= 0:
            return False

        required = job.predicted_peak_memory_gb or 0.0
        if required <= 0:
            return True

        return required <= worker.available_memory_gb

    def clear(self) -> None:
        self._queue.clear()

    def empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)
