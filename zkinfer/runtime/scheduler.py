from collections import deque
from typing import Deque, Iterable, Optional

from zkinfer.runtime.models import ProofJob


class JobScheduler:
    def __init__(self, policy: str = "fifo"):
        self.policy = (policy or "fifo").lower()

        if self.policy not in {"fifo", "lpt"}:
            raise ValueError(f"Unsupported scheduling policy: {self.policy}")

        self._queue: Deque[ProofJob] = deque()

    def add_jobs(self, jobs: Iterable[ProofJob]) -> None:
        for job in jobs:
            self.requeue(job)

    def next_job(self) -> Optional[ProofJob]:
        if not self._queue:
            return None

        return self._queue.popleft()

    def requeue(self, job: ProofJob) -> None:
        if self.policy == "fifo":
            self._queue.append(job)
            return

        # LPT keeps the ready queue ordered by predicted duration.  This also
        # keeps retried/requeued jobs from falling to the end after a failure.
        duration = job.predicted_duration or 0.0
        for idx, queued_job in enumerate(self._queue):
            queued_duration = queued_job.predicted_duration or 0.0
            if duration > queued_duration:
                self._queue.insert(idx, job)
                return

        self._queue.append(job)

    def clear(self) -> None:
        self._queue.clear()

    def empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)
