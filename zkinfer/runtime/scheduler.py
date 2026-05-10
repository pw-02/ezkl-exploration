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
        jobs = list(jobs)

        if self.policy == "lpt":
            jobs.sort(
                key=lambda job: job.predicted_duration or 0.0,
                reverse=True,
            )

        self._queue.extend(jobs)

    def next_job(self) -> Optional[ProofJob]:
        if not self._queue:
            return None

        return self._queue.popleft()

    def requeue(self, job: ProofJob) -> None:
        self._queue.append(job)

    def clear(self) -> None:
        self._queue.clear()

    def empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)