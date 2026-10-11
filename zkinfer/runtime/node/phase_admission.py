from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psutil

try:
    import fcntl
except ImportError as exc:  # pragma: no cover - zkInfer nodes are Linux targets
    raise RuntimeError("phase-aware admission requires POSIX file locking") from exc


class PhaseAdmissionGate:
    """Small interprocess semaphore for memory-heavy prover phases.

    Every prover subprocess on a node points at the same gate directory. A job
    waits before entering KEY_GEN/PROVING until fewer than ``limit`` heavy
    phases are active. Holder files contain the owning PID and are cleaned if a
    process dies unexpectedly.
    """

    def __init__(self, root_dir: str, limit: int, poll_interval_s: float = 0.25):
        self.root_dir = Path(root_dir)
        self.limit = max(1, int(limit or 1))
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root_dir / "gate.lock"
        self.holders_dir = self.root_dir / "holders"
        self.holders_dir.mkdir(parents=True, exist_ok=True)
        self.holder_path = self.holders_dir / f"{os.getpid()}.json"

    def _clean_stale_holders(self) -> None:
        for path in self.holders_dir.glob("*.json"):
            try:
                pid = int(path.stem)
            except ValueError:
                path.unlink(missing_ok=True)
                continue
            if not psutil.pid_exists(pid):
                path.unlink(missing_ok=True)

    def acquire(self, stage: str) -> float:
        started = time.perf_counter()
        while True:
            with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    self._clean_stale_holders()
                    holders = list(self.holders_dir.glob("*.json"))
                    if self.holder_path.exists() or len(holders) < self.limit:
                        self.holder_path.write_text(
                            json.dumps(
                                {
                                    "pid": os.getpid(),
                                    "stage": stage,
                                    "acquired_at": time.time(),
                                }
                            ),
                            encoding="utf-8",
                        )
                        return time.perf_counter() - started
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            time.sleep(self.poll_interval_s)

    def release(self) -> None:
        with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self.holder_path.unlink(missing_ok=True)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def hold(self, stage: str) -> Iterator[float]:
        wait_time_s = self.acquire(stage)
        try:
            yield wait_time_s
        finally:
            self.release()
