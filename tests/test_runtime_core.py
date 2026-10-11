from __future__ import annotations

import multiprocessing as mp
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import NodeResources, ProofJob
from zkinfer.runtime.node.cache import NodeCache
from zkinfer.runtime.node.phase_admission import PhaseAdmissionGate
from zkinfer.runtime.scheduler import JobScheduler
from zkinfer.runtime.state_store import CoordinatorStateStore


def _job(request_id: str, name: str) -> ProofJob:
    empty = ArtifactRef.empty()
    return ProofJob(
        inference_request_name=request_id,
        model_name=name,
        inference_request_id=request_id,
        model_ref=empty,
        input_ref=empty,
        setup_cache_ref=empty,
        output_prefix_ref=empty,
        predicted_peak_memory_gb=1.0,
    )


def _wait_on_phase_gate(root: str, output: mp.Queue) -> None:
    gate = PhaseAdmissionGate(root, limit=1, poll_interval_s=0.02)
    started = time.perf_counter()
    with gate.hold("PROVING") as wait_s:
        output.put((wait_s, time.perf_counter() - started))


class SchedulerTests(unittest.TestCase):
    def test_requests_rotate_before_one_request_monopolizes_queue(self):
        scheduler = JobScheduler("fifo")
        scheduler.add_jobs(
            [
                _job("request-a", "a1"),
                _job("request-a", "a2"),
                _job("request-a", "a3"),
                _job("request-b", "b1"),
            ]
        )
        node = NodeResources(node_id="node", memory_available_gb=64)

        first = scheduler.next_job(node)
        second = scheduler.next_job(node)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.inference_request_id, second.inference_request_id)

    def test_coordinator_does_not_filter_on_local_slot_count(self):
        scheduler = JobScheduler("fifo")
        scheduler.add_jobs([_job("request-a", "a1")])
        node = NodeResources(
            node_id="node",
            memory_available_gb=64,
            max_concurrent_jobs=0,
            active_job_ids=("already-running",),
        )
        self.assertIsNotNone(scheduler.next_job(node))


class StateStoreMigrationTests(unittest.TestCase):
    def test_old_worker_assignment_column_is_migrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE jobs (
                        job_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        assigned_worker_id TEXT
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
                    ("job-1", "request-1", "IN_PROGRESS", "{}", "old-worker"),
                )

            CoordinatorStateStore(str(db_path))

            with sqlite3.connect(db_path) as conn:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
                }
                self.assertIn("assigned_node_id", columns)
                self.assertIn("lease_id", columns)
                self.assertIn("lease_updated_at", columns)
                node_id = conn.execute(
                    "SELECT assigned_node_id FROM jobs WHERE job_id = 'job-1'"
                ).fetchone()[0]
                self.assertEqual(node_id, "old-worker")


class NodeCacheTests(unittest.TestCase):
    def test_pinned_setup_directory_is_not_evicted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = NodeCache(tmp, max_bytes=32)
            pinned = Path(cache.setup_dir("pinned"))
            victim = Path(cache.setup_dir("victim"))
            (pinned / "data.bin").write_bytes(b"a" * 24)
            time.sleep(0.01)
            (victim / "data.bin").write_bytes(b"b" * 24)

            cache.pin_setup("pinned")
            try:
                cache.evict_if_needed()
                self.assertTrue(pinned.exists())
                self.assertFalse(victim.exists())
            finally:
                cache.unpin_setup("pinned")


class PhaseAdmissionTests(unittest.TestCase):
    def test_gate_is_shared_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = PhaseAdmissionGate(tmp, limit=1, poll_interval_s=0.02)
            output: mp.Queue = mp.Queue()

            with gate.hold("PROVING"):
                child = mp.Process(target=_wait_on_phase_gate, args=(tmp, output))
                child.start()
                time.sleep(0.15)
                self.assertTrue(output.empty())

            child.join(timeout=3)
            self.assertEqual(child.exitcode, 0)
            wait_s, elapsed_s = output.get(timeout=1)
            self.assertGreaterEqual(wait_s, 0.10)
            self.assertGreaterEqual(elapsed_s, 0.10)


if __name__ == "__main__":
    unittest.main()
