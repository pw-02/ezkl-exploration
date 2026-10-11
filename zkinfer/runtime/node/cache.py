from __future__ import annotations

from zkinfer.runtime.worker_cache import WorkerLocalCache


class NodeCache(WorkerLocalCache):
    """Node-level cache shared by all local prover subprocesses.

    This class intentionally wraps the previous worker-local cache implementation
    while moving the abstraction to the correct level: one cache per node agent,
    not one cache per prover task.
    """

    pass
