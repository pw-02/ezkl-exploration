"""Evaluation-only registration snapshot; no change to placement decisions."""
import json
import os
from pathlib import Path
from zkinfer.runtime.coordinator import Coordinator
from zkinfer.runtime.coordinator_grpc import serve

original = Coordinator._record_node_locked

def record(self, node, status):
    original(self, node, status)
    path = Path(os.environ['ZKINFER_READY_FILE'])
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(self.node_status, default=str))
    tmp.replace(path)

Coordinator._record_node_locked = record
if __name__ == '__main__':
    serve()
