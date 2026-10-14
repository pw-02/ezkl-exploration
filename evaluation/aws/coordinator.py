"""Evaluation-only registration snapshot; no change to placement decisions."""
import json
import os
from pathlib import Path
import hydra
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

@hydra.main(config_path=str(Path(__file__).resolve().parents[2] / 'zkinfer/config'),
            config_name='config', version_base=None)
def main(cfg):
    # Calling the imported decorated serve() makes Hydra search for a Python
    # config package. The runtime config is a directory, not such a package.
    # Compose it here using an absolute path, then invoke the undecorated server.
    serve.__wrapped__(cfg)

if __name__ == '__main__':
    main()
