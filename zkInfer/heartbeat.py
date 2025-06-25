# zkInfer/heartbeat_watcher.py

import time
import sys
import grpc
import zkservice_pb2 as pb, zkservice_pb2_grpc as pb_grpc
import psutil
import os
target, worker_id, job_id, sub_job_id, status_file, parent_pid = sys.argv[1:7]
parent_pid = int(parent_pid)

channel = grpc.insecure_channel(target)
stub = pb_grpc.ZKJobServiceStub(channel)

while True:
    try:
        if not psutil.pid_exists(parent_pid):
            print("Parent died, exiting heartbeat.")
            sys.exit(0)
        # check if the status file exists
        if os.path.exists(status_file):
            with open(status_file) as f:
                stage = f.read().strip()
            stub.SendHeartbeat(
                pb.HeartbeatRequest(
                    worker_id=worker_id,
                    sub_job_id=sub_job_id,
                    job_id=job_id,
                    status="STARTED" if stage not in ["DONE", "FAILED"] else stage,
                    message=stage
                )
            )
        if stage and stage in ("DONE", "FAILED"):
                break
    except Exception as e:
        print("Heartbeat exception:", e, file=sys.stderr)
    time.sleep(15)
channel.close()
