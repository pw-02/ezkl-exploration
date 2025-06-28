import time
import sys
import grpc
import zkservice_pb2 as pb
import zkservice_pb2_grpc as pb_grpc
import psutil
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--worker_id", required=True)
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--status_file", required=True)
    parser.add_argument("--parent_pid", required=True, type=int)
    parser.add_argument("--interval", default=15, type=int, help="Seconds between heartbeats")
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.target)
    stub = pb_grpc.ZKJobServiceStub(channel)

    while True:
        try:
            if not psutil.pid_exists(args.parent_pid):
                print("Parent died, exiting heartbeat.")
                sys.exit(0)

            status_msg = None
            if os.path.exists(args.status_file):
                with open(args.status_file) as f:
                    status_msg = f.read().strip()
                stub.SendHeartbeat(
                    pb.HeartbeatRequest(
                        worker_id=args.worker_id,
                        job_id=args.job_id,
                        status=status_msg.upper() if status_msg else None,
                        message=f"Worker {args.worker_id} is alive",
                    )
                )
            if status_msg is not None:
                if status_msg.upper() in ("DONE", "FAILED"):
                    break
        except Exception as e:
            print("Heartbeat exception:", e, file=sys.stderr)
        time.sleep(args.interval)
    channel.close()

if __name__ == "__main__":
    main()
