import argparse
import logging
import os
import sys
import time
from typing import Optional

import grpc
import psutil

import zkinfer.proto.zkservice_pb2 as pb
import zkinfer.proto.zkservice_pb2_grpc as pb_grpc


TERMINAL_STATUSES = {"DONE", "FAILED", "COMPLETED"}


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] heartbeat: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("heartbeat")


def read_status(status_file: str) -> str:
    if not os.path.exists(status_file):
        return "UNKNOWN"

    with open(status_file, "r", encoding="utf-8") as file:
        status = file.read().strip()

    return status.upper() if status else "UNKNOWN"


def send_heartbeat(
    stub,
    worker_id: str,
    job_id: str,
    status: str,
    message: str,
) -> None:
    stub.SendHeartbeat(
        pb.HeartbeatRequest(
            worker_id=worker_id,
            job_id=job_id,
            status=status,
            message=message,
        )
    )


def run_heartbeat(
    target: str,
    worker_id: str,
    job_id: str,
    status_file: str,
    parent_pid: int,
    interval_sec: int,
    logger: logging.Logger,
) -> None:
    channel = grpc.insecure_channel(target)
    stub = pb_grpc.ZKJobServiceStub(channel)

    try:
        while True:
            if not psutil.pid_exists(parent_pid):
                logger.info("Parent process %s exited; stopping heartbeat.", parent_pid)
                return

            status = read_status(status_file)

            try:
                send_heartbeat(
                    stub=stub,
                    worker_id=worker_id,
                    job_id=job_id,
                    status=status,
                    message=f"Worker {worker_id} is alive",
                )
            except grpc.RpcError as exc:
                logger.warning("Failed to send heartbeat: %s", exc)

            if status in TERMINAL_STATUSES:
                logger.info("Observed terminal status %s; stopping heartbeat.", status)
                return

            time.sleep(interval_sec)

    finally:
        channel.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--worker_id", required=True)
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--status_file", required=True)
    parser.add_argument("--parent_pid", required=True, type=int)
    parser.add_argument("--interval_sec", default=5, type=int)
    return parser.parse_args()


def main() -> None:
    logger = setup_logger()
    args = parse_args()

    run_heartbeat(
        target=args.target,
        worker_id=args.worker_id,
        job_id=args.job_id,
        status_file=args.status_file,
        parent_pid=args.parent_pid,
        interval_sec=args.interval_sec,
        logger=logger,
    )


if __name__ == "__main__":
    main()