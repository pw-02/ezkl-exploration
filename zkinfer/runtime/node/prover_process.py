from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from zkinfer.runtime.node.prover_task import LocalProverTask, ProverTaskRunner


def _write_json_atomic(path: str, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=destination.name + ".", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, destination)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_task(task_file: str, result_file: str) -> int:
    started = time.perf_counter()
    with open(task_file, "r", encoding="utf-8") as file:
        task = LocalProverTask.from_dict(json.load(file))

    logger = logging.getLogger(f"prover-process-{task.job_id}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    try:
        result = ProverTaskRunner(logger=logger).run(task)
        payload = {
            "ok": True,
            "pid": os.getpid(),
            "wall_time_s": time.perf_counter() - started,
            "result": result.to_dict(),
        }
        _write_json_atomic(result_file, payload)
        return 0
    except Exception as exc:
        logger.exception("Prover process failed")
        _write_json_atomic(
            result_file,
            {
                "ok": False,
                "pid": os.getpid(),
                "wall_time_s": time.perf_counter() - started,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one isolated zkInfer prover task")
    parser.add_argument("--task_file", required=True)
    parser.add_argument("--result_file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run_task(args.task_file, args.result_file))


if __name__ == "__main__":
    main()
