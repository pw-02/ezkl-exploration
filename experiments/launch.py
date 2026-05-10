import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from zkinfer.client.api import ZKInferenceClient


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] launch: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("launch")


def open_log(logs_dir: str, name: str):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    return open(Path(logs_dir) / f"{name}.log", "w", encoding="utf-8")


def start_process(
    name: str,
    command: List[str],
    logs_dir: str,
    env: dict,
    logger: logging.Logger,
) -> subprocess.Popen:
    log_file = open_log(logs_dir, name)
    logger.info("Starting %s: %s", name, " ".join(command))

    return subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )


def stop_processes(processes: List[subprocess.Popen], logger: logging.Logger) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()

    time.sleep(2)

    for proc in processes:
        if proc.poll() is None:
            logger.warning("Killing process %s", proc.pid)
            proc.kill()


def wait_for_port(host: str, port: int, timeout_sec: int, logger: logging.Logger) -> None:
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                logger.info("Coordinator is ready at %s:%s", host, port)
                return
        except OSError:
            time.sleep(0.5)

    raise TimeoutError(f"Coordinator did not become ready at {host}:{port}")


def submit_workload(cfg: DictConfig, logger: logging.Logger) -> str:
    target = f"{cfg.launch.coordinator_host}:{cfg.launch.coordinator_port}"

    logger.info("Submitting workload to %s", target)
    logger.info("Workload:\n%s", OmegaConf.to_yaml(cfg.workload))

    client = ZKInferenceClient(target=target)

    return client.submit_inference_request(
        name=cfg.workload.name,
        onnx_model_path=cfg.workload.onnx_file,
        input_data_path=cfg.workload.input_file,
        split_mode=cfg.execution.split_mode,
        ops_per_chunk=cfg.execution.ops_per_chunk,
        scheduler=cfg.jobs.scheduler,
    )


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logger()

    logs_dir = cfg.paths.logs_dir
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"

    processes: List[subprocess.Popen] = []

    try:
        coordinator_cmd = [
            sys.executable,
            "-m",
            "zkinfer.runtime.coordinator_grpc",
            f"coordinator.host={cfg.launch.coordinator_host}",
            f"coordinator.port={cfg.launch.coordinator_port}",
            f"paths.logs_dir={cfg.paths.logs_dir}",
            f"paths.reports_dir={cfg.paths.reports_dir}",
        ]

        processes.append(
            start_process(
                name="coordinator",
                command=coordinator_cmd,
                logs_dir=logs_dir,
                env=env,
                logger=logger,
            )
        )

        wait_for_port(
            host=cfg.launch.coordinator_host,
            port=cfg.launch.coordinator_port,
            timeout_sec=cfg.launch.coordinator_ready_timeout_sec,
            logger=logger,
        )

        for idx in range(cfg.launch.num_workers):
            worker_id = f"worker_{idx + 1}"

            worker_cmd = [
                sys.executable,
                "-m",
                "zkinfer.runtime.worker",
                f"coordinator.host={cfg.launch.coordinator_host}",
                f"coordinator.port={cfg.launch.coordinator_port}",
                f"worker.worker_id={worker_id}",
            ]

            processes.append(
                start_process(
                    name=worker_id,
                    command=worker_cmd,
                    logs_dir=logs_dir,
                    env=env,
                    logger=logger,
                )
            )

        time.sleep(cfg.launch.worker_startup_delay_sec)

        request_id = submit_workload(cfg, logger)
        logger.info("Submitted request: %s", request_id)
        logger.info("Experiment running. Press Ctrl+C to stop coordinator/workers.")

        while True:
            time.sleep(5)

            exited = [proc for proc in processes if proc.poll() is not None]
            if exited:
                logger.warning(
                    "%d managed process(es) exited. Check logs in %s",
                    len(exited),
                    logs_dir,
                )

    except KeyboardInterrupt:
        logger.info("Stopping experiment...")

    finally:
        stop_processes(processes, logger)
        logger.info("Done.")


if __name__ == "__main__":
    main()