# experiments/run_local_suite.py

import time
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from zkinfer.client.api import ZKInferenceClient
import logging
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

WORKLOADS = [
    "mnist_classifier",
]

# experiments/local_launch.py

def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] launch: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("launch")


@dataclass
class ManagedProcess:
    name: str
    command: List[str]
    env: dict
    logger: logging.Logger
    proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.logger.info("Starting %s: %s", self.name, " ".join(self.command))

        self.proc = subprocess.Popen(
            self.command,
            env=self.env,
        )

    def returncode(self) -> Optional[int]:
        if self.proc is None:
            return None

        return self.proc.poll()

    def stop(self, timeout_sec: int = 5) -> None:
        if self.proc is None:
            return

        if self.proc.poll() is None:
            self.logger.info("Stopping %s", self.name)
            self.proc.terminate()

            try:
                self.proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                self.logger.warning("Killing %s pid=%s", self.name, self.proc.pid)
                self.proc.kill()
                self.proc.wait(timeout=timeout_sec)


class ProcessGroup:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.processes: List[ManagedProcess] = []

    def add(self, process: ManagedProcess) -> ManagedProcess:
        self.processes.append(process)
        process.start()
        return process

    def stop_all(self) -> None:
        for process in reversed(self.processes):
            process.stop()

    def failed_processes(self) -> List[ManagedProcess]:
        return [
            process
            for process in self.processes
            if process.returncode() is not None and process.returncode() != 0
        ]

    def raise_if_any_failed(self) -> None:
        failed = self.failed_processes()

        if not failed:
            return

        details = "\n".join(
            f"- {process.name} exited with code {process.returncode()}"
            for process in failed
        )

        raise RuntimeError(f"One or more subprocesses failed:\n{details}")


def make_env() -> dict:
    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f".:{existing_pythonpath}" if existing_pythonpath else "."
    env["PYTHONUNBUFFERED"] = "1"

    return env


def build_common_overrides(cfg: DictConfig) -> List[str]:
    return [
        f"coordinator.host={cfg.launch.coordinator_host}",
        f"coordinator.port={cfg.launch.coordinator_port}",
        f"jobs.scheduler={cfg.jobs.scheduler}",
    ]


def build_coordinator_cmd(cfg: DictConfig) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "zkinfer.runtime.coordinator_grpc",
        *build_common_overrides(cfg),
    ]


def build_worker_cmd(cfg: DictConfig, worker_id: str) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "zkinfer.runtime.worker",
        *build_common_overrides(cfg),
        f"worker.worker_id={worker_id}",
    ]


def wait_for_port_or_crash(
    host: str,
    port: int,
    timeout_sec: int,
    process_group: ProcessGroup,
    logger: logging.Logger,
) -> None:
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        process_group.raise_if_any_failed()

        try:
            with socket.create_connection((host, port), timeout=1):
                logger.info("Coordinator is ready at %s:%s", host, port)
                return
        except OSError:
            time.sleep(0.5)

    process_group.raise_if_any_failed()
    raise TimeoutError(f"Coordinator did not become ready at {host}:{port}")


def wait_for_request_or_crash(
    client: ZKInferenceClient,
    request_id: str,
    poll_interval_sec: int,
    process_group: ProcessGroup,
    logger: logging.Logger,
) -> None:
    logger.info("Waiting for request %s to finish", request_id)

    while True:
        process_group.raise_if_any_failed()

        status = client.get_request_status(request_id)

        logger.info(
            "Request %s | status=%s | completed=%s/%s | failed=%s",
            request_id,
            status.status,
            status.completed_jobs,
            status.total_jobs,
            status.failed_jobs,
        )

        if status.status == "completed":
            logger.info("Request %s completed", request_id)
            return

        if status.status == "failed":
            raise RuntimeError(f"Request {request_id} failed: {status.message}")

        if status.status == "unknown":
            raise RuntimeError(f"Request {request_id} unknown: {status.message}")

        time.sleep(poll_interval_sec)



def load_workload_cfg(workload_name: str) -> DictConfig:
    path = Path(__file__).parent / "config" / "workload" / f"{workload_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Workload config not found: {path}")

    return OmegaConf.load(path)


def submit_workload(
    client: ZKInferenceClient,
    cfg: DictConfig,
    workload_cfg: DictConfig,
) -> str:
    return client.submit_inference_request(
        name=workload_cfg.name,
        onnx_model_path=workload_cfg.onnx_file,
        input_data_path=workload_cfg.input_file,
        split_mode=cfg.execution.split_mode,
        ops_per_chunk=cfg.execution.ops_per_chunk,
        scheduler=cfg.jobs.scheduler,
        simplify_model=cfg.execution.get("simplify_model", False),
        input_shapes=workload_cfg.get("input_shapes", None),
    )


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logger()

    env = make_env()
    process_group = ProcessGroup(logger)

    try:
        process_group.add(
            ManagedProcess(
                name="coordinator",
                command=build_coordinator_cmd(cfg),
                env=env,
                logger=logger,
            )
        )

        wait_for_port_or_crash(
            host=cfg.launch.coordinator_host,
            port=cfg.launch.coordinator_port,
            timeout_sec=cfg.launch.coordinator_ready_timeout_sec,
            process_group=process_group,
            logger=logger,
        )

        for idx in range(cfg.launch.num_workers):
            worker_id = f"worker_{idx + 1}"

            process_group.add(
                ManagedProcess(
                    name=worker_id,
                    command=build_worker_cmd(cfg, worker_id),
                    env=env,
                    logger=logger,
                )
            )

        time.sleep(cfg.launch.worker_startup_delay_sec)

        client = ZKInferenceClient(
            target=f"{cfg.launch.coordinator_host}:{cfg.launch.coordinator_port}"
        )

        for workload_name in WORKLOADS:
            workload_cfg = load_workload_cfg(workload_name)

            logger.info("=" * 80)
            logger.info("Running workload: %s", workload_cfg.name)
            logger.info("Workload config:\n%s", OmegaConf.to_yaml(workload_cfg))

            request_id = submit_workload(client, cfg, workload_cfg)
            logger.info("Submitted request: %s", request_id)

            wait_for_request_or_crash(
                client=client,
                request_id=request_id,
                poll_interval_sec=cfg.launch.poll_completion_interval_sec,
                process_group=process_group,
                logger=logger,
            )

            logger.info("Finished workload: %s", workload_cfg.name)

        logger.info("All workloads completed.")

    finally:
        process_group.stop_all()
        logger.info("Suite finished.")


if __name__ == "__main__":
    main()