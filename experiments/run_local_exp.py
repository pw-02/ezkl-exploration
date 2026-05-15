# experiments/run_local_suite.py
# python experiments/run_local_exp.py suite.workloads='[mnist_classifier,mobilenet_v2]'

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from zkinfer.client.api import ZKInferenceClient


WORKLOADS = [
    "mnist_classifier",
    "mnist_gan",
    "mobilenet_v2",
    "nano_gpt_4_layers_64_embd"
]


class ActiveRunLog:
    def __init__(self) -> None:
        self.file = None
        self.lock = threading.Lock()

    def set_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with self.lock:
            self.close()
            self.file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text: str) -> None:
        with self.lock:
            if self.file is not None:
                self.file.write(text)

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


class ActiveRunLogHandler(logging.Handler):
    def __init__(self, active_log: ActiveRunLog) -> None:
        super().__init__()
        self.active_log = active_log

    def emit(self, record: logging.LogRecord) -> None:
        self.active_log.write(self.format(record) + "\n")


def setup_logger(active_log: ActiveRunLog) -> logging.Logger:
    logger = logging.getLogger("exp")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = ActiveRunLogHandler(active_log)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


@dataclass
class ManagedProcess:
    name: str
    command: List[str]
    env: dict
    logger: logging.Logger
    active_log: ActiveRunLog
    proc: Optional[subprocess.Popen] = None
    _tee_thread: Optional[threading.Thread] = None
    write_subprocess_output_to_console: bool = True

    def start(self) -> None:
        self.logger.info("Starting %s: %s", self.name, " ".join(self.command))

        self.proc = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self.env,
            text=True,
            bufsize=1,
        )

        self._tee_thread = threading.Thread(
            target=self._tee_output,
            name=f"{self.name}-tee",
            daemon=True,
        )
        self._tee_thread.start()

    def _tee_output(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None

        for line in self.proc.stdout:
            # prefixed = f"[{self.name}] {line}"
            prefixed = f"[{line}"
            if self.write_subprocess_output_to_console:
                print(prefixed, end="")
            self.active_log.write(prefixed)

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

        if self._tee_thread is not None:
            self._tee_thread.join(timeout=timeout_sec)


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

        logger.debug(
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


def request_logs_dir(cfg: DictConfig, request_id: str) -> Path:
    return Path(cfg.paths.runs_dir) / request_id / "logs"


def save_request_metadata(
    cfg: DictConfig,
    workload_cfg: DictConfig,
    request_id: str,
) -> None:
    logs_dir = request_logs_dir(cfg, request_id)
    logs_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config=cfg, f=logs_dir / "exp_config.yaml")
    OmegaConf.save(config=workload_cfg, f=logs_dir / "workload_config.yaml")


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    active_log = ActiveRunLog()
    logger = setup_logger(active_log)

    env = make_env()
    process_group = ProcessGroup(logger)

    try:
        process_group.add(
            ManagedProcess(
                name="coordinator",
                command=build_coordinator_cmd(cfg),
                env=env,
                logger=logger,
                active_log=active_log,
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
                    active_log=active_log,
                )
            )

        time.sleep(cfg.launch.worker_startup_delay_sec)

        client = ZKInferenceClient(
            target=f"{cfg.launch.coordinator_host}:{cfg.launch.coordinator_port}"
        )
        failed_workloads = []

        for workload_name in cfg.suite.workloads:
            try:
                workload_cfg = load_workload_cfg(workload_name)

                logger.info("=" * 80)
                logger.info("Running workload: %s", workload_cfg.name)
                logger.info("Workload config:\n%s", OmegaConf.to_yaml(workload_cfg))

                request_id = submit_workload(client, cfg, workload_cfg)

                logs_dir = request_logs_dir(cfg, request_id)
                exp_log_path = logs_dir / "run.log"

                active_log.set_path(exp_log_path)
                save_request_metadata(cfg, workload_cfg, request_id)

                logger.info("Submitted request: %s", request_id)
                logger.info("Writing workload log to %s", exp_log_path)

                wait_for_request_or_crash(
                    client=client,
                    request_id=request_id,
                    poll_interval_sec=cfg.launch.poll_completion_interval_sec,
                    process_group=process_group,
                    logger=logger,
                )

                logger.info("Finished workload: %s", workload_cfg.name)

            except Exception as exc:
                logger.exception("Workload failed: %s", workload_name)
                failed_workloads.append((workload_name, str(exc)))

            finally:
                active_log.close()

        if failed_workloads:
            logger.warning("Suite completed with %d failed workloads:", len(failed_workloads))
            for name, error in failed_workloads:
                logger.warning("- %s: %s", name, error)
        else:
            logger.info("All workloads completed.")

    finally:
        active_log.close()
        process_group.stop_all()
        logger.info("Suite finished.")


if __name__ == "__main__":
    main()