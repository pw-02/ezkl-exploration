import logging
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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


def create_run_dirs(cfg: DictConfig) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}_{cfg.workload.name}"

    runs_dir = Path(cfg.paths.runs_dir)
    run_dir = runs_dir / run_id

    dirs = {
        "run_dir": run_dir,
        "logs_dir": run_dir / "logs",
        "reports_dir": run_dir / "reports",
        "artifacts_dir": run_dir / "artifacts",
        "shared_dir": run_dir / "shared",
        "tmp_dir": run_dir / "tmp",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


@dataclass
class ManagedProcess:
    name: str
    command: List[str]
    logs_dir: Path
    env: dict
    logger: logging.Logger
    proc: Optional[subprocess.Popen] = None
    _tee_thread: Optional[threading.Thread] = None

    @property
    def log_path(self) -> Path:
        return self.logs_dir / f"{self.name}.log"

    def start(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

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

        with open(self.log_path, "w", encoding="utf-8") as log_file:
            for line in self.proc.stdout:
                print(f"[{self.name}] {line}", end="")
                log_file.write(line)
                log_file.flush()

    def returncode(self) -> Optional[int]:
        if self.proc is None:
            return None
        return self.proc.poll()

    def stop(self, timeout_sec: int = 3) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return

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
        failed = []

        for process in self.processes:
            code = process.returncode()
            if code is not None and code != 0:
                failed.append(process)

        return failed

    def raise_if_any_failed(self) -> None:
        failed = self.failed_processes()

        if not failed:
            return

        details = "\n".join(
            f"- {p.name} exited with code {p.returncode()}. Log: {p.log_path}"
            for p in failed
        )

        raise RuntimeError(f"One or more subprocesses failed:\n{details}")


def make_env() -> dict:
    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f".:{existing_pythonpath}" if existing_pythonpath else "."
    env["PYTHONUNBUFFERED"] = "1"

    return env


def build_common_overrides(
    cfg: DictConfig,
    run_dir: Path,
) -> List[str]:
    return [
        f"coordinator.host={cfg.launch.coordinator_host}",
        f"coordinator.port={cfg.launch.coordinator_port}",
        "storage.type=filesystem",
        "storage.root_dir=.",
        f"storage.run_dir={run_dir}",
        f"jobs.scheduler={cfg.jobs.scheduler}",
    ]


def build_coordinator_cmd(
    cfg: DictConfig,
    run_dir: Path,
) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "zkinfer.runtime.coordinator_grpc",
        *build_common_overrides(cfg, run_dir),
    ]


def build_worker_cmd(
    cfg: DictConfig,
    worker_id: str,
    run_dir: Path,
) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "zkinfer.runtime.worker",
        *build_common_overrides(cfg, run_dir),
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
        simplify_model=cfg.execution.get("simplify_model", False),
        simplify_input_shapes=cfg.workload.get("input_shapes", None),
    )


def wait_for_request_report_or_crash(
    reports_dir: Path,
    poll_interval_sec: int,
    process_group: ProcessGroup,
    logger: logging.Logger,
) -> None:
    report_path = reports_dir / "request_report.csv"

    logger.info("Waiting for request report: %s", report_path)

    while not report_path.exists():
        process_group.raise_if_any_failed()
        time.sleep(poll_interval_sec)

    logger.info("Request completed. Report written to %s", report_path)


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logger()
    run_dirs = create_run_dirs(cfg)

    run_dir = run_dirs["run_dir"]
    logs_dir = run_dirs["logs_dir"]
    reports_dir = run_dirs["reports_dir"]

    logger.info("Run directory: %s", run_dir)

    env = make_env()
    process_group = ProcessGroup(logger)

    try:
        process_group.add(
            ManagedProcess(
                name="coordinator",
                command=build_coordinator_cmd(cfg, run_dir),
                logs_dir=logs_dir,
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
                    command=build_worker_cmd(cfg, worker_id, run_dir),
                    logs_dir=logs_dir,
                    env=env,
                    logger=logger,
                )
            )

        time.sleep(cfg.launch.worker_startup_delay_sec)

        request_id = submit_workload(cfg, logger)
        logger.info("Submitted request: %s", request_id)

        if cfg.launch.shutdown_when_done:
            wait_for_request_report_or_crash(
                reports_dir=reports_dir,
                poll_interval_sec=cfg.launch.poll_completion_interval_sec,
                process_group=process_group,
                logger=logger,
            )
        else:
            logger.info("Experiment running. Press Ctrl+C to stop coordinator/workers.")

            while True:
                process_group.raise_if_any_failed()
                time.sleep(cfg.launch.poll_completion_interval_sec)

    except KeyboardInterrupt:
        logger.info("Stopping experiment...")

    finally:
        process_group.stop_all()
        logger.info("Done. Run directory: %s", run_dir)


if __name__ == "__main__":
    main()