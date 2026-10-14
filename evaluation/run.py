#!/usr/bin/env python3
"""Plan or execute one local evaluation cell; never launches a whole sweep."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINES = json.loads((Path(__file__).parent / "configs/baselines.json").read_text())


def override(key, value):
    return f"{key}={json.dumps(value)}"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", choices=BASELINES, required=True)
    p.add_argument("--workload", default="mnist_gan")
    p.add_argument("--requests", type=int, default=1)
    p.add_argument("--arrival", choices=["burst", "poisson"], default="burst")
    p.add_argument("--rate", type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--partition-size", type=int)
    p.add_argument("--simplify-model", action="store_true")
    p.add_argument("--verify", action="store_true", help="include runtime proof verification")
    p.add_argument("--setup", choices=["cold", "warm"], default="cold")
    p.add_argument("--setup-dir", type=Path)
    p.add_argument("--profile-dir", type=Path)
    p.add_argument("--allow-unprofiled", action="store_true", help="smoke tests only; log this choice")
    p.add_argument("--reserve-gib", type=float, default=8)
    p.add_argument("--cache-gib", type=float, default=500, help="use identical capacity in all arms")
    p.add_argument("--history-percentile", type=float, default=0.9)
    p.add_argument("--sampling-seconds", type=float, default=0.25)
    p.add_argument("--submit-workers", type=int, default=16)
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--timeout", type=float, default=86400)
    p.add_argument("--node-startup-seconds", type=float, default=2)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    return p


def build_plan(args):
    baseline = BASELINES[args.baseline]
    output = (args.output or ROOT / "evaluation/results" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + args.baseline
    )).resolve()
    setup = args.setup_dir.resolve() if args.setup_dir else output / "setup"
    shared = {
        "coordinator.host": "localhost", "coordinator.port": args.port,
        "coordinator.scheduling_policy": baseline["policy"],
        "coordinator.runs_dir": str(output / "requests"),
        "coordinator.logs_dir": str(output / "logs/coordinator"),
        "coordinator.state_db_path": str(output / "state/coordinator.sqlite3"),
        "node.runs_dir": str(output / "requests"),
        "node.logs_dir": str(output / "logs/node"),
        "storage.backend": "filesystem",
        "storage.request_data.path": str(output / "request-data"),
        "storage.reusable_setup.path": str(setup),
        "storage.reusable_setup.enabled": args.setup == "warm",
        "verification.enabled": args.verify,
    }
    node = {
        "node.node_id": "evaluation-node", "node.runtime_state_dir": str(output / "node-state"),
        "node.local_cache.path": str(output / "cache"), "node.local_cache.max_gb": args.cache_gib,
        "node.max_concurrent_jobs": baseline["cap"],
        "node.resource_sampling_interval_sec": args.sampling_seconds,
        "node.stage_admission.enabled": baseline["admission"],
        "node.stage_admission.mode": baseline["mode"],
        "node.stage_admission.memory_safety_reserve_gb": args.reserve_gib,
        "node.stage_admission.history_percentile": args.history_percentile,
    }
    common = [override(k, v) for k, v in shared.items()]
    coordinator = [sys.executable, "-u", "-m", "zkinfer.runtime.coordinator_grpc", *common]
    worker = [sys.executable, "-u", "-m", "zkinfer.runtime.node.supervisor",
              *common, *[override(k, v) for k, v in node.items()]]
    traffic = [sys.executable, "-u", str(ROOT / "evaluation/traffic.py"),
               "--target", f"localhost:{args.port}", "--workload", args.workload,
               "--requests", str(args.requests), "--arrival", args.arrival,
               "--seed", str(args.seed), "--partition-size", str(args.partition_size or baseline["partition_size"]),
               "--timeout", str(args.timeout), "--submit-workers", str(args.submit_workers),
               "--output", str(output / "traffic")]
    if args.rate is not None:
        traffic += ["--rate", str(args.rate)]
    if not baseline["partitioning"]:
        traffic += ["--monolithic"]
    if args.simplify_model:
        traffic += ["--simplify-model"]
    return {"output": str(output), "baseline": args.baseline,
            "runtime_overrides": {**shared, **node},
            "commands": {"coordinator": coordinator, "node": worker, "traffic": traffic}}


def validate(args):
    if args.requests < 1 or args.submit_workers < 1 or args.timeout <= 0:
        raise ValueError("requests, submit-workers and timeout must be positive")
    if args.partition_size is not None and args.partition_size < 1:
        raise ValueError("partition-size must be positive")
    if args.reserve_gib < 0 or args.cache_gib <= 0 or args.sampling_seconds <= 0:
        raise ValueError("invalid memory/cache/sampling setting")
    if not 0.5 <= args.history_percentile <= 1 or not 1 <= args.port <= 65535:
        raise ValueError("invalid percentile or port")
    if args.node_startup_seconds < 0:
        raise ValueError("node-startup-seconds must not be negative")
    if args.arrival == "poisson" and (args.rate is None or args.rate <= 0):
        raise ValueError("Poisson arrivals require --rate > 0")
    if args.execute:
        names = args.workload.split(",")
        for name in names:
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("workloads must be config stems")
            if not (ROOT / "experiments/config/workload" / f"{name}.yaml").is_file():
                raise ValueError(f"missing workload config: {name}")
        if args.setup == "warm" and (
            args.setup_dir is None or not args.setup_dir.is_dir() or not any(args.setup_dir.iterdir())
        ):
            raise ValueError("warm runs require --setup-dir with pre-generated matching artifacts")
        if BASELINES[args.baseline]["admission"] and args.profile_dir is None and not args.allow_unprofiled:
            raise ValueError("admission experiments require --profile-dir (or --allow-unprofiled for a smoke test)")
        if BASELINES[args.baseline]["admission"] and args.setup == "cold" and not args.allow_unprofiled:
            raise ValueError("cold setup disables coordinator profile loading in the current runtime; use warm for controlled comparisons, or --allow-unprofiled for an explicitly unprofiled smoke test")
        if args.setup == "warm" and not args.allow_unprofiled and not list(args.setup_dir.rglob("resource_profile.json")):
            raise ValueError("warm comparisons require coordinator resource_profile.json files in the setup snapshot as well as node profiles")
        if args.profile_dir is not None and not list(args.profile_dir.glob("*.json")):
            raise ValueError("profile-dir must contain resource profile JSON files")


def stop(process):
    # Only signal process groups created by this runner with start_new_session.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def execute(args, plan):
    output = Path(plan["output"])
    # Refuse to attach to an unrelated already-running coordinator.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True, exist_ok=False)
    if args.profile_dir:
        destination = output / "node-state/resource_profiles"
        destination.mkdir(parents=True)
        for source in args.profile_dir.glob("*.json"):
            shutil.copy2(source, destination / source.name)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=False)
    plan.update(arguments=vars(args), git_sha=revision.stdout.strip(),
                started_at=datetime.now(timezone.utc).isoformat(),
                thread_environment={key: os.environ.get(key) for key in
                    ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    (output / "plan.json").write_text(json.dumps(plan, default=str, indent=2) + "\n")
    processes, handles = [], []
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    def launch(name):
        handle = (output / f"{name}.log").open("x")
        handles.append(handle)
        proc = subprocess.Popen(plan["commands"][name], cwd=ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(proc)
        return proc
    result = {"status": "failed"}
    try:
        coordinator = launch("coordinator")
        ready_until = time.monotonic() + 60
        while True:
            if coordinator.poll() is not None:
                raise RuntimeError("coordinator exited; see coordinator.log")
            try:
                with socket.create_connection(("127.0.0.1", args.port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() > ready_until:
                    raise TimeoutError("coordinator startup timeout")
                time.sleep(0.2)
        worker = launch("node")
        time.sleep(args.node_startup_seconds)
        if worker.poll() is not None:
            raise RuntimeError("node exited; see node.log")
        traffic = launch("traffic")
        until = time.monotonic() + args.timeout + 30
        while traffic.poll() is None:
            if coordinator.poll() is not None or worker.poll() is not None:
                raise RuntimeError("runtime process exited during trial; inspect logs")
            if time.monotonic() > until:
                raise TimeoutError("load-generator deadline exceeded")
            time.sleep(0.25)
        result = {"status": "completed" if traffic.returncode == 0 else "failed",
                  "traffic_returncode": traffic.returncode}
        return traffic.returncode
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        for process in reversed(processes):
            stop(process)
        for handle in handles:
            handle.close()
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "run_status.json").write_text(json.dumps(result, indent=2) + "\n")


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        for name, command in plan["commands"].items():
            print(f"\n# {name}\n{shlex.join(command)}")
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
