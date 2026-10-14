#!/usr/bin/env python3
"""Reproducible request traffic for an existing zkInfer coordinator.

All latency values are client-observed seconds; completion polling adds up to
approximately poll_seconds of observation delay. RPCs have explicit deadlines.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]


def make_schedule(workloads, count, arrival, rate, seed):
    if count < 1 or not workloads:
        raise ValueError("positive request count and at least one workload required")
    if arrival not in {"burst", "poisson"}:
        raise ValueError("arrival must be burst or poisson")
    if arrival == "poisson" and (rate is None or rate <= 0):
        raise ValueError("Poisson arrivals require a positive requests/second rate")
    timing = random.Random(seed)
    models = random.Random(seed + 1000003)
    elapsed = 0.0
    result = []
    for index in range(count):
        if index and arrival == "poisson":
            elapsed += timing.expovariate(rate)
        result.append({"index": index, "workload": models.choice(workloads),
                       "scheduled_s": elapsed})
    return result


def summarize(records, elapsed):
    completed = [r for r in records if r["status"] == "completed"]
    latencies = [r["latency_s"] for r in completed]
    statuses = {}
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    return {
        "request_count": len(records), "statuses": statuses,
        "trial_elapsed_s": elapsed,
        "drain_inclusive_successes_per_s": len(completed) / elapsed if elapsed > 0 else None,
        "successful_request_latency_median_s": statistics.median(latencies) if latencies else None,
        "successful_request_latency_max_s": max(latencies) if latencies else None,
        "batch_makespan_s": (
            max(r["end_s"] for r in records) - min(r["start_s"] for r in records)
            if records and len(completed) == len(records) else None
        ),
        "note": "Latencies are successful requests only; statuses retain failures. Rate includes drain, not steady-state throughput.",
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="localhost:50051")
    p.add_argument("--workload", default="mnist_gan", help="comma-separated config stems")
    p.add_argument("--requests", type=int, default=1)
    p.add_argument("--arrival", choices=["burst", "poisson"], default="burst")
    p.add_argument("--rate", type=float, help="Poisson requests/second")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--partition-size", type=int, default=1)
    p.add_argument("--monolithic", action="store_true")
    p.add_argument("--simplify-model", action="store_true")
    p.add_argument("--submit-workers", type=int, default=16)
    p.add_argument("--poll-seconds", type=float, default=1)
    p.add_argument("--rpc-timeout", type=float, default=300)
    p.add_argument("--timeout", type=float, default=86400, help="whole traffic trial deadline seconds")
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.partition_size, args.submit_workers, args.poll_seconds,
           args.rpc_timeout, args.timeout) <= 0:
        raise SystemExit("sizes and timeouts must be positive")
    names = [name.strip() for name in args.workload.split(",") if name.strip()]
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise SystemExit("workloads must be configuration stems, not paths")
    schedule = make_schedule(names, args.requests, args.arrival, args.rate, args.seed)
    if args.dry_run:
        print(json.dumps(schedule, indent=2))
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --dry-run")
    # Backend-specific imports are deferred so schedule generation needs stdlib only.
    import sys
    sys.path.insert(0, str(ROOT))
    import grpc
    from omegaconf import OmegaConf
    from zkinfer.proto import zkservice_pb2 as pb
    from zkinfer.proto import zkservice_pb2_grpc as rpc

    workloads = {name: OmegaConf.load(ROOT / "experiments/config/workload" / f"{name}.yaml")
                 for name in names}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    (args.output / "traffic_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n")
    channel = grpc.insecure_channel(args.target, options=[
        ("grpc.max_send_message_length", 64 * 1024**2),
        ("grpc.max_receive_message_length", 64 * 1024**2),
    ])
    stub = rpc.ZKJobServiceStub(channel)
    started = time.monotonic()
    deadline = started + args.timeout
    def now():
        return time.monotonic() - started
    def submit(item):
        record = {**item, "start_s": now()}
        record["submission_lag_s"] = record["start_s"] - item["scheduled_s"]
        cfg = workloads[item["workload"]]
        try:
            shapes = cfg.get("input_shapes")
            if shapes is not None:
                shapes = OmegaConf.to_container(shapes, resolve=True)
            reply = stub.SubmitInferenceRequest(pb.InferenceRequest(
                name=cfg.name, onnx_model_path=cfg.onnx_file, input_data_path=cfg.input_file,
                split_mode="none" if args.monolithic else "fixed",
                ops_per_chunk=args.partition_size, simplify_model=args.simplify_model,
                input_shapes=json.dumps(shapes) if shapes is not None else "",
            ), timeout=min(args.rpc_timeout, max(0.01, deadline - time.monotonic())))
            if not reply.request_id:
                raise RuntimeError("empty request_id")
            record.update(request_id=reply.request_id, accepted_s=now(), status="submitted")
        except Exception as exc:
            record.update(status="submission_error", error=str(exc), end_s=now())
        return record

    records, pending, active = [], {}, {}
    next_index = 0
    next_poll = 0.0
    executor = ThreadPoolExecutor(max_workers=args.submit_workers)
    with (args.output / "requests.jsonl").open("x") as results, \
            (args.output / "events.jsonl").open("x") as events:
        def event(value):
            events.write(json.dumps({"observed_s": now(), **value}) + "\n")
            events.flush()
        def finish(record):
            record.setdefault("end_s", now())
            record["latency_s"] = (record["end_s"] - record["start_s"]
                                   if record.get("start_s") is not None else None)
            record["arrival_to_end_s"] = (
                record["end_s"] - record["scheduled_s"]
                if record["end_s"] >= record["scheduled_s"] else None
            )
            records.append(record)
            results.write(json.dumps(record) + "\n")
            results.flush()
        try:
            while len(records) < len(schedule) and time.monotonic() < deadline:
                elapsed = now()
                while next_index < len(schedule) and schedule[next_index]["scheduled_s"] <= elapsed:
                    item = schedule[next_index]
                    pending[executor.submit(submit, item)] = item
                    next_index += 1
                for future in list(pending):
                    if future.done():
                        pending.pop(future)
                        record = future.result()
                        if record["status"] == "submitted":
                            active[record["request_id"]] = record
                            event({"event": "accepted", **record})
                        else:
                            finish(record)
                if now() >= next_poll:
                    for request_id, record in list(active.items()):
                        if time.monotonic() >= deadline:
                            break
                        try:
                            state = stub.GetRequestStatus(pb.RequestStatusRequest(request_id=request_id),
                                timeout=min(args.rpc_timeout, max(0.01, deadline - time.monotonic())))
                            event({"event": "status", "request_id": request_id,
                                   "status": state.status, "completed_jobs": state.completed_jobs,
                                   "total_jobs": state.total_jobs})
                            if state.status in {"completed", "failed", "unknown"}:
                                record.update(status=state.status, message=state.message)
                                finish(record)
                                active.pop(request_id)
                        except grpc.RpcError as exc:
                            event({"event": "poll_error", "request_id": request_id, "error": str(exc)})
                    event({"event": "backlog", "accepted_active": len(active),
                           "submission_pending": len(pending), "terminal": len(records)})
                    next_poll = now() + args.poll_seconds
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            for record in active.values():
                record["status"] = "client_timeout"
                finish(record)
            for future, item in pending.items():
                if future.done():
                    record = future.result()
                    if record["status"] == "submitted":
                        record["status"] = "client_timeout"
                else:
                    future.cancel()
                    record = {**item, "start_s": None, "status": "submission_unresolved",
                              "note": "RPC may have reached server; inspect coordinator logs"}
                finish(record)
            for item in schedule[next_index:]:
                finish({**item, "start_s": None, "status": "not_submitted_before_deadline"})
        finally:
            channel.close()
            executor.shutdown(wait=True, cancel_futures=True)
    summary = summarize(records, now())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(r["status"] == "completed" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
