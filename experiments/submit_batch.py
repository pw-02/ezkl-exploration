#!/usr/bin/env python3
"""Submit many zkInfer requests and measure batch makespan/throughput.

This is intentionally request-level batching: every input becomes an ordinary
zkInfer request, so the normal coordinator fairness, cache-aware placement, and
node phase-admission policies remain in effect.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

from zkinfer.client.api import ZKInferenceClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="localhost:50051")
    parser.add_argument("--name", required=True, help="Logical model/workload name")
    parser.add_argument("--model", required=True, help="ONNX model path visible to coordinator")
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Input JSON path. Repeat to submit multiple distinct inputs.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the supplied input list this many times.",
    )
    parser.add_argument("--split-mode", default="fixed")
    parser.add_argument("--ops-per-chunk", type=int, default=1)
    parser.add_argument("--scheduler", default="cache_aware")
    parser.add_argument("--simplify-model", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/batch_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    inputs: List[str] = list(args.inputs) * args.repeat
    client = ZKInferenceClient(target=args.target)

    batch_started = time.perf_counter()
    submitted: List[Dict[str, object]] = []
    for index, input_path in enumerate(inputs, start=1):
        request_started = time.perf_counter()
        request_id = client.submit_inference_request(
            name=f"{args.name}-batch-{index}",
            onnx_model_path=args.model,
            input_data_path=input_path,
            split_mode=args.split_mode,
            ops_per_chunk=args.ops_per_chunk,
            scheduler=args.scheduler,
            simplify_model=args.simplify_model,
        )
        submitted.append(
            {
                "request_id": request_id,
                "input_path": input_path,
                "submit_offset_s": time.perf_counter() - batch_started,
                "submit_rpc_time_s": time.perf_counter() - request_started,
                "status": "submitted",
            }
        )

    remaining = {str(item["request_id"]): item for item in submitted}
    while remaining:
        for request_id in list(remaining):
            status = client.get_request_status(request_id)
            if status.status not in {"completed", "failed", "unknown"}:
                continue
            item = remaining.pop(request_id)
            item["status"] = status.status
            item["message"] = status.message
            item["completion_offset_s"] = time.perf_counter() - batch_started
        if remaining:
            time.sleep(max(0.1, args.poll_interval))

    makespan_s = time.perf_counter() - batch_started
    completed = sum(1 for item in submitted if item["status"] == "completed")
    failed = len(submitted) - completed
    throughput = completed / makespan_s if makespan_s > 0 else 0.0

    summary = {
        "name": args.name,
        "num_requests": len(submitted),
        "completed_requests": completed,
        "failed_requests": failed,
        "makespan_s": makespan_s,
        "completed_requests_per_s": throughput,
        "split_mode": args.split_mode,
        "ops_per_chunk": args.ops_per_chunk,
        "scheduler": args.scheduler,
        "requests": submitted,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "batch_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "batch_requests.csv"
    fields = sorted({key for item in submitted for key in item})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(submitted)

    print(json.dumps({key: value for key, value in summary.items() if key != "requests"}, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
