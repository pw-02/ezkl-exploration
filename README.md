# zkInfer

zkInfer is a distributed runtime for partitioned zkML proving with an EZKL/Halo2 backend. The `sacdad27` branch focuses on systems behavior: partitioning, node-level resource control, reusable proving artifacts, distributed scheduling, verification, failure recovery, and experiment accounting.

Backward compatibility with the older worker-centric runtime is not a goal on this branch.

## Architecture

```text
Client
  |
  v
Coordinator
  |- prepares an inference request as many proof jobs
  |- keeps the global ready queue
  |- schedules one feasible job when a node polls
  |- issues leases and persists state in SQLite
  |- requeues jobs from failed nodes
  |- writes job/request/data-movement reports
  `- verifies completed requests asynchronously
          |
          | gRPC control plane only
          v
NodeSupervisor                         one per machine
  |- reports fresh CPU/memory/cache state
  |- owns the bounded node cache
  |- restores reusable setup files from shared storage
  |- applies local admission controls
  |- launches one isolated OS process per proof job
  `- publishes results + metrics
          |
          v
Prover subprocess
  |- receives local filesystem paths
  |- runs EZKL/Halo2
  |- waits at the node heavy-phase gate when necessary
  |- records its exact stage in resource samples
  `- exits after one proof job

Shared storage
  |- content-addressed model/input artifacts
  |- reusable setup cache
  `- proof/result manifests
```

Large model, witness, key, and proof objects do not move through gRPC.

## Node execution and admission

The coordinator schedules **nodes**, not fixed prover slots. Local execution is controlled with:

```yaml
node:
  local_concurrency_limit: 1
  max_heavy_phases: 1
  memory_reserve_gb: 8
  resource_monitor_interval_sec: 1
```

`local_concurrency_limit` is a safety ceiling on active proof processes. `max_heavy_phases` controls how many of those processes may simultaneously enter memory-heavy key-generation/proving regions. `memory_reserve_gb` prevents a node from accepting more work when free RAM is already low.

Each proof job runs through `zkinfer/runtime/node/prover_process.py`. The supervisor uses threads only to manage child-process lifecycles; EZKL/Halo2 does not run inside supervisor threads. This isolates `EZKL_LOG_DIR`, native prover state, failures, and process-level resource measurements.

All prover subprocesses on a node share one interprocess phase gate. A prover waits before `KEY_GEN` or `PROVING` when all heavy-phase permits are occupied. The fused cold EZKL path holds one permit across its combined keygen/prove operation because Rust moves directly from one phase into the other without returning to Python.

`phase_admission_wait_time(s)` records the time spent waiting for a heavy-phase permit.

## Exact stage/resource traces

Each prover writes its current stage to `status.txt`, and every `resource_usage.csv` sample carries the same stage label. Typical stages are:

```text
CALIBRATING
COMPILING
GETTING_SRS
GENERATING_WITNESS
KEY_GEN
PROVING
PERSISTING_KEYS
PERSISTING_PROOF
PUBLISHING_KEYS
DONE
```

The resource logger tracks the prover process tree while excluding the logger process itself. `resource_parser.py` derives whole-job and per-stage metrics such as:

```text
stage_key_gen_max_memory(GB)
stage_key_gen_avg_cpu_machine(%)
stage_proving_max_memory(GB)
stage_proving_avg_cpu_machine(%)
```

These measurements are intended to determine whether phase-aware admission actually improves concurrency rather than assuming that it does.

## Node cache

The node cache lives in `zkinfer/runtime/node/cache.py` and contains:

```text
objects/       content-addressed model/input artifacts
setup/         reusable settings, compiled circuits, PKs and VKs
```

It provides per-object locking, per-setup-key locking, active-setup pinning, bounded LRU-style eviction, PK compression when publishing setup state, and PK decompression when restoring it.

Terminology in the runtime is deliberately literal:

```text
materialize artifact   make an immutable object available locally
restore setup cache    copy reusable setup files into the node cache
publish setup cache    copy newly generated setup files to shared storage
```

## EZKL cold path

The custom `pw-02/ezkl:dzkml-v10.29` fork includes a fused cold setup-and-prove operation:

```text
load circuit + witness
load SRS once
KEY_GEN
  generate PK/VK in memory
PROVING
  prove with the same in-memory PK
PERSISTING_KEYS
  save PK/VK for future warm requests
PERSISTING_PROOF
  save proof
```

This avoids generating a large PK, serializing it, returning to Python, and immediately reading/deserializing the same PK for the first proof. Warm requests still use the normal cached-PK prove path. There is deliberately no cross-request in-memory PK cache.

See `docs/EZKL_BACKEND.md` for details.

## Scheduling

Coordinator policies are:

```text
fifo
lpt
resource_aware
cache_aware
```

All policies first require the job's predicted peak memory to fit the node's current available memory. Scheduling is request-aware: the coordinator selects a least-served request before applying the chosen job-ranking policy within that request. This prevents one large partitioned request from monopolizing the ready queue.

`cache_aware` prefers jobs whose setup key is already present on the node. The old protobuf `max_concurrent_jobs` field is retained only for wire compatibility and is ignored by coordinator admission; local concurrency belongs to the node supervisor.

## Failure recovery and persistent state

Coordinator state can be persisted in SQLite:

```yaml
coordinator:
  state_store_enabled: true
  state_db_path: experiments/runs/coordinator_state.sqlite3
```

The state store migrates older databases that lack `assigned_node_id`, `lease_id`, or lease timestamps, including copying legacy `assigned_worker_id` values into `assigned_node_id`.

A dead node causes every `IN_PROGRESS` job assigned to that node to be requeued. Recovery does not depend only on the node's last heartbeat list. Heartbeat updates that omit an active-job list preserve the previously known list.

## Verification

Enable runtime verification with:

```yaml
verification:
  enabled: true
  srs_dir: srs
  require_split_chain: false
  write_request_manifest: true
```

When the final proof job reports success, its gRPC call is acknowledged after job bookkeeping/reporting. Request verification and final request-report generation run in a coordinator finalizer thread so a node is not held busy while the coordinator verifies a potentially large request.

The request enters `VERIFYING`, and its completion timestamp is recorded only after verification. End-to-end request time therefore includes verification when enabled.

Verification checks every individual EZKL proof and then checks partition boundaries. For each downstream partition, zkInfer copies the proof to scratch space, applies `ezkl.swap_proof_commitments(...)` with the previous partition's witness, and verifies the rewritten proof with the downstream partition's VK/settings/SRS. Original proofs are never modified. Requests containing failed jobs skip verification.

## Reports and data movement

A normal request can produce:

```text
job_report.csv
perf_metrics.csv
request_report.csv
transfer_overhead_report.csv
transfer_overhead_report.json
request_artifact_manifest.json
verification_report.json        when verification is enabled
```

Backend instrumentation may additionally produce:

```text
resource_usage.csv
halo2_circuit.csv
halo2_prover_cpu.csv
halo2_ffts.csv
halo2_msms.csv
```

Report CSV writing is schema-safe: if later jobs expose additional stage columns, the report is rewritten with the union schema rather than appending malformed rows.

Data-movement accounting includes model/input materialization, setup restoration, PK compression/decompression, setup publication, output publication, shared-store bytes, and node-cache bytes. `agg_data_movement_service_time(s)` is an aggregate service-time metric, not a request critical-path metric.

## Local experiments

The local launcher now follows the real architecture: **one NodeSupervisor represents one machine**. Increase local concurrency instead of launching several fake workers on the same host.

For example:

```bash
python experiments/run_local_exp.py \
  "suite.workloads=[mnist_gan]" \
  node.local_concurrency_limit=2 \
  node.max_heavy_phases=1
```

The tmux launchers use the same node terminology. `analysis/analyze_phase_admission.py` converts request/report directories into request-level and stage-level CSVs for comparing concurrency/admission settings.

## Main runtime files

```text
zkinfer/
  backends/ezkl/ezkl_prover.py

  profiling/
    overhead.py
    reports.py
    resource_logger.py
    resource_parser.py

  runtime/
    coordinator.py
    coordinator_grpc.py
    scheduler.py
    state_store.py
    request_builder.py

    node/
      cache.py
      phase_admission.py
      prover_task.py
      prover_process.py
      supervisor.py

  verification/
    verifier.py
    request_manifest.py
    split_chain.py
```

The old `runtime/worker_cache.py` implementation has been removed. `runtime/worker.py` remains only as a compatibility entrypoint that forwards to the node supervisor.

## Validation sequence

The code changes are committed, but the custom EZKL/Rust build and full runtime tests must be run in an environment containing the sibling Halo2 checkouts used by your EZKL fork.

Recommended validation order:

```bash
# 1. Rebuild the custom EZKL Python extension.
cd <ezkl-checkout>
maturin develop --release --features python-bindings

# 2. Static/import-oriented Python compilation.
cd <zkinfer-checkout>
python -m compileall zkinfer experiments analysis tests

# 3. Core scheduler/cache/state/admission regressions.
python -m unittest tests.test_runtime_core -v
```

Then run one tiny cold proof, repeat it warm, inspect `resource_usage.csv`, run two concurrent local jobs with `local_concurrency_limit=2,max_heavy_phases=1`, kill a prover subprocess and then a whole node to exercise recovery, restart the coordinator against an existing SQLite DB, and finally enable verification to exercise split-boundary checks.

## Research/evaluation direction

The runtime is now set up to evaluate:

- monolithic vs partitioned proving
- cold vs warm setup state
- fused cold setup/prove vs the previous file round trip
- memory feasibility and peak-memory reduction
- request-aware distributed scheduling
- setup-cache locality
- phase-aware local admission
- scale-out efficiency
- data-movement overhead
- verification overhead
- concurrent requests

The next research decision should come from measurements: characterize key generation and proving CPU/memory by stage, then compare unrestricted local concurrency against a lower heavy-phase limit. If phase-aware admission reduces OOMs or improves throughput/latency under realistic concurrency, it becomes a defensible zkInfer mechanism; if not, the mechanism stays simple or is removed.
