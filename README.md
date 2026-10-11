# zkInfer

zkInfer is a distributed runtime for partitioned zkML proving with an EZKL/Halo2 backend. The `sacdad27` branch focuses on systems behavior: partitioning, node-level resource control, reusable proving artifacts, distributed scheduling, verification, failure recovery, and experiment accounting.

Backward compatibility with the older worker-centric runtime is not a goal on this branch.

## Architecture

```text
Client
  |
  v
Coordinator
  |- prepares one request as many proof jobs
  |- keeps the global ready queue
  |- schedules jobs when nodes poll
  |- issues leases and persists state in SQLite
  |- requeues jobs from failed nodes
  |- writes job/request/transfer reports
  `- optionally verifies all proofs and split boundaries
          |
          | gRPC control plane only
          v
NodeSupervisor                 one per machine
  |- reports fresh CPU/memory/cache state
  |- owns the bounded node cache
  |- restores reusable setup files from shared storage
  |- applies local phase-aware admission
  |- launches one isolated OS process per proof job
  `- publishes results and metrics
          |
          v
Prover subprocess
  |- receives local filesystem paths
  |- runs EZKL/Halo2
  |- records exact proof stage in resource samples
  `- exits after one proof job

Shared storage
  |- content-addressed model/input artifacts
  |- reusable setup cache
  `- proof/result manifests
```

Large model, witness, key, and proof objects do not move through gRPC.

## Node execution model

The coordinator schedules **nodes**, not fixed prover slots. A node decides locally whether it can take more work.

The important local controls are:

```yaml
node:
  local_concurrency_limit: 1
  max_heavy_phases: 1
  memory_reserve_gb: 8
```

`local_concurrency_limit` is a safety ceiling on proof processes. `max_heavy_phases` limits how many subprocesses may simultaneously enter memory-heavy key-generation/proving regions. `memory_reserve_gb` keeps a node from accepting new work when free RAM is too low.

Each proof job runs in its own OS process through:

```text
zkinfer/runtime/node/prover_process.py
```

The supervisor itself never calls EZKL/Halo2 inside its controller threads. This keeps `EZKL_LOG_DIR`, process memory accounting, and native prover state isolated per job.

## Phase-aware local admission

Each prover writes its current stage to `status.txt`. Resource samples carry the same stage label.

Important stages include:

```text
CALIBRATING
COMPILING
GETTING_SRS
GENERATING_WITNESS
KEY_GEN
PROVING
PERSISTING_KEYS
PERSISTING_PROOF
DONE
```

Before entering a heavy stage, prover subprocesses acquire a node-local interprocess admission token. This prevents several already-running jobs from all entering key generation/proving at once after being admitted during lighter stages.

The fused cold EZKL path is treated as one heavy region because Rust moves directly from key generation into proving without returning to Python. Resource logs still distinguish its `KEY_GEN` and `PROVING` portions.

## Node cache

The node cache lives in:

```text
zkinfer/runtime/node/cache.py
```

It contains two kinds of data:

```text
objects/       content-addressed model/input artifacts
setup/         reusable settings, compiled circuits, PKs and VKs
```

The cache now provides:

- per-object locking so duplicate downloads do not race
- per-setup-key locking so duplicate cold setup does not race
- pinning so active setup directories cannot be evicted
- bounded LRU-style eviction
- PK compression when publishing to shared storage
- PK decompression when restoring from shared storage

Terminology in the runtime is intentionally plain:

```text
materialize artifact   -> make a local copy of an immutable object
restore setup cache    -> copy reusable setup files into the node cache
publish setup cache    -> write newly generated setup files to shared storage
```

## EZKL cold path

The custom EZKL fork adds a fused setup-and-prove operation for a cold setup miss:

```text
load circuit + witness
load SRS once
KEY_GEN
  generate PK/VK in memory
PROVING
  prove with the same in-memory PK
PERSISTING_KEYS
  save PK/VK for later warm requests
PERSISTING_PROOF
  save proof
```

This avoids generating a large PK, serializing it, returning to Python, and immediately reading/deserializing the same PK for the first proof.

Warm requests continue to use the ordinary cached-PK prove path.

The fused path records:

```text
keygen_time_s
prove_time_s
key_persist_time_s
proof_persist_time_s
total_time_s
```

See `docs/EZKL_BACKEND.md` for details.

## Scheduling

Coordinator policies are:

```text
fifo
lpt
resource_aware
cache_aware
```

All policies first filter jobs that fit the node's currently available memory. Scheduling is request-aware: the coordinator first chooses a least-served request and then applies the selected job-ranking policy within that request. This prevents one large partitioned request from monopolizing the queue.

`cache_aware` prefers jobs whose setup key is already present on the node. `resource_aware` also considers memory fit and predicted runtime.

The coordinator ignores the old protobuf `max_concurrent_jobs` field for admission. Local concurrency is owned by the node supervisor.

## Failure recovery

Coordinator requests, jobs, leases, and events can be persisted in SQLite:

```yaml
coordinator:
  state_store_enabled: true
  state_db_path: experiments/runs/coordinator_state.sqlite3
```

The state store performs in-place migration for older databases that lack `assigned_node_id`, `lease_id`, or lease timestamp columns.

A dead node causes **all** `IN_PROGRESS` jobs assigned to that node to be requeued; recovery does not rely only on the node's last heartbeat list.

Heartbeat updates that omit an active-job list preserve the previously known list instead of accidentally replacing it with an empty list.

## Verification

Enable runtime verification with:

```yaml
verification:
  enabled: true
  srs_dir: srs
  require_split_chain: false
  write_request_manifest: true
```

A request enters `VERIFYING` after every proof job succeeds. Its completion time is recorded only after verification, so request E2E time includes verification when enabled.

Verification does two things:

1. verifies every individual EZKL proof with `proof.pf`, `settings.json`, `vk.json`, and the required SRS;
2. verifies split boundaries by copying each downstream proof, applying `ezkl.swap_proof_commitments(...)` with the preceding partition's witness, and verifying the rewritten proof.

Original proof files are never modified during split-boundary verification.

Failed proof requests skip verification entirely.

## Measurement and reports

Every proof subprocess records `resource_usage.csv` with:

```text
timestamp
stage
process-tree CPU
process-tree RSS
system CPU
system memory
```

`resource_parser.py` derives both whole-job and per-stage metrics such as:

```text
stage_key_gen_max_memory(GB)
stage_key_gen_avg_cpu_machine(%)
stage_proving_max_memory(GB)
stage_proving_avg_cpu_machine(%)
```

A normal request writes:

```text
job_report.csv
perf_metrics.csv
request_report.csv
transfer_overhead_report.csv
transfer_overhead_report.json
request_artifact_manifest.json
verification_report.json        when verification is enabled
```

Backend profiling may additionally produce:

```text
halo2_circuit.csv
halo2_prover_cpu.csv
halo2_ffts.csv
halo2_msms.csv
```

Transfer accounting distinguishes bytes from shared storage, bytes served from the node cache, bytes uploaded, artifact materialization time, setup restore time, PK compression/decompression, setup publish time, and output publish time. `agg_data_movement_service_time(s)` is explicitly an aggregate service-time metric, not request critical-path time.

## Storage layout

For the filesystem backend:

```text
storage/
  artifacts/        canonical immutable objects
  cache/            shared reusable setup files
  node_cache/       bounded local node cache
  transfer/         request-scoped staging
```

For distributed deployments, `artifacts`, `cache`, and `transfer` can live in S3 while `node_cache` stays on local SSD/NVMe.

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

The old `runtime/worker_cache.py` implementation has been removed; the cache now belongs solely to the node runtime.

## Suggested validation sequence

Before large experiments, validate in this order:

```text
1. Build/install the custom EZKL wheel.
2. Run one tiny cold proof and confirm fused setup+prove is used.
3. Run the same proof warm and confirm cached-PK proving is used.
4. Inspect resource_usage.csv for correct phase transitions.
5. Run two local proof jobs and verify heavy-phase admission waits are recorded.
6. Kill a prover subprocess and then a whole node; confirm lease recovery.
7. Restart the coordinator against an existing SQLite DB.
8. Enable verification and confirm split-boundary checks pass.
9. Move to multi-node/S3 experiments.
```

After code or proto changes, run at least:

```bash
python -m compileall zkinfer
```

If `zkservice.proto` changes, regenerate the Python bindings before compiling.

## Research/evaluation direction

The runtime now supports clean experiments around:

- monolithic vs partitioned proving
- cold vs warm setup state
- memory feasibility and peak-memory reduction
- request-aware distributed scheduling
- setup-cache locality
- phase-aware local admission
- scale-out efficiency
- data-movement overhead
- verification overhead
- concurrent requests

The next research step should be empirical rather than architectural: characterize keygen/proving memory and CPU by phase, then measure whether phase-aware admission improves throughput, latency, and OOM avoidance under realistic concurrent load.
