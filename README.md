# zkInfer

zkInfer is a distributed runtime for partitioned zkML proving. The current `sacdad27` branch focuses on systems infrastructure around an EZKL/Halo2 backend: model partitioning, node-level resource management, reusable proving artifacts, distributed scheduling, bounded caching, verification, and experiment accounting.

This branch is intentionally not backward-compatible with the older worker-centric runtime.

## Current architecture

```text
Client
  |
  | SubmitInferenceRequest
  v
Coordinator
  |- partitions/prepares one inference request into proving jobs
  |- keeps the global ready queue
  |- chooses a job for each polling node
  |- issues job leases
  |- persists requests/jobs/events in SQLite
  |- detects dead nodes and requeues unfinished leases
  |- records job reports and request reports
  `- optionally verifies completed request proofs
          |
          | gRPC control plane only
          v
Node Supervisor / Node Agent   (one per machine/instance)
  |- reports node memory/CPU/cache state
  |- owns the node-local cache
  |- materializes model/input/setup artifacts directly
  |- applies local concurrency safety limits
  |- launches local prover tasks
  `- publishes proof artifacts + metrics
          |
          v
Stateless Prover Task
  |- receives local filesystem paths
  |- invokes EZKL/Halo2
  |- emits proof/witness/settings/vk
  `- does not know about gRPC, scheduling, or distributed state

Shared storage
  |- content-addressed immutable artifacts
  |- shared setup cache
  |- proof output manifests
  `- S3 or filesystem backend
```

The coordinator is a control plane. Large proof/model/key objects do not move through gRPC.

## Main design decisions

### One node agent per machine

The distributed scheduling unit is a **node**, not an individual long-running prover worker. A `NodeSupervisor` represents one physical/virtual machine and reports fresh resource information when asking for work.

Coordinator-facing scheduling therefore uses resource state rather than abstract worker slots:

- available and total memory
- available and total CPU cores
- current active jobs
- cache size and capacity
- resident setup/proving artifacts

`node.max_concurrent_jobs` still exists, but only as a local safety cap to prevent runaway local overcommit. It is not the main global scheduling abstraction.

### Stateless prover tasks

`zkinfer/runtime/node/prover_task.py` contains the local proof execution path. A prover task receives local paths, runs the backend, writes results, and returns metrics. It does not communicate with the coordinator or own distributed cache policy.

This makes the proof execution layer independently testable and leaves distributed policy in the supervisor/coordinator layers.

### Node-level cache

The cache belongs to the node supervisor and is shared by all local prover tasks.

The cache is bounded by:

```yaml
node:
  local_cache_dir: storage/node_cache
  local_cache_max_gb: 100
```

A value of `0` disables eviction.

The cache stores reusable models, inputs, compiled circuits, settings, verification keys, and proving keys. Shared proving keys are stored compressed as `pk.json.gz` when possible; the local hot cache keeps the uncompressed `pk.json` expected by EZKL.

The current eviction implementation is LRU-style. More specialized proving-key-aware retention/eviction policies are a possible research direction, but are not implemented yet.

### Content-addressed artifacts

Large immutable artifacts are represented by `ArtifactRef` metadata and stored in a content-addressed artifact store.

An artifact reference currently carries:

```text
backend
uri
sha256
size_bytes
s3_bucket
```

The node materializes artifacts directly from filesystem/S3 and verifies SHA-256 when present.

### Job leases

A coordinator assignment is a lease containing metadata such as:

```text
lease_id
request_id
job_id
model ArtifactRef
input ArtifactRef
setup-cache reference
output location
predicted runtime
predicted peak memory
setup/cache key
```

Results are accepted only from the node holding the current lease. Dead-node jobs are requeued.

### Persistent coordinator state

Coordinator state is stored in SQLite when enabled:

```yaml
coordinator:
  state_store_enabled: true
  state_db_path: experiments/runs/coordinator_state.sqlite3
```

Requests, jobs, assignments, and state-change events are persisted. On restart, unfinished work is conservatively requeued while completed artifacts remain reusable.

## Scheduling

Current scheduling policies are:

```text
fifo
lpt
resource_aware
cache_aware
```

A node polls with current resources. The coordinator first filters to jobs that fit the node and then ranks feasible jobs according to the selected policy.

`cache_aware` prefers jobs whose reusable setup key is already resident on the node. `resource_aware` additionally considers memory fit and expected runtime.

Stage-aware scheduling is **not implemented** in this branch yet. It is intentionally being left for later discussion/characterization.

## Verification

Runtime verification lives under:

```text
zkinfer/verification/
  verifier.py
  request_manifest.py
  split_chain.py
```

Enable it with:

```yaml
verification:
  enabled: true
  srs_dir: srs
  require_split_chain: false
  write_request_manifest: true
```

When enabled, the coordinator materializes each completed job's proof artifacts and runs `ezkl.verify(...)` using:

```text
proof.pf
settings.json
vk.json
required SRS
```

A request is not considered successfully verified unless all job proofs verify. Split-chain verification status is recorded separately so that "all subproofs are valid" is not silently treated as "their boundaries form a valid composed execution."

`require_split_chain` should only be enabled when the request artifacts contain the boundary evidence required by the split-chain checker.

A completed request can also emit:

```text
request_artifact_manifest.json
```

which records the final artifact references, verification information, and aggregate experiment metrics.

## Transfer and cache overhead accounting

The runtime records data-plane overhead so scale-out experiments can distinguish useful proving speedup from storage/network cost.

Current job-level accounting includes values such as:

```text
model/input cache hit
model/input bytes downloaded
model/input bytes served locally
setup-cache hydrate time
setup bytes downloaded
PK decompression time
setup publish bytes/time
PK compression time
proof/output publish bytes/time
cache evictions and evicted bytes
```

Request-level aggregation is written by:

```text
zkinfer/profiling/overhead.py
```

and produces:

```text
transfer_overhead_report.json
transfer_overhead_report.csv
```

Useful scale-out metrics include:

- bytes downloaded from shared storage
- bytes uploaded to shared storage
- bytes served from the node cache
- node-cache byte hit rate
- setup hydrate/publish time
- proof output publish time
- cache eviction cost

## Important runtime directories

```text
zkinfer/
  backends/
    ezkl/
      ezkl_prover.py

  config/
    config.yaml
    runtime.py

  profiling/
    overhead.py
    reports.py
    resource_logger.py
    resource_parser.py
    msm_fft_parser.py

  proto/
    zkservice.proto
    zkservice_pb2.py
    zkservice_pb2_grpc.py

  runtime/
    coordinator.py
    coordinator_grpc.py
    data_plane.py
    models.py
    request_builder.py
    scheduler.py
    state_store.py

    node/
      cache.py
      prover_task.py
      supervisor.py

  storage/
    artifact_store.py
    io.py
    s3.py

  verification/
    verifier.py
    request_manifest.py
    split_chain.py
```

`zkinfer/runtime/worker.py` is currently only a compatibility entrypoint/shim around the node-supervisor architecture. New code should use the node terminology.

## Storage layout

With the filesystem backend, the important namespaces are conceptually:

```text
storage/
  artifacts/        # canonical immutable/content-addressed objects
  cache/            # shared setup/proving cache
  node_cache/       # bounded cache local to a node
  transfer/         # request-scoped scratch/control-plane related staging
```

For distributed runs, the shared namespaces can live in S3 while `node_cache` remains local SSD/NVMe.

## Configuration sketch

The current defaults are approximately:

```yaml
coordinator:
  host: localhost
  port: 50051
  state_store_enabled: true
  state_db_path: experiments/runs/coordinator_state.sqlite3

node:
  node_id: null
  job_poll_interval_sec: 5
  resource_monitor_interval_sec: 3
  max_concurrent_jobs: 1
  local_cache_dir: storage/node_cache
  local_cache_max_gb: 100

jobs:
  scheduler: cache_aware
  max_retries: 1
  transfer_only: false

verification:
  enabled: false
  srs_dir: srs
  require_split_chain: false
  write_request_manifest: true

storage:
  backend: filesystem       # filesystem | s3
  transfer_prefix: storage/transfer
  artifact_prefix: storage/artifacts
  proving_cache_enabled: true
  proving_cache_prefix: storage/cache
  proving_cache_overwrite: false
  s3_bucket: null
  s3_prefix: zkinfer
```

## Generated reports

A normal request can generate files under its run/report directory such as:

```text
job_report.csv
perf_metrics.csv
request_report.csv
transfer_overhead_report.csv
transfer_overhead_report.json
request_artifact_manifest.json
```

Backend profiling may additionally generate:

```text
resource_usage.csv
halo2_circuit.csv
halo2_prover_cpu.csv
halo2_ffts.csv
halo2_msms.csv
```

These are useful for later characterization of key generation, proving, FFT/MSM work, memory pressure, and transfer overhead.

## Running after a proto change

Whenever `zkinfer/proto/zkservice.proto` changes, regenerate Python bindings:

```bash
python -m grpc_tools.protoc \
  -Izkinfer \
  --python_out=zkinfer/proto \
  --grpc_python_out=zkinfer/proto \
  zkinfer/proto/zkservice.proto
```

Then run a syntax/import compilation pass:

```bash
python -m compileall zkinfer
```

The generated protobuf files are checked into the repository, so regenerate and commit them whenever the schema changes.

## Suggested test progression

Start with the filesystem backend before moving to S3.

```text
1. Coordinator + one node, verification disabled.
2. Same run with a warm node cache.
3. Verification enabled.
4. Multiple node agents using filesystem/shared storage where applicable.
5. S3-backed distributed run.
6. Cache-capacity experiments.
7. Multiple concurrent inference requests.
```

For early correctness testing, keep `node.max_concurrent_jobs: 1`. Increase it only after memory/resource accounting has been validated.

## What is not implemented yet

The following are deliberately not claimed as finished:

- stage-aware scheduling inside the proving pipeline
- SLO/deadline-aware request scheduling
- request batching
- sophisticated multi-tenant fairness/admission control
- proving-key regeneration-cost-aware eviction
- automatic proving-key replication/prefetch
- autoscaling

Those are potential next research/system directions rather than hidden behavior in the current code.

## Current research/evaluation framing

The current system already provides a useful base for studying:

- monolithic vs partitioned zkML proving
- cold vs warm reusable setup state
- memory feasibility and peak-memory reduction
- scale-out across node agents
- cache/proving-key locality
- storage/network overhead as nodes are added
- verification overhead
- multiple concurrent requests

A reasonable baseline ladder is:

```text
monolithic EZKL
partitioned zkInfer on one node
partitioned distributed FIFO
memory/resource-aware placement
cache-aware placement
```

The infrastructure in this branch is intended to make those experiments reproducible before introducing more speculative scheduling mechanisms.
