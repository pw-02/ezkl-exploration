# zkInfer

zkInfer is a distributed runtime for partitioned zkML proving with an EZKL/Halo2 backend. The `sacdad27` branch focuses on systems behavior: partitioning, resource-aware local execution, reusable proving artifacts, multi-request scheduling, verification, failure recovery, and experiment accounting.

Backward compatibility with the older worker-centric runtime is not a goal on this branch.

## Architecture

```text
Client(s)
   |
   | inference/proof requests
   v
Coordinator
  |- prepares each request as many proof jobs
  |- keeps one global ready queue across requests
  |- applies request-aware scheduling/fairness
  |- chooses feasible jobs when nodes poll
  |- issues leases and persists state in SQLite
  |- requeues jobs from failed nodes
  |- writes request/job/data-movement reports
  `- verifies completed requests asynchronously
          |
          | gRPC control plane only
          v
NodeSupervisor                         one per machine
  |- reports fresh CPU/memory/cache state
  |- owns the bounded node cache
  |- restores reusable setup files from shared storage
  |- launches one isolated OS process per proof job
  |- admits each prover phase from predicted resource demand
  `- publishes results + measured resource profiles
          |
          v
Prover subprocess
  |- receives local filesystem paths
  |- runs EZKL/Halo2
  |- asks for admission before each major EZKL phase
  |- records its exact stage in resource samples
  `- exits after one proof job

Shared storage
  |- content-addressed model/input artifacts
  |- reusable setup cache
  `- proof/result manifests
```

Large model, witness, key, and proof objects do not move through gRPC.

## Resource-aware phase admission

zkInfer no longer uses `local_concurrency_limit` or `max_heavy_phases` as the normal execution policy. The number of locally overlapping jobs/phases should emerge from node capacity and measured workload demand, not from manually chosen counts.

The relevant configuration is:

```yaml
node:
  phase_admission:
    enabled: true
    memory_reserve_gb: 8
    cpu_budget_percent: 100
    profile_percentile: 0.90
    fallback_cpu_percent: 100
    emergency_process_limit: null
```

`enabled` is the experiment switch. With `true`, every major EZKL phase reserves predicted CPU/memory before entering. With `false`, the same runtime executes without phase-level resource gating, giving a direct baseline.

`memory_reserve_gb` is an OS/safety reserve, not a process-count limit. `cpu_budget_percent` expresses the usable whole-machine CPU budget. `emergency_process_limit` is optional last-resort protection against a software bug spawning an unbounded number of processes; `null` means there is no normal count-based concurrency policy.

### Historical resource profiles

Each completed proof contributes measured per-stage resource observations for its `setup_key`. Profiles are stored in the node-local cache under the phase-admission metadata directory and currently retain recent observations for stages such as:

```text
CALIBRATING
COMPILING
GETTING_SRS
GENERATING_WITNESS
KEY_GEN
PROVING
```

For each stage the runtime records values such as:

```text
stage_key_gen_max_memory(GB)
stage_key_gen_avg_cpu_machine(%)
stage_proving_max_memory(GB)
stage_proving_avg_cpu_machine(%)
```

The admission controller uses a conservative estimate from recent runs of the same setup key. With fewer than five observations it uses the observed maximum; after that it uses the configured percentile. This lets repeated jobs gradually replace conservative fallback estimates with measured stage behavior.

If there is no stage history yet, zkInfer falls back to the job's existing whole-job `predicted_peak_memory_gb`. Unknown CPU demand defaults conservatively so the first unprofiled run does not assume cheap compute.

Conceptually, a phase is admitted when its predicted demand fits alongside currently reserved phases:

```text
sum(reserved phase memory) + predicted phase memory
    <= total node memory - memory reserve

sum(reserved phase CPU) + predicted phase CPU
    <= configured CPU budget
```

This means a node may run one, two, or several phases concurrently depending on their measured envelopes. There is no fixed statement such as "only one keygen" or "at most four jobs".

The fused cold EZKL operation is the one exception to phase boundaries visible to Python: Rust moves directly from key generation into proving. zkInfer therefore reserves the larger historical envelope of `KEY_GEN` and `PROVING` for that combined region while the stage trace still records the internal transition.

### Admission metrics

Each job records whether phase admission was enabled, total wait time, and stage-level prediction metadata, for example:

```text
phase_admission_enabled
phase_admission_wait_time(s)
phase_admission_proving_wait_time(s)
phase_admission_proving_predicted_memory(GB)
phase_admission_proving_predicted_cpu_machine(%)
phase_admission_proving_estimate_source
phase_admission_proving_estimate_samples
```

This makes admission behavior directly measurable rather than hidden inside the runtime.

## Exact stage/resource traces

Each prover writes its current stage to `status.txt`, and every `resource_usage.csv` sample carries the same stage label. Waiting is labelled explicitly as `WAITING_FOR_<STAGE>` so queueing at the local admission controller is not misattributed to the previous EZKL phase.

The resource logger tracks the prover process tree while excluding the logger itself. `resource_parser.py` derives whole-job and per-stage CPU/RSS measurements, and successful jobs feed those measurements back into the historical profile store.

## Multi-request execution

The coordinator supports multiple inference/proof requests being active at once. Jobs from all active requests share the global ready queue, but scheduling is request-aware: zkInfer first chooses among least-served requests and then applies the configured job-ranking policy within the chosen request. This prevents one large partitioned request from consuming the cluster simply because it contains many more partitions.

This supports two useful operating modes:

```text
online / interactive
  individual requests arrive over time
  latency and fairness matter

offline / batch
  many inputs or requests are queued together
  throughput, utilization, cache reuse, and makespan matter
```

The runtime is intentionally not tied to only one of these modes. Resource-aware local admission is especially relevant to batch proof generation because the scheduler can keep nodes busy while preventing combinations of proof phases that exceed machine capacity.

A natural batch workload is repeated proof generation for a model over many inputs: setup artifacts can be reused while witness/proof work is scheduled across the cluster. Evaluation should therefore report both per-request latency and aggregate throughput/makespan.

## Node cache

The node cache lives in `zkinfer/runtime/node/cache.py` and contains:

```text
objects/       content-addressed model/input artifacts
setup/         reusable settings, compiled circuits, PKs and VKs
phase_admission/
  profiles/    measured per-setup-key stage resource history
  reservations/ current cross-process phase reservations
```

The cache provides per-object locking, per-setup-key locking, active-setup pinning, bounded LRU-style eviction, PK compression when publishing setup state, and PK decompression when restoring it.

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

All policies first require a job to fit the node's current reported resources. `cache_aware` prefers jobs whose setup key is already present on the node. The old protobuf `max_concurrent_jobs` field is retained only for wire compatibility and is ignored as an admission primitive.

Global scheduling and local phase admission are separate responsibilities:

```text
Coordinator
  decides which whole proof job should run on which node

NodeSupervisor / ResourceAdmissionController
  decides when each local phase can safely enter execution
```

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

When the final proof job reports success, request verification and final report generation run asynchronously in the coordinator. The request enters `VERIFYING`, and its completion timestamp is recorded only after verification. End-to-end request time therefore includes verification when enabled.

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

Report CSV writing is schema-safe: if later jobs expose additional stage/admission columns, the report is rewritten with the union schema rather than appending malformed rows.

Data-movement accounting includes model/input materialization, setup restoration, PK compression/decompression, setup publication, output publication, shared-store bytes, and node-cache bytes. `agg_data_movement_service_time(s)` is an aggregate service-time metric, not request critical-path latency.

## Experiments

The local launcher follows the real architecture: one NodeSupervisor represents one machine.

Admission enabled:

```bash
python experiments/run_local_exp.py \
  "suite.workloads=[mnist_gan]" \
  node.phase_admission.enabled=true
```

Baseline with the phase controller disabled:

```bash
python experiments/run_local_exp.py \
  "suite.workloads=[mnist_gan]" \
  node.phase_admission.enabled=false
```

`analysis/analyze_phase_admission.py` summarizes request- and stage-level measurements. The important comparison is no longer a manually chosen heavy-phase count; it is the same workload/hardware with resource-aware admission **off versus on**.

Useful admission experiments include:

```text
single request, admission off/on
multiple simultaneous requests, admission off/on
cold setup versus warm setup
one node versus multiple nodes
profiles empty versus profiles learned from prior runs
interactive latency versus batch throughput/makespan
```

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

`runtime/worker_cache.py` has been removed. `runtime/worker.py` remains only as a compatibility entrypoint that forwards to the node supervisor.

## Validation sequence

The custom EZKL/Rust build and full runtime tests must be run in an environment containing the sibling Halo2 checkouts used by the EZKL fork.

Recommended validation order:

```bash
# 1. Rebuild the custom EZKL Python extension.
cd <ezkl-checkout>
maturin develop --release --features python-bindings

# 2. Python compilation.
cd <zkinfer-checkout>
python -m compileall zkinfer experiments analysis tests

# 3. Core scheduler/cache/state/admission regressions.
python -m unittest tests.test_runtime_core -v
```

Then run one tiny cold proof, repeat it warm so a historical phase profile is created, inspect `resource_usage.csv`, compare `node.phase_admission.enabled=false` and `true`, exercise multiple simultaneous requests, kill a prover subprocess and then a whole node to test recovery, restart the coordinator against an existing SQLite DB, and finally enable verification.

## Research/evaluation direction

The runtime is set up to evaluate:

- monolithic vs partitioned proving
- cold vs warm setup state
- fused cold setup/prove vs the previous file round trip
- historical-resource-aware local admission
- prediction error for stage resource estimates
- memory feasibility and OOM avoidance
- multi-request fairness and throughput
- online latency versus offline batch makespan
- setup-cache locality
- scale-out efficiency
- data-movement overhead
- verification overhead

The main phase-admission experiment should hold hardware, workload, partitioning, and global scheduler fixed while toggling `node.phase_admission.enabled`. If the learned resource envelopes improve utilization/throughput or avoid OOMs without unacceptable latency inflation, the mechanism is a defensible zkInfer contribution.
