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
  |- uses historical resource profiles during placement
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
  |- records exact stage transitions and sampled resources
  `- exits after one proof job

Shared storage
  |- setup bundles and request-owned inputs
  |- reusable setup store
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

This means a node may run one, two, or several phases concurrently depending on their measured resource demand. There is no fixed statement such as "only one keygen" or "at most four jobs".

Admission also checks live available RAM. Waiting processes and retained allocations remain charged through that reading; only predicted growth beyond each process tree's existing RSS is added to it. A phase whose predicted CPU or memory demand exceeds its budget fails immediately. These are prediction-based checks, not OS-enforced memory limits.

The fused cold EZKL operation is the one exception to phase boundaries visible to Python: Rust moves directly from key generation into proving. zkInfer therefore reserves the larger historical resource estimate of `KEY_GEN` and `PROVING` for that combined region while the stage trace still records the internal transition.

### Admission metrics

Each job records whether phase admission was enabled, total wait time, stage-level prediction metadata, and prediction error, for example:

```text
phase_admission_enabled
phase_admission_wait_time(s)
phase_admission_proving_wait_time(s)
phase_admission_proving_predicted_memory(GB)
phase_admission_proving_actual_memory(GB)
phase_admission_proving_memory_error(GB)
phase_admission_proving_memory_abs_pct_error
phase_admission_proving_predicted_cpu_machine(%)
phase_admission_proving_actual_cpu_machine(%)
phase_admission_proving_cpu_abs_pct_error
phase_admission_proving_estimate_source
phase_admission_proving_estimate_samples
```

This makes both the admission decision and the quality of the learned estimate directly measurable.

## Exact stage/resource traces

Each prover writes its current stage to `status.txt`. Two complementary traces are emitted:

```text
stage_transitions.csv
  exact enter-time for every stage/wait state
  microsecond wall-clock timestamp
  monotonic elapsed_s

resource_usage.csv
  periodic CPU/RSS samples
  microsecond wall-clock timestamp
  monotonic elapsed_s
  current stage label
```

Waiting is labelled explicitly as `WAITING_FOR_<STAGE>` so queueing at the local admission controller is not misattributed to the previous EZKL phase. Exact stage durations therefore do not depend on the resource sampling interval.

The normal resource sampling period is 1.0s, but `node.resource_monitor_interval_sec` is a floating-point value. Characterization runs can use `0.25` or `0.5` seconds when short NanoGPT-style phases need finer CPU/RSS resolution.

The resource logger tracks the prover process tree while excluding the logger itself. System rows additionally expose node packing context such as active job count and the set of currently observed job phases. Where reservation metadata is available, the trace also has columns for predicted memory/CPU reserved by concurrent phases.

`resource_parser.py` derives whole-job and per-stage CPU/RSS measurements, and successful jobs feed those measurements back into the historical profile store.

Every `perf_metrics.csv` and `job_report.csv` row carries the stable `setup_key`, so repeated executions of the same circuit can be grouped directly across requests and inputs.

Before a proof starts, zkInfer writes `admission_profile_before.json` into the job artifact directory. This is the exact historical profile visible to the admission controller before that execution contributes a new observation, making learned-profile experiments reproducible.

The custom Halo2 profiler additionally emits labels such as `PROVING::<subphase>` and `halo2_stage_transitions.csv`, allowing proving itself to be characterized without making Halo2 subphases independently schedulable.

## Multi-request execution

The coordinator supports multiple inference/proof requests being active at once. Jobs from all active requests share the global ready queue. Requests rotate in arrival order after each dispatch, skipping requests with no feasible job on the polling node; the configured job-ranking policy is applied within the chosen request. Newly arriving requests join the rotation rather than receiving a burst to catch up with older requests' cumulative dispatch counts.

Scheduling policy is configured once at the coordinator through `jobs.scheduler`. Requests do not select independent policies through the client API.

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

`experiments/submit_batch.py` reports both request throughput and proof-job throughput:

```text
completed_requests_per_s
completed_proof_jobs_per_s
batch makespan
completed/failed requests
completed/total proof jobs
```

Proof-job throughput is important because request sizes can differ substantially after partitioning.

## Storage lifetimes

The shared store groups files by their lifetime:

```text
storage/
  setup/<setup_key>/
    model.onnx
    settings.json
    network.compiled
    pk.json.gz
    vk.json
    resource_profile.json
  requests/<request_id>/
    inputs/<partition>.json
    proofs/<partition>/attempts/<lease>/
      proof.pf
      witness.json
      settings.json
      vk.json
      manifest.json
  node/
    cache/
      setup/<setup_key>/      local model.onnx + uncompressed derived setup
      objects/               local immutable input copies
    runtime/
      reservations/          current phase reservations
      resource_profiles/     node-local resource history
```

The setup key is SHA-256 of the serialized partition ONNX, which includes its weights and tensor shapes. Identical partition bytes share one bundle across requests. Different requests always have separate input and proof locations. Backend/calibration configuration changes require a fresh setup prefix; the model hash alone does not version those settings.

Runtime configuration is:

```yaml
storage:
  setup_prefix: storage/setup
  requests_prefix: storage/requests
  reuse_setup: true
```

`reuse_setup=false` computes derived settings/circuits/keys in a private per-attempt temporary directory, even if warm keys already exist locally. Immutable ONNX transfer caching remains available. SRS handling is unchanged.

The node pins the model/setup bundle and inputs until the job finishes. It serializes setup mutation per key, compresses PKs for shared publication, and restores them uncompressed locally. Model materialization is counted separately from derived setup restoration, so transfer bytes are not charged twice. Only complete derived bundles are advertised as warm.

New jobs use this layout. Existing artifacts are not automatically moved or deleted, and new SHA-256 setup identities do not reuse the old MD5 cache entries. Warm up the new bundles before measuring warm serving. The protobuf field `setup_cache` and persisted `setup_cache_ref` remain transport compatibility names for the bundle reference.

See `docs/CONFIGURATION.md` for cleanup, S3 configuration, and old-to-new configuration mappings.

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
  save PK/VK for future cached-setup requests
PERSISTING_PROOF
  save proof
```

This avoids generating a large PK, serializing it, returning to Python, and immediately reading/deserializing the same PK for the first proof. Cached-setup requests still use the normal cached-PK prove path. There is deliberately no cross-request in-memory PK cache.

See `docs/EZKL_BACKEND.md` for details.

## Scheduling

Coordinator policies are:

```text
fifo
lpt
resource_aware
cache_aware
```

There is no coordinator-facing job-slot count. Nodes report memory, CPU, cache state, and active jobs. `resource_aware` uses historical stage profiles to determine whether a proof can start useful work on a node while the node-local controller remains responsible for later stage admission. `cache_aware` prefers jobs whose setup key is already present on the node.

Global scheduling and local phase admission are separate responsibilities:

```text
Coordinator
  chooses which whole proof job should run on which node
  uses request fairness, cache state, and historical resource profiles

NodeSupervisor / ResourceAdmissionController
  uses live node state and historical profiles
  decides when each local phase can safely enter execution
```

A proof job remains on the node to which it was leased; zkInfer does not migrate a partially executed proof between machines.

## Failure recovery and persistent state

Coordinator state can be persisted in SQLite:

```yaml
coordinator:
  state_store_enabled: true
  state_db_path: experiments/runs/coordinator_state.sqlite3
```

The current state store still contains migration code for older worker-centric databases. This is not part of the new runtime abstraction and can be removed once old experiment databases are no longer needed.

A dead node causes every `IN_PROGRESS` job assigned to that node to be requeued. Recovery does not depend only on the node's last heartbeat list. Heartbeat updates that omit an active-job list preserve the previously known list.

Each attempt writes raw outputs beneath `artifacts/jobs/<job>/attempts/<lease>/` and publishes to an attempt-specific shared prefix. A stale attempt therefore cannot overwrite the accepted attempt's artifacts. Completion requires the current lease ID; matching terminal-result retries are acknowledged without repeating reports or verification, including after coordinator restart when persistent state is enabled. `job_report.csv` records the accepted `lease_id`, and Section 3 tools use it to select the canonical trace while retaining support for legacy paths.

Shared resource profiles are merged under a per-profile lock and written atomically on the filesystem. New observations refresh queued jobs for the same profile. Request finalization waits until all accepted job reports have finished writing.

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

Per-job instrumentation may additionally produce:

```text
status.txt
stage_transitions.csv
resource_usage.csv
admission_profile_before.json
halo2_stage_transitions.csv
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

For Section 3 characterization, use finer sampling and keep admission disabled:

```bash
python experiments/run_local_exp.py \
  "suite.workloads=[mobilenet_v2]" \
  node.resource_monitor_interval_sec=0.25 \
  node.phase_admission.enabled=false
```

The full Section 3 experiment matrix, interpretation rules, expected tables, and expected figures are documented in `analysis/section_3/README.md`.

Generate the Section 3 analysis with:

```bash
python analysis/section_3/analyze_characterization.py \
  --runs-dir experiments/runs \
  --output-dir analysis/section_3/generated \
  --memory-budget-gb <usable-node-memory>
```

The analysis compares whole-job, EZKL-stage, and Halo2-subphase memory accounting; measures repeatability by setup key; evaluates historical prediction accuracy; separates cold and cached-setup executions; and summarizes observed multi-request concurrency.

A useful paper workflow is:

```text
1. Granularity characterization
   existing full-model -> operator-level time/memory experiment

2. Phase characterization
   isolated cold and cached-setup jobs
   stage duration + stage peak RSS/CPU

3. Repeatability
   same setup_key across many inputs/runs
   prediction from runs 1..k versus actual run k+1

4. Reservation opportunity
   compare whole-job peak reservation with stage-specific resource estimates
   using traces, without yet claiming end-to-end speedup

5. Runtime evaluation
   phase admission disabled versus enabled
   single request, concurrent requests, batch and mixed workloads
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
    resource_profiles.py
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
python -m unittest discover -s tests -v
```

Then run one tiny cold proof and one cached-setup repeat, inspect `stage_transitions.csv`, `halo2_stage_transitions.csv`, and `resource_usage.csv`, and confirm the resource trace contains Halo2 labels such as `PROVING::<subphase>` before starting the larger characterization campaign.

## Research/evaluation direction

The runtime is set up to evaluate:

- monolithic vs partitioned proving
- cold vs cached setup state
- fused cold setup/prove vs the previous file round trip
- historical-resource-aware global placement
- historical-resource-aware local admission
- repeatability and prediction error for stage resource estimates
- whole-job reservation slack versus stage-specific reservation
- EZKL-stage versus Halo2-subphase resource variation
- memory feasibility and OOM avoidance
- multi-request fairness and throughput
- online latency versus offline batch makespan
- setup-bundle locality
- scale-out efficiency
- data-movement overhead
- verification overhead

The main admission experiment should hold hardware, workload, partitioning, and global scheduler fixed while toggling `node.phase_admission.enabled`. Section 3 should first establish whether measured stage-level resource variation is large and repeatable enough to justify that mechanism.

