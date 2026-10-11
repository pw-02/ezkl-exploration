# zkInfer experiment configuration and storage

This document describes the settings used by the current `sacdad27` experiment runtime. It separates experiment-facing controls from internal runtime details so runs can be reproduced without guessing what a YAML field means.

## Which config file controls what?

- `zkinfer/config/config.yaml` contains runtime defaults used by the coordinator and NodeSupervisor.
- `experiments/config/config.yaml` contains experiment-facing choices. `experiments/run_local_exp.py` translates the selected experiment settings into explicit runtime overrides when it launches the coordinator and node.

For local experiments, edit/override `experiments/config/config.yaml`. The launcher saves the resolved experiment and workload configuration under `experiments/runs/<request>/logs/`.

## Partitioning

Experiment-facing partition modes are:

- `whole_model`: one proof job for the complete ONNX model.
- `single_operator`: one proof job for each meaningful ONNX operator. Shape/passthrough operators that are folded into partition boundaries are not treated as independent proof jobs.
- `fixed_size`: group consecutive meaningful ONNX operators into partitions of `operators_per_partition` operators.

Example:

```yaml
execution:
  partitioning:
    mode: fixed_size
    operators_per_partition: 4
```

This creates proof jobs containing up to four consecutive meaningful operators. `operators_per_partition` is ignored for `whole_model` and `single_operator`.

The current RPC/splitter implementation still uses older internal strings (`none`, `single_ops`, and `fixed`) and the field name `ops_per_chunk`. `run_local_exp.py` performs that translation. Experiment configs should use the names above.

## Scheduling

`jobs.scheduler` is coordinator-wide. Supported policies are:

- `fifo`: request-fair queue order.
- `lpt`: longest predicted processing time first within the selected request.
- `cache_aware`: prefer a job whose setup is already present on the polling node; predicted duration is a tie-breaker.
- `resource_aware`: use historical per-stage memory estimates to decide whether a job can begin useful work on the node, while also considering setup locality.

Request fairness is applied before the within-request policy.

## Node concurrency

`node.max_active_jobs` is an optional whole-job cap:

```yaml
node:
  max_active_jobs: null
```

- `null`: no fixed job-count cap; stage/resource admission controls overlap.
- `1`: at most one proof job at a time on the node. This is the sequential baseline.
- `N`: at most N active proof jobs.

For a strict sequential baseline use:

```yaml
node:
  max_active_jobs: 1
  phase_admission:
    enabled: false
```

For unconstrained/naive concurrent execution use:

```yaml
node:
  max_active_jobs: null
  phase_admission:
    enabled: false
```

For normal resource-aware zkInfer execution use:

```yaml
node:
  max_active_jobs: null
  phase_admission:
    enabled: true
```

## Phase admission

`node.phase_admission` controls local admission at EZKL stage boundaries.

- `enabled`: whether a stage may wait for memory/CPU capacity before it starts.
- `memory_reserve_gb`: memory kept unavailable to zkInfer for the OS and other processes. It is not a per-job allocation.
- `cpu_budget_percent`: whole-machine CPU reservation budget used by the admission model.
- `profile_percentile`: historical percentile used once enough observations exist. Sparse history uses a conservative observed maximum.
- `fallback_cpu_percent`: CPU estimate when no historical stage estimate exists.

For isolated Section 3 characterization, use `enabled: false` so admission waits do not alter execution timing.

## Resource sampling

`node.resource_monitor_interval_sec` controls RSS/CPU sampling in `resource_usage.csv`. Exact EZKL stage transitions and Halo2 subphase transitions have their own timestamps and do not depend on this interval.

Use approximately `0.25` seconds for Section 3 characterization unless profiling overhead becomes measurable.

## Storage layout

The default layout is:

```text
storage/
  artifacts/                 shared data-plane objects and published outputs
  setup_cache/               persistent reusable proving setup
  node/
    cache/
      objects/               local copies of model/input artifacts
      setup/                 local copies of setup files
    runtime/
      reservations.json      live phase reservations
      resource_profiles/     node-local stage history
```

`experiments/runs/` is separate. It is the durable experiment record and contains per-request logs, raw per-job traces, and reports.

### `storage/artifacts`

This is not the proving cache. It is the shared data plane. The coordinator stores materialized model/input objects there and nodes publish proof-output objects there so only small `ArtifactRef` metadata needs to travel over gRPC. A local node also writes the raw job output under `experiments/runs/<request>/artifacts/jobs/<job>`.

For local experiment suites, `storage.delete_artifacts_after_run: true` in `experiments/config/config.yaml` removes `storage/artifacts` after all coordinator/node processes have stopped. It does **not** delete `experiments/runs` or the setup cache. Set it to `false` when intentionally retaining the data-plane store, for example when testing persistent/shared storage behavior.

### `storage/setup_cache`

This is the persistent warm/cold proving cache. It contains reusable files such as `settings.json`, `network.compiled`, `pk.json`, `vk.json`, and `resource_profile.json`.

Runtime settings:

```yaml
storage:
  setup_cache_enabled: true
  setup_cache_prefix: storage/setup_cache
  setup_cache_overwrite: false
```

`setup_cache_enabled: false` disables restore/publication through the shared persistent setup cache. The node-local cache is a separate layer.

### `storage/node/cache`

This is local to one NodeSupervisor machine. `objects/` contains locally materialized model/input artifacts. `setup/` contains local setup files used by prover subprocesses. Its purpose is to avoid repeated shared-storage transfers on a distributed deployment.

### `storage/node/runtime`

This is runtime state, not cached model data. Phase reservations and node-local historical resource profiles live here. Separating it from `node/cache` makes it clear that deleting cached objects is different from deleting scheduling history.

## Reporting

Raw execution data stays with each proof job:

```text
experiments/runs/<request>/artifacts/jobs/<job>/
  resource_usage.csv
  stage_transitions.csv
  halo2_stage_transitions.csv
  halo2_prover_cpu.csv
  admission_profile_before.json
  prover_process.log
  ... proof/setup artifacts produced by EZKL
```

Derived request reports live under:

```text
experiments/runs/<request>/reports/
  job_report.csv
  request_report.csv
  transfer_overhead_report.csv
  transfer_overhead_report.json
  ... verification/manifest reports when enabled
```

`job_report.csv` is the canonical per-job scalar table. The old duplicate `perf_metrics.csv` is no longer written. `request_report.csv` is an aggregate across jobs. `resource_profile.json` is not a report: it is persistent scheduling history stored with the setup cache.

Halo2 transition rows contain `timestamp_unix_us,event,stage,duration_us,duration_s`. Start events leave duration blank; matching end events carry the measured duration. Keeping start/end rows makes an incomplete phase visible if a prover terminates unexpectedly.

## Useful Section 3 configurations

Isolated fine-grained characterization:

```yaml
execution:
  partitioning:
    mode: fixed_size
    operators_per_partition: 1
node:
  max_active_jobs: 1
  resource_monitor_interval_sec: 0.25
  phase_admission:
    enabled: false
```

Naive concurrent baseline:

```yaml
node:
  max_active_jobs: null
  phase_admission:
    enabled: false
```

Resource-aware runtime:

```yaml
jobs:
  scheduler: resource_aware
node:
  max_active_jobs: null
  phase_admission:
    enabled: true
```
