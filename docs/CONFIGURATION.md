# zkInfer experiment configuration and storage

This document describes the settings used by the current `sacdad27` experiment runtime. It separates experiment-facing controls from internal runtime details so runs can be reproduced without guessing what a YAML field means.

## Which config file controls what?

- `zkinfer/config/config.yaml` contains runtime defaults used by the coordinator and NodeSupervisor.
- `experiments/config/config.yaml` contains experiment-facing choices. `experiments/run_local_exp.py` translates the selected experiment settings into explicit runtime overrides when it launches the coordinator and node.

For local experiments, edit/override `experiments/config/config.yaml`.

## Run directory layout

The tmux launchers create one **campaign directory** for an experiment invocation. The campaign directory carries the human-readable timestamp/workload/configuration label. Individual inference requests live below an explicit `requests/` directory and use short opaque IDs so the same information is not repeated in nested directory names.

```text
experiments/runs/
  2026-10-12_02-44-54_nano_gpt_4_layers_64_embd_g1_admission-false/
    config/
      launch.txt
    logs/
      coordinator.log
      coordinator.tmux.log
      node.tmux.log
      node_local-node.log
      submit.tmux.log
    requests/
      req_a5b2fa8a2c31/
        artifacts/
          jobs/
            <job-id>/
              resource_usage.csv
              stage_transitions.csv
              halo2_stage_transitions.csv
              halo2_prover_cpu.csv
              admission_profile_before.json
              prover_process.log
              ... proof outputs
        reports/
          job_report.csv
          request_report.csv
          ...
        tmp/                    temporary while jobs are active
```

The campaign name describes the experiment. `req_<id>` is only a unique request identifier; workload, partitioning, scheduler, cache state and timestamps belong in the saved configuration/reports rather than being encoded repeatedly in the directory name.

`artifacts/jobs/` is the durable raw-output area for individual proof jobs, while `reports/` contains request-level derived tables. Keeping these separate prevents generated summaries from being mixed with prover traces and proof files.

Persistent runtime state is intentionally **not** nested under the campaign directory. It lives under `storage/` so cache/setup state can persist across runs when a warm-cache experiment requires it.

Older runs may still contain verbose request directories such as `nano-gpt-..._split-fixed_ops-1_...`; those are valid historical results, but new requests use concise IDs.

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

The current RPC/splitter implementation still uses older internal strings (`none`, `single_ops`, and `fixed`) and the field name `ops_per_chunk`. Experiment launchers perform that translation. Experiment configs should use the names above.

## Scheduling

`jobs.scheduler` is coordinator-wide. Supported policies are:

- `fifo`: request-fair queue order.
- `lpt`: longest predicted processing time first within the selected request.
- `cache_aware`: prefer a job whose setup is already present on the polling node; predicted duration is a tie-breaker.
- `resource_aware`: use historical per-stage memory estimates to decide whether a job can begin useful work on the node, while also considering setup locality.

Request fairness is applied before the within-request policy.

## Node concurrency

`node.max_active_jobs` is an optional whole-job cap. The safe runtime and isolated-characterization default is `1`:

```yaml
node:
  max_active_jobs: 1
```

- `1`: at most one proof job at a time on the node. This is the isolated/sequential baseline.
- `N`: at most N active proof jobs.
- `null`: no fixed job-count cap. Use this only for experiments that intentionally study concurrency or for resource-aware operation with established profiles.

When `max_active_jobs: null`, the node still does **not** blindly lease the entire queue. Newly leased jobs create provisional memory reservations until their first real prover stage is visible. If an active job has no historical memory estimate, the node avoids leasing another unknown job unless the experiment explicitly sets a numeric concurrency limit. This prevents the polling loop from launching hundreds of provers before process RSS catches up.

For a strict sequential baseline use:

```yaml
node:
  max_active_jobs: 1
  phase_admission:
    enabled: false
```

For an intentional naive-concurrency experiment, explicitly opt in:

```yaml
node:
  max_active_jobs: null
  phase_admission:
    enabled: false
```

For resource-aware zkInfer execution after profiles have been collected:

```yaml
node:
  max_active_jobs: null
  phase_admission:
    enabled: true
```

The tmux launchers default `MAX_ACTIVE_JOBS=1`. To deliberately remove the fixed cap, run them with for example:

```bash
MAX_ACTIVE_JOBS=null experiments/scripts/launch_tmux.sh ...
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

The default local layout is:

```text
storage/
  artifacts/                 shared data-plane objects and published outputs
  transfer/                  transient request transfer objects
  setup_cache/               persistent reusable proving setup
  node/
    cache/
      objects/               local copies of model/input artifacts
      setup/                 local copies of setup files
    runtime/
      reservations/          live per-process phase reservations
      resource_profiles/     node-local stage history
```

`experiments/runs/` is separate. It is the durable experiment record and contains per-request raw job traces and reports.

### `storage/artifacts`

This is not the proving cache. It is the shared data plane. The coordinator stores materialized model/input objects there and nodes publish proof-output objects there so only small `ArtifactRef` metadata needs to travel over gRPC. A local node also writes the raw job output into the durable request run directory.

For local experiment suites, the experiment-facing switch is positive and explicit:

```yaml
storage:
  retain_artifacts_after_run: false
```

- `false`: remove `storage/artifacts` after the suite finishes.
- `true`: keep the data-plane objects for debugging or a follow-up process.

This switch never deletes `experiments/runs`, `storage/setup_cache`, or `storage/node/runtime`.

The cleanup switch exists because `storage/artifacts` represents the shared/distributed data plane, not the experiment record. On a local run it is usually a transport copy of information already represented by the durable run directory and reports, so retaining it indefinitely is unnecessary.

### `storage/setup_cache`

This is the persistent warm/cold proving cache. It contains reusable files such as `settings.json`, `network.compiled`, `pk.json`, `vk.json`, and `resource_profile.json`.

Runtime settings:

```yaml
storage:
  setup_cache_enabled: true
  setup_cache_prefix: storage/setup_cache
  setup_cache_overwrite: false
```

`setup_cache_enabled: false` disables reusable proving setup for each proof job: the node creates a private temporary setup directory rather than restoring or reusing shared/node-local settings, compiled circuits, or proving keys. This is the setting to use for a no-setup-cache/cold baseline. Model/input data-plane objects are still materialized through the node's local object cache because that cache is a transport optimization rather than proving setup reuse.

### `storage/node/cache`

This is local to one NodeSupervisor machine. `objects/` contains locally materialized model/input artifacts. `setup/` contains local setup files used when the setup cache is enabled. Its purpose is to avoid repeated shared-storage transfers on a distributed deployment.

The model/input cache is intentionally separate from `setup_cache_enabled`: disabling proving setup reuse should not force every process to repeatedly transfer identical immutable ONNX/input bytes. If an experiment is specifically measuring data-plane transfer behavior, use the transfer-only experiment/launcher and clear the node cache deliberately before the measurement.

### `storage/node/runtime`

This is runtime state, not cached model data. `reservations/` contains one small JSON reservation per admitted prover process. `resource_profiles/` contains node-local historical stage measurements. Separating this from `node/cache` makes it clear that deleting cached bytes is different from deleting scheduling history.

For distributed deployments, each physical node has its own local `storage/node/...` equivalent. The shared data-plane and setup-cache locations may instead be S3 or another shared store.

## Reporting

Raw execution data stays with each proof job. Under the campaign launchers the intended layout is:

```text
experiments/runs/<campaign>/requests/<request>/artifacts/jobs/<job>/
  resource_usage.csv
  stage_transitions.csv
  halo2_stage_transitions.csv
  halo2_prover_cpu.csv
  admission_profile_before.json
  prover_process.log
  ... proof outputs produced by EZKL
```

Derived request reports live under:

```text
experiments/runs/<campaign>/requests/<request>/reports/
  job_report.csv
  request_report.csv
  transfer_overhead_report.csv
  transfer_overhead_report.json
  ... verification/manifest reports when enabled
```

`job_report.csv` is the canonical per-job scalar table. The old duplicate `perf_metrics.csv` is no longer written. `request_report.csv` is an aggregate across jobs. `resource_profile.json` is not a report: it is persistent coordinator/shared setup-key history used for future placement, while `storage/node/runtime/resource_profiles/` is node-local admission history.

Halo2 transition rows contain `timestamp_unix_us,event,stage,duration_us,duration_s`. Start events leave duration blank; matching end events carry the measured duration. Keeping start/end rows makes an incomplete phase visible if a prover terminates unexpectedly.

## Progress reporting

The coordinator logs request progress when each job completes:

```text
Job nano_gpt_..._sub_model_17 completed | request=req_a5b2fa8a2c31 progress=18/192 (9.4%) | running=1 | queued=173 | failed=0
```

The submit process also polls request state and prints the same request-level view:

```text
Request req_a5b2fa8a2c31 | status=in_progress | completed=18/192 | running=1 | queued=173 | failed=0
```

`queued` means jobs that have not yet been leased. It is not used as a proxy for request completion. Track `completed/total`, `running`, `queued`, and `failed` together.

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

Strict cold/no-setup-reuse characterization additionally uses:

```yaml
storage:
  setup_cache_enabled: false
```

Naive concurrent baseline (explicit opt-in):

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
