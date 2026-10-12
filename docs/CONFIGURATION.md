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
              manifest.json
        config/
          coordinator_<startup>.json
          node_<startup>.json
        reports/
          job_report.csv
          request_report.csv
          ...
        tmp/                    temporary while jobs are active
```

The campaign name describes the experiment. `req_<id>` is only a unique request identifier; workload, partitioning, scheduler, cache state and timestamps belong in the saved configuration/reports rather than being encoded repeatedly in the directory name.

`artifacts/jobs/` is the durable raw-trace area for individual proof jobs, while `reports/` contains request-level derived tables. By default, large proof artifacts are **not** duplicated into this run directory; the canonical request outputs live under `storage/requests` until request-store cleanup runs.

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

### Run-artifact retention

The experiment config controls whether large generated proof files are also copied into `experiments/runs`:

```yaml
jobs:
  retain_run_artifacts: false
```

With the default `false`, the run directory keeps traces, logs and `manifest.json`, but does not duplicate `proof.pf`, `witness.json`, `vk.json`, or `settings.json`. Those files are still published to `storage/requests/<request_id>/...` so verification and downstream consumers continue to work. With `true`, the old debugging behaviour is restored and those files are also retained beside the job traces.

The tmux launchers expose the same switch through an environment variable:

```bash
RETAIN_RUN_ARTIFACTS=true experiments/scripts/launch_tmux.sh ...
```

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

Shared state has two lifetimes; local cache/runtime state is separate:

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
    proofs/<partition>/
      <lease>--proof.pf
      <lease>--witness.json
      <lease>--settings.json
      <lease>--vk.json
      <lease>--manifest.json
  node/
    cache/
      setup/<setup_key>/       local ONNX + settings/circuit/uncompressed PK/VK
      objects/                local content-addressed input copies
    runtime/
      reservations/
      resource_profiles/
```

`experiments/runs/` remains the durable raw-trace and report record. The `setup_cache` protobuf field and persisted `setup_cache_ref` carry the setup bundle reference; their legacy names remain for wire/state compatibility.

### Reusable setup

The coordinator writes the partition ONNX to `setup/<setup_key>/model.onnx`. Nodes restore or generate the derived settings/circuit/keys in the same local bundle. Newly generated derivatives are published beside the canonical ONNX; PKs are compressed for shared storage.

The setup key is **not just the model hash**. zkInfer hashes the partition model together with a versioned EZKL setup-policy identity (currently including backend, settings/calibration policy, proof type and selector-compression policy). This prevents a changed setup policy from silently reusing settings, compiled circuits, PKs or VKs generated under incompatible semantics. `ArtifactRef.sha256` still records the raw model-content hash separately for transfer integrity. When setup-affecting EZKL behaviour changes, increment `setup_identity_version` in `zkinfer/runtime/request_builder.py`.

```yaml
storage:
  backend: filesystem
  setup_prefix: storage/setup
  requests_prefix: storage/requests
  reuse_setup: true
```

`reuse_setup=false` is the cold baseline: derived setup lives in a private temporary directory per attempt and resident/shared keys are not reused. Model bytes may still be served from the node cache. The switch does not delete reusable state or node resource history. For a full transfer-cold experiment, deliberately clear the node cache too.

### Request-owned inputs and proofs

Inputs live beneath the request ID. Proofs, witnesses and verification copies are published in one folder per partition; filenames carry the lease ID so a delayed old worker cannot overwrite an accepted proof. Their `ArtifactRef` metadata retains content hashes for model/input transfer validation.

The local experiment suite uses:

```yaml
storage:
  setup_dir: storage/setup
  requests_dir: storage/requests
  reuse_setup: false
  retain_requests_after_run: false
```

With retention disabled, cleanup deletes only request IDs submitted by that suite, after stopping its processes. It preserves setup bundles, other requests, node state and experiment run records. Enable retention if another process needs the published proof or input locations after the suite exits.

### S3

For S3, prefixes are relative to `s3_prefix`:

```yaml
storage:
  backend: s3
  s3_bucket: your-bucket
  s3_prefix: zkinfer
  setup_prefix: setup
  requests_prefix: requests
  reuse_setup: true
```

This gives `s3://your-bucket/zkinfer/setup/<setup_key>/` and `s3://your-bucket/zkinfer/requests/<request_id>/`. Both launchers use this arrangement. Setup and request roots/prefixes cannot overlap.

### Configuration migration

| Previous setting | New setting |
| --- | --- |
| `storage.artifact_prefix` | `storage.requests_prefix` |
| `storage.setup_cache_prefix` | `storage.setup_prefix` |
| `storage.setup_cache_enabled` | `storage.reuse_setup` |
| `storage.transfer_prefix` | Removed; inputs are request-owned |
| `storage.setup_cache_overwrite` | Removed; use cold mode or a fresh setup prefix |
| Suite `storage.artifact_dir` | Suite `storage.requests_dir` |
| Suite `storage.retain_artifacts_after_run` | Suite `storage.retain_requests_after_run` |

The old runtime settings are rejected with migration guidance rather than silently changing cache behavior. Updated launchers use only the new settings. Existing stored files are not moved or deleted automatically. The fingerprinted setup-key scheme does not reuse bundles created under the earlier model-hash-only scheme, so the first run after this change is cold; warm up before measuring warm performance. Previously recorded artifact URIs remain explicit and can still be read by verification tools.

## Reporting

Raw execution data stays with each proof job. Under the campaign launchers the default layout is:

```text
experiments/runs/<campaign>/requests/<request>/artifacts/jobs/<job>/
  resource_usage.csv
  stage_transitions.csv
  halo2_stage_transitions.csv
  halo2_prover_cpu.csv
  admission_profile_before.json
  prover_process.log
  manifest.json
```

If `jobs.retain_run_artifacts: true` (or `RETAIN_RUN_ARTIFACTS=true` for a tmux launcher), the same directory additionally keeps `proof.pf`, `witness.json`, `vk.json`, and `settings.json`.

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
jobs:
  retain_run_artifacts: false
node:
  max_active_jobs: 1
  resource_monitor_interval_sec: 0.25
  phase_admission:
    enabled: false
```

Strict cold/no-setup-reuse characterization additionally uses:

```yaml
storage:
  reuse_setup: false
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



### Run configuration and stage traces

Coordinator and node startup save timestamped JSON snapshots under the sibling
`config/` directory of their configured `runs_dir`. Each snapshot contains the
complete resolved Hydra configuration, normalized runtime configuration, Git
commit and tracked working-tree changes, package versions, and UTC startup time.
The coordinator copies its snapshot into each request's `config/` directory;
each node also copies its snapshot into the request's `config/` directory.
Configuration is not duplicated under individual jobs. These small request
records remain available when bulky run artifacts are not retained.
A missing Git checkout is recorded as an unknown commit rather than invented.

Submit acknowledgements and request status responses report the coordinator's
actual normalized scheduler policy. The submit script no longer reports its
local `jobs.scheduler` as if that configured the remote coordinator.

For exact transitions inside fused setup/proving, rebuild and install the EZKL
wheel from `pw-02/ezkl` branch `dzkml-v10.29` after updating it. Rust emits
`fused_stage_transitions.csv` next to `status.txt`, including key generation,
proving, and key/proof persistence. Python merges those events into
`stage_transitions.csv`, preserving their UTC timestamps and monotonic elapsed
times, including partial traces from failed calls. The metrics include
`ezkl_fused_stage_trace_available` to distinguish wheels with native events.
Older wheels remain supported: admission acquisition immediately ends the waiting
state and starts `FUSED_KEYGEN_PROVE`, but detailed native transitions require
the rebuilt wheel. This logging does not change admission or SRS behavior.


### Retry folder lifecycle

Each node keeps one local `artifacts/jobs/<job>/` folder and one temporary
`tmp/<job>/` folder. A new attempt clears both before writing new traces. Known
failed or coordinator-rejected attempts remove their local folders and their own
published files; their error is still recorded in the supervisor log and result
message. The next attempt uses the same local paths. Request-level `config/` is
preserved. Local job writers are serialized with a runtime lock, including
supervisors sharing a run directory.

Shared outputs use `<lease>--<artifact>` filenames inside a single job folder;
there are no new `attempts/` subfolders. Files belonging to other leases are never
deleted by a worker. If a completion acknowledgement is lost, published files and
local traces are preserved because the coordinator may already have accepted the
proof. Historical attempt subfolders remain readable and are not bulk-deleted.


### Missing resource measurements

Stage transition timestamps are exact execution events, while RSS/CPU readings
are periodic samples. A short stage may execute entirely between samples. Such
a stage is listed with `stage_<stage>_samples=0` and
`stage_<stage>_memory_samples=0`; its memory/CPU metrics are unknown (`null` in
JSON and blank in CSV), rather than zero. A sampled row with missing or invalid
RSS does not become a zero-memory reading or erase valid CPU measurements.
Measured zero values remain zero. This change does not increase sampling rate or
estimate peaks for unsampled stages.

Request maxima ignore missing values, NaN and infinities and use only finite
measurements. If none exist, the maximum is blank. Each stage also reports
`num_jobs_with_stage_<stage>_memory_samples`, the number of jobs contributing a
valid memory maximum. A maximum based on a subset of jobs does not establish the
peak of every job. Admission error metrics likewise leave unobserved actual
usage unknown instead of inventing a zero or percentage error.

Before checking/restoring setup keys, `run_all()` records
`CHECKING_SETUP_CACHE`. The fused cold path's only `KEY_GEN` event comes from the
native key-generation transition after admission. A warm cache hit does not
report key generation; the setup-only path retains its actual keygen event.
