# zkInfer configuration

zkInfer separates experiment-facing configuration from runtime defaults.

- `experiments/config/config.yaml` describes the experiment you want to run.
- `zkinfer/config/config.yaml` contains runtime defaults for the coordinator and node processes.
- `experiments/run_local_exp.py` translates the experiment-facing settings into runtime overrides.

The experiment configuration intentionally exposes only choices that are useful for experiments or system behavior. Networking, heartbeat timing, coordinator state paths, and other plumbing remain runtime settings.

## Partitioning

The experiment configuration uses two fields only:

```yaml
execution:
  partitioning_enabled: true
  partition_size: 1
```

`partitioning_enabled: false` creates one proof job for the whole model.

`partitioning_enabled: true` partitions the model into consecutive groups of `partition_size` meaningful ONNX operators. A `partition_size` of `1` therefore gives one proof job per meaningful operator.

The runtime still translates these settings to the splitter's internal `none` and `fixed` modes, but those strings are not experiment-facing configuration.

## Coordinator scheduling

The coordinator chooses which whole proof job to lease to a polling node. The experiment-facing setting is:

```yaml
coordinator:
  scheduling_policy: resource_aware
```

Supported policies are:

- `fifo` gives request-fair queue order.
- `lpt` chooses the longest predicted job within the selected request.
- `cache_aware` prefers a job whose reusable setup is already present on the polling node, with predicted duration as a tie-breaker.
- `resource_aware` first rejects jobs whose predicted peak cannot fit on the machine, then uses cache locality and duration as coarse placement preferences.

Request fairness is applied before the within-request policy. Stage-level packing is not a coordinator responsibility.

## Node concurrency

```yaml
node:
  max_concurrent_jobs: 1
```

This is an optional hard whole-job cap.

- `1` gives the isolated sequential baseline.
- `N` gives a fixed-concurrency baseline.
- `null` removes the fixed job-count cap so local stage admission can control memory safety.

`max_concurrent_jobs` is not the main zkInfer scheduling policy. It is a safety or baseline override.

## Stage admission

The node decides when an already assigned proof job may enter its next execution stage.

```yaml
node:
  stage_admission:
    enabled: true
    memory_safety_reserve_gb: 8
    history_percentile: 0.90
```

`enabled` turns node-local stage admission on or off.

`memory_safety_reserve_gb` is RAM that zkInfer intentionally leaves unavailable to proof workloads. It is a fixed safety margin and is independent of historical predictions.

`history_percentile` controls the historical stage-memory estimate after enough observations exist. Sparse history uses the observed maximum. The default policy uses memory for admission. CPU is still measured for characterization but is not a configurable admission constraint.

For isolated Section 3 characterization, use:

```yaml
node:
  max_concurrent_jobs: 1
  stage_admission:
    enabled: false
```

For resource-aware concurrent execution after profiles have been collected, use:

```yaml
node:
  max_concurrent_jobs: null
  stage_admission:
    enabled: true
```

## Resource sampling

```yaml
node:
  resource_sampling_interval_sec: 0.25
```

This controls periodic RSS/CPU samples written to `resource_usage.csv`. Exact stage-transition timestamps are recorded separately and do not depend on the sampling interval.

## Storage

The storage configuration is organized by data lifetime.

```yaml
storage:
  reusable_setup:
    enabled: true
    path: storage/setup

  request_data:
    path: storage/requests

  local_cache:
    path: storage/node/cache
    max_gb: 50
```

`reusable_setup` contains circuit-specific state that can be reused across requests, including the partition model, settings, compiled circuit, proving key, and verification key. `enabled: false` gives the cold setup baseline. `enabled: true` permits reuse when the setup identity matches.

`request_data` contains request-owned inputs, proofs, witnesses, and manifests.

`local_cache` contains node-local materialized copies used to avoid repeated transfer or decompression. It is distinct from shared reusable setup storage.

The lower-level runtime configuration also supports `filesystem` and `s3` backends. For S3, the configured paths are interpreted as prefixes beneath `storage.s3_prefix`.

## Output retention

The experiment configuration uses two explicit switches:

```yaml
output:
  keep_proof_artifacts: false
  keep_request_data: false
```

`keep_proof_artifacts` controls whether bulky proof-generation artifacts are copied into `experiments/runs/.../artifacts/jobs/...` for debugging. Traces, logs, manifests, and reports are retained regardless of this setting.

`keep_request_data` controls whether the canonical request-owned files under `storage.request_data.path/<request_id>/` are preserved after the experiment finishes. When it is `false`, the launcher removes only the requests submitted by that experiment. Reusable setup, node state, and experiment records are preserved.

These settings affect different copies of the data. A typical characterization run therefore uses both as `false` to keep reports and traces without retaining duplicate proof files or stale request-owned storage.

## Runtime-only settings

The runtime defaults also contain settings that are not research-policy knobs, including coordinator host/port, gRPC message limits, heartbeat timing, coordinator state storage, verification configuration, and the node runtime-state directory. They remain configurable for deployment but are intentionally not duplicated in the experiment-facing policy surface.

## Useful experiment configurations

Isolated fine-grained characterization:

```yaml
execution:
  partitioning_enabled: true
  partition_size: 1

node:
  max_concurrent_jobs: 1
  resource_sampling_interval_sec: 0.25
  stage_admission:
    enabled: false

storage:
  reusable_setup:
    enabled: false

output:
  keep_proof_artifacts: false
  keep_request_data: false
```

Resource-aware runtime:

```yaml
coordinator:
  scheduling_policy: resource_aware

node:
  max_concurrent_jobs: null
  stage_admission:
    enabled: true
    memory_safety_reserve_gb: 8
    history_percentile: 0.90

storage:
  reusable_setup:
    enabled: true
```

## Terminology

The configuration uses the following terms consistently:

- **partitioning** describes how a model becomes proof jobs.
- **coordinator scheduling** chooses which whole proof job is leased to a node.
- **stage admission** decides when an assigned job may enter its next execution stage.
- **reusable setup** is circuit-specific state reused across requests.
- **request data** is owned by one inference request.
- **local cache** is node-local materialized data.

The experimental `transfer_only` runtime mode has been removed. Transfer measurements remain available as part of normal proof execution without changing proof-job semantics.
