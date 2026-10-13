# EZKL backend notes

zkInfer uses a custom EZKL/Halo2 fork rather than treating EZKL as an opaque CLI. The serving path remains file-backed so reusable setup artifacts can live in the node cache and shared storage, while the cold fused path avoids immediately serializing and reloading a proving key before proof construction.

## Cold versus warm proving

A **warm** execution restores compatible settings, compiled circuit, PK, and VK from reusable setup and then generates the request-specific witness and proof.

A **cold** execution has no reusable derived setup. The custom fused path keeps the circuit, prover parameters, and newly generated proving key alive from key generation into proof construction, then persists reusable setup afterward.

```text
load witness + compiled circuit
load SRS/prover params
KEY_GEN
  generate VK/PK in memory
PROVING
  prove with the same in-memory PK
PERSISTING_KEYS
  write PK/VK for future warm requests
PERSISTING_PROOF
  write proof
```

This does not create an in-memory proving-key cache across requests. It only removes the immediate generate -> serialize -> deserialize cycle within one cold execution.

The custom EZKL wheel must be rebuilt whenever the Rust/PyO3 implementation changes.

## Key-generation instrumentation

The current EZKL instrumentation also records trace-only boundaries around:

```text
LOAD_PARAMS
KEYGEN_VK
KEYGEN_PK
```

These events are for characterization. They are not scheduler-visible admission boundaries. They should become controllable runtime stages only if measurement shows a resource-distinct boundary that the backend can safely expose.

## Stage-labelled resource traces

Each prover writes its current stage to `status.txt`. `resource_usage.csv` samples the prover process tree and records the current stage alongside CPU and RSS. Exact transition timestamps are stored separately, so stage duration does not depend on the resource sampling interval.

Typical runtime labels include:

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

When a prover is waiting for local admission, the trace records `WAITING_FOR_<STAGE>` rather than attributing the wait to the preceding stage.

## Node-local stage admission

`StageAdmissionController` gates major EZKL stages using predicted peak memory for the same setup key. `StageResourceHistory` stores recent node-local observations from successful runs.

With fewer than five prior memory observations, the estimate is the observed maximum. Once enough history exists, the controller uses the configured historical percentile. If no stage history exists, it falls back to the job's whole-job peak estimate.

The configuration is:

```yaml
node:
  stage_admission:
    enabled: true
    memory_safety_reserve_gb: 8
    history_percentile: 0.90
```

`memory_safety_reserve_gb` is a fixed node reserve that remains unavailable to proof workloads independently of prediction history.

CPU is measured for characterization but is not an admission constraint.

The fused cold call is admitted as `FUSED_KEYGEN_PROVE` because Python cannot intervene between key generation and proof construction inside the Rust call. Its memory estimate conservatively uses the larger available historical envelope from `KEY_GEN` and `PROVING` when both have been observed.

Stage-admission metrics include wait time, predicted memory, actual memory, prediction error, estimate source, and historical sample count.

## Recommended comparisons

For cold versus warm setup, compare end-to-end latency, setup/key-generation time, proof time, setup restoration/persistence, proving-key size, and peak memory by stage.

For the node admission mechanism, compare:

```yaml
node:
  stage_admission:
    enabled: false
```

against the same workload with admission enabled, holding hardware, partitioning, setup condition, and coordinator scheduling policy fixed. Useful outcomes include throughput, request latency, makespan, peak memory, OOM/failure rate, admission wait time, and memory-prediction error.
