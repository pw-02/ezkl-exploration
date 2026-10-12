# EZKL backend notes

zkInfer uses a custom EZKL/Halo2 fork rather than treating EZKL as an opaque CLI. The serving path remains file-backed so reusable setup artifacts can live in the node cache and shared storage, but the first proof for a circuit avoids one unnecessary proving-key round trip.

## Cold versus warm proving

A **warm** request already has `pk.json`/`vk.json` in the node setup bundle, or restores them from the shared setup store. zkInfer then uses the normal `ezkl.prove(...)` path. EZKL still deserializes the cached proving key once for that proof.

A **cold** request has no cached proving key. The old path was:

```text
EZKL setup
  load compiled circuit
  load SRS
  generate VK/PK
  serialize VK/PK to disk
  return to Python

EZKL prove
  load witness
  reload compiled circuit
  deserialize the PK just generated
  reload SRS
  prove
```

The custom `pw-02/ezkl:dzkml-v10.29` branch now contains a fused cold-path primitive. It loads the witness/circuit once, loads the prover parameters once, generates the PK, and passes that in-memory PK directly into proof creation. After proving, the PK/VK are persisted so later requests can use the normal warm cache.

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
PUBLISHING_KEYS
  publish setup artifacts to shared cache when configured
```

This optimization deliberately does **not** add an in-memory proving-key cache across requests. It only removes the immediate generate -> serialize -> deserialize cycle on a cold request.

## Python bridge

The custom EZKL shared library exports a small auxiliary PyO3 module named `_ezkl_fused` in addition to the existing `ezkl` module. `zkinfer/backends/ezkl/ezkl_prover.py` loads `_ezkl_fused` lazily from the installed EZKL extension only when a normal request has a setup-cache miss.

The custom EZKL wheel must therefore be rebuilt after updating the fork. If zkInfer reports that `_ezkl_fused` or `setup_and_prove` is unavailable, the installed wheel is older than the source checkout.

The warm path and `setup_only` path continue to use the existing EZKL bindings.

## Fused-path metrics

The fused binding reports separate timings for:

```text
keygen_time_s
prove_time_s
key_persist_time_s
proof_persist_time_s
total_time_s
```

zkInfer records these alongside the existing request/job metrics. `ezkl_fused_unattributed_time(s)` is the remainder of the fused call, primarily loading/preparing the witness and compiled circuit plus binding overhead.

## Stage-labelled resource traces

`zkinfer.profiling.resource_logger` adds a `stage` column to every system and prover-process sample in `resource_usage.csv`. It reads the job's `status.txt`, which the EZKL stage wrapper updates whenever the proof changes phase.

Typical labels are:

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

When a prover is blocked by local resource admission, the trace records `WAITING_FOR_<STAGE>` instead of attributing the wait to the previous phase.

Each proof job runs in its own OS process. The resource logger monitors that prover process and its children, so concurrent local jobs get separate resource files and separate process-tree CPU/RSS measurements.

## Resource-aware node admission

The local controller does not use a configured job count or heavy-phase count. Before each major EZKL phase, the prover subprocess requests a resource reservation based on historical measurements for the same `setup_key`.

A successful run feeds values such as:

```text
stage_key_gen_max_memory(GB)
stage_key_gen_avg_cpu_machine(%)
stage_proving_max_memory(GB)
stage_proving_avg_cpu_machine(%)
```

back into a node-local profile. Subsequent runs use those observations to estimate the requested phase. With only a few samples zkInfer uses the observed maximum; with more samples it uses the configured percentile.

If no phase history is available, admission falls back to the existing whole-job peak-memory prediction and a conservative CPU estimate. The phase runs only when its predicted demand fits alongside currently reserved phases within the node's memory and CPU budgets.

The switch is:

```yaml
node:
  phase_admission:
    enabled: true
```

Set it to `false` for the same runtime without phase-level gating. This is the intended experiment baseline.

The fused cold path reserves the larger historical resource envelope of `KEY_GEN` and `PROVING` for the combined Rust call because Python cannot intervene between the two phases. The stage file still changes internally, so resource traces retain the distinction.

Admission metrics include total/stage wait time, predicted memory/CPU, estimate source, and number of historical samples.

## What to compare after rebuilding

For the same cold workload, compare the old and fused paths using:

```text
cold end-to-end latency
PK generation time
PK serialization time
PK deserialization/read time
SRS loading time
proof time
bytes written/read for PK artifacts
peak memory by stage
```

Then compare `node.phase_admission.enabled=false` and `true` under the same hardware, workload, partitioning, and global scheduler. Useful outcomes are throughput, request latency, makespan, peak memory, OOM rate, admission wait time, CPU utilization, and prediction error between the reserved and measured stage envelopes.

