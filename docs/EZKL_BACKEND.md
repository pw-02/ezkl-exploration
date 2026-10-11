# EZKL backend notes

zkInfer uses a custom EZKL/Halo2 fork rather than treating EZKL as an opaque CLI. The serving path remains file-backed so reusable setup artifacts can live in the node cache and shared storage, but the first proof for a circuit avoids one unnecessary proving-key round trip.

## Cold versus warm proving

A **warm** request already has `pk.json`/`vk.json` in the node setup cache, or restores them from the shared setup cache. zkInfer then uses the normal `ezkl.prove(...)` path. EZKL still deserializes the cached proving key once for that proof.

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

Each proof job now runs in its own OS process. The resource logger monitors that prover process and its children, so concurrent local jobs get separate resource files and separate process-tree CPU/RSS measurements.

## Phase-aware node admission

Every prover subprocess on a node shares one local interprocess admission gate. Before entering `KEY_GEN` or `PROVING`, the process waits until a heavy-phase permit is available. The number of simultaneous heavy phases is configured with:

```yaml
node:
  max_heavy_phases: 1
```

The fused cold path holds one permit across its combined keygen/prove call because Rust moves directly from one phase into the other without returning to Python. The stage file still changes from `KEY_GEN` to `PROVING`, so resource traces retain the distinction.

`phase_admission_wait_time(s)` records how long a job waited for these local permits.

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

Then use concurrent runs to compare whole-job concurrency against phase-aware admission using throughput, latency, OOM rate, and per-stage memory/CPU traces. The system does not assume that phase-aware admission will help; the measurements should establish when it does.
