# EZKL backend notes

zkInfer uses a custom EZKL/Halo2 fork rather than treating EZKL as an opaque CLI. The serving path is still file-backed so setup artifacts can be cached locally or in shared storage, but the first proof for a circuit now avoids one unnecessary proving-key round trip.

## Cold versus warm proving

A **warm** request already has `pk.json`/`vk.json` in the node setup cache. zkInfer materializes those files if necessary and uses the normal `ezkl.prove(...)` path. EZKL therefore still deserializes the cached proving key once for that proof.

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

This means the custom EZKL wheel must be rebuilt after updating the fork. If zkInfer reports that `_ezkl_fused` or `setup_and_prove` is unavailable, the installed wheel is older than the source checkout.

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

zkInfer records these alongside the existing request/job metrics. `ezkl_fused_unattributed_time(s)` is the small remainder of the fused call, primarily loading/preparing the witness and compiled circuit plus binding overhead.

## Stage-labelled resource traces

`zkinfer.profiling.resource_logger` now adds a `stage` column to every system and process sample in `resource_usage.csv`. It reads the job's `status.txt`, which is already updated by the EZKL stage wrapper.

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

This is intended for characterization before implementing phase-aware admission. It lets us measure CPU and memory by actual proving phase instead of inferring phase boundaries after the run.

### Current caveat

The current node runtime still executes local proof tasks in a `ThreadPoolExecutor`, and the resource logger monitors the containing Python process PID. With one local job this is useful and precise enough for stage characterization. With concurrent local jobs, per-job process attribution is not yet clean because multiple tasks share a process. Moving proof tasks to independent processes remains a separate runtime cleanup.

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

The expected direct saving is the eliminated immediate PK read/deserialization plus one redundant circuit/SRS loading path. The experiment should determine whether that saving is material for each model/partition size; no speedup is assumed in advance.
