# Memory-admission sensitivity
Question: does changing the reservation boundary improve serving under memory pressure?

## Run
Compare **whole-job** versus **stage** only. Use the same resource_aware scheduler,
same process cap (null), threads, setup and profile snapshots. Use 8 and 16 requests
initially, cold and warm separately. Suggested usable budgets: 64, 128, 256 GiB,
subject to all jobs fitting and the host having enough physical memory.
Predeclare the sweep and keep small/negative gains.

Set --reserve-gib to measured host total minus desired budget; see 00_setup.
This is an accounting-budget experiment, not an OS memory-limit experiment.
Use warm setup for the runnable controlled comparison (add --setup warm and
--setup-dir). Profiled cold comparison is planned but requires the profile-loading
decoupling described in 00_setup; the runner rejects it by default.

```bash
python evaluation/run.py --baseline whole-job --workload mnist_gan --requests 8 --reserve-gib 499 --profile-dir /absolute/profiles/resource_profiles
python evaluation/run.py --baseline stage --workload mnist_gan --requests 8 --reserve-gib 499 --profile-dir /absolute/profiles/resource_profiles
```

499 is only an example for a host reporting exactly 755 GiB and budget 256 GiB.
Replace it using actual host totals. Keep a fixed reserve for each comparison.

## Report
Request latency and makespan versus budget, actual peak memory, failures, time
waiting for admission, and one representative stage/reservation timeline.
A faster run that fails requests is not an unqualified improvement. Explain the
whole-job reservation boundary (prover pipeline, not supervisor materialization).
Changing memory accounting can also change execution order despite using the same
selection rule; do not claim an optimal or purely order-independent packing bound.

## Reuse old work
Stage traces, repeatability and the packing simulation justify the experiment.
They cannot replace concurrent runtime results. In particular, cold opportunity
does not establish warm-serving opportunity.
