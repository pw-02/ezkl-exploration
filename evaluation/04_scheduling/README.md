# Scheduling ablation
Question: does the coordinator policy improve request completion beyond admission?

## Run
Compare **stage-fifo**, **stage-lpt**, and **stage** on the same burst and sustained
request traces, with stage admission enabled throughout. Use warm setup primarily,
8 and 16 requests, individual workloads plus the fixed mixed workload.

Important implementation detail: fifo/lpt use a currently-available-memory
placement check; resource_aware uses machine feasibility and local admission.
Therefore stage-fifo versus stage includes placement-feasibility differences as
well as ordering/locality. Label this a coordinator-policy comparison, not a pure
LPT or locality ablation. FIFO versus LPT shares the same fit predicate and better
isolates within-request ordering.

```bash
python evaluation/run.py --baseline stage-fifo --workload mnist_gan --requests 8 --profile-dir /absolute/profiles/resource_profiles
python evaluation/run.py --baseline stage-lpt --workload mnist_gan --requests 8 --profile-dir /absolute/profiles/resource_profiles
```

## Report
Per-request completion, makespan, queue wait and job start order; for mixed models,
show per-model completion/slowdown as well as aggregates. Isolated request latency
for the same model/setup is the reference for slowdown.

## Conditional extensions
Request fairness is currently built into every coordinator policy; a fairness-off
experiment needs a new implementation toggle. Do not call FIFO request-oblivious.
A pure locality ablation also needs common fit/order logic with only cache
preference changed. If locality is a major claim, add that toggle and controlled
cache placement, then report transfer/startup costs and hit rates.
Neither missing toggle is pretended to be supported by the scripts.

## Reuse old work
Old job-duration heterogeneity motivates ordering. New concurrent ablations are
required; aggregate job-time differences alone do not prove scheduler benefit.
