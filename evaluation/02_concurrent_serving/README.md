# Concurrent proof serving (headline experiment)
Question: does zkInfer serve multiple requests better on the same resources?

## Exact comparisons
**fixed-1, fixed-2, fixed-4, fixed-8, whole-job, stage**.
All use resource_aware scheduling, identical partitions, profiles, input sequence,
artifact state, memory reserve and prover thread configuration. A tuned fixed-k
setting is an informative strong baseline, not only fixed-1.

## Burst experiment
Start with 1, 4, 8, 16 requests for each model separately. Extend to 32 or 64 only
when useful for identifying steady behavior; do not choose request counts to
maximize gains. Primary condition: warm; companion condition: cold.
Use the same predefined budget for the main figure (e.g. 256 GiB if feasible on
the host). State the budget separately from physical RAM.

```bash
python evaluation/run.py --baseline stage --workload mnist_gan --requests 4 --profile-dir /absolute/profiles/resource_profiles
```

For warm runs supply --setup warm --setup-dir /absolute/trial/setup.
For a mixed workload pass --workload mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd.
The generator draws models uniformly with the arrival seed. This is a synthetic
equal-probability request mix, not equal compute demand and not a production trace.
Keep individual-model results as well.

## Sustained arrivals
Pilot the approximate service rate, then choose rates below, near and above it.
Use fixed-seed Poisson arrival schedules and a fixed request count / declared
arrival window. Example syntax (0.01 requests/s is an illustration, not a chosen
experimental operating point):

```bash
python evaluation/run.py --baseline whole-job --workload mnist_gan --requests 32 --arrival poisson --rate 0.01 --profile-dir /absolute/profiles/resource_profiles
```

The traffic tool submits without waiting for prior request completion. Submission
RPCs use a bounded client pool; inspect submission lag before claiming an offered
rate was achieved. Increase pool capacity in a pilot if the injector is the limit.

## Report
Burst request completion distribution, makespan, failures; sustained achieved
throughput, latency and backlog over time. Record scheduled/start/accepted/end
times so client-side delay is visible. Include CPU/memory as explanatory signals.
Do not report p95 from four or eight requests; show points/ranges instead.
Suggested figures: completion-time comparison for bursts; latency versus offered
load for sustained traffic, with achieved throughput and failures.

## Reuse old work
Old single-request speedups are context only. The concurrent-serving measurements
must be new. Motivation simulations are not end-to-end system measurements.
