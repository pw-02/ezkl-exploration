# zkInfer evaluation

This directory is the experiment protocol, not a collection of measured results.
Run from the repository root with the project's normal EZKL/runtime environment.
The main contribution tested here is serving partitioned proofs on shared resources.
Partitioning, extra hardware, admission, scheduling, and setup reuse are measured
separately so improvements are not attributed to the wrong mechanism.

## Reading and execution order

1. [Setup and measurement](00_setup/README.md): required controls and reporting.
2. [Decomposition](01_decomposition/README.md): reuse/refresh old single-worker results.
3. [Concurrent serving](02_concurrent_serving/README.md): highest-priority new experiment.
4. [Memory admission](03_memory_admission/README.md): isolate stage versus job accounting.
5. [Scheduling](04_scheduling/README.md): isolate job ordering/locality.
6. [Scale-out](05_scale_out/README.md): reuse old single-request curves; add serving.
7. [Partition granularity](06_partition_granularity/README.md): move old Section 3 here.
8. [Setup reuse](07_setup_reuse/README.md): reuse isolated setup measurements.
9. [Optional claims](08_optional/README.md): recovery, overhead, second backend.

## Exact baseline matrix

| Experiment | Configurations to compare |
| --- | --- |
| Decomposition | monolithic, partitioned-sequential |
| Concurrent serving | fixed-1, fixed-2, fixed-4, fixed-8, whole-job, stage |
| Memory-budget sensitivity | whole-job, stage |
| Scheduling | stage-fifo, stage-lpt, stage (resource_aware) |
| Scale-out | whole-job and stage, each at every node count |
| Partition granularity | stage at sizes 1, 2, 4, 8; optional sequential diagnostic |
| Reuse | stage cold versus stage warm |

`stage` is the zkInfer configuration in these comparisons. All concurrent
admission baselines use the same `resource_aware` coordinator policy. Fixed-k
retains the ordinary node resource checks; it is not completely memory-unaware.
FIFO is request-fair FIFO, not a globally request-oblivious queue.

## Tools

- `configs/baselines.json`: explicit runtime settings for every arm.
- `run.py`: prints concrete coordinator/node/load-generator commands by default;
  `--execute` runs a one-node experiment and saves commands, logs and outcomes.
- `traffic.py`: submits bursts or seeded Poisson arrivals to an already running
  coordinator, recording every request including errors and timeouts.
- Each subsection has a `matrix.json` describing its planned sweep. These matrices
  are protocols, not an automatic launcher for an expensive full sweep.

Start with a dry run:

```bash
python evaluation/run.py --baseline fixed-2 --workload mnist_gan --requests 4
```

Then a cold one-node smoke run (with actual models/backend installed):

```bash
python evaluation/run.py --baseline fixed-2 --workload mnist_gan --requests 1 --execute --output evaluation/results/smoke-fixed2
```

For profiled warm admission comparisons, prepare matched artifacts and node-local
resource profiles first, then use fresh per-trial copies of both:

```bash
python evaluation/run.py --baseline whole-job --workload mnist_gan --requests 4 --setup warm --setup-dir /absolute/trial/setup --profile-dir /absolute/profiles/resource_profiles --execute --output evaluation/results/warm-whole-r1
```

Repeat with `--baseline stage`, a separate output directory, and equivalent cache
and profile snapshots. `--profile-dir` must contain profile JSON files; the runner
copies them into a fresh node state directory. It does not warm an experiment
silently. Warm mode requires an existing nonempty setup directory; verify actual
cache hits in the reports because directory existence alone is insufficient.

To test the same requests on a running multi-node deployment, use `traffic.py`
directly. The scripts do not provision cloud machines, alter cgroups, or inject
failures. No complete proof workload was run when these scripts were authored.
The first real run must be checked against the measurement protocol before scale-up.

## Scope and known boundaries

- Whole-job reservation starts at the prover pipeline, after supervisor artifact
  materialization, and ends after output publication. Stage admission follows the
  backend's existing controllable boundaries; these scripts do not change them.
- The node uses host `psutil.virtual_memory()`. A container memory limit is not
  automatically an admission budget. Use the documented reserve calculation.
- Profiling and artifact state are different: cold artifacts do not imply unknown
  memory demand. Compare policies using matched precollected profiles.
- Current runtime limitation: disabling reusable setup also disables coordinator
  profile loading. Node profiles alone cannot remove the supervisor's unprofiled
  job startup restriction. Thus fully profiled cold admission comparisons require
  separating profile loading from setup reuse in a later runtime change. The runner
  rejects these by default. `--allow-unprofiled` is only an explicitly labeled smoke
  test, not the controlled cold comparison. Warm snapshots must include coordinator
  `resource_profile.json` files as well as the node resource_profiles snapshot.
- Simulation in motivation estimates packing opportunity. Real CPU/storage
  contention must be measured here; do not use simulated times as system results.
- The local launcher supports one node. Scale-out requires an existing deployment.
- The load generator's client submission can itself be slow. Record submission lag;
  overloaded client injection must not be mistaken for a stable offered load.

Results and logs under `evaluation/results/` are intentionally untracked.

Use `--verify` for correctness-checked runs and record whether verification is
included in timing. The default follows the runtime's verification-disabled
configuration; a completed request alone is not evidence that verification ran.

## Validation of this harness

```bash
python -m unittest discover -s evaluation -p test_evaluation.py -v
python -m unittest discover -s tests -p test_whole_job_admission.py -v
```

Harness tests cover reproducible arrivals, consistent baseline settings, failure
accounting, dry-run behavior, and a mocked coordinator RPC lifecycle. Admission
tests cover real competing processes with controlled memory readings. Neither
test suite substitutes for an end-to-end proof smoke test on the evaluation host.
