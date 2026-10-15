# Targeted evaluation: setup once, profile once, then measure

The default `run_all.sh` is now a **targeted first pass**, not the exhaustive
campaign. The old large list is retained as `run_full.sh`.

## What changed

- One **shared setup-only bootstrap per model/partitioning**: calibration, circuit
  compilation, SRS preparation and key generation. No witness or proof generation.
- Bootstrap runs in separate per-partition processes and retains its artifacts.
  Completed partition setup is checkpointed, so an interrupted bootstrap can resume.
- Measured warm trials now **reference the bootstrap setup directly** instead of
  creating a second physical copy of every compiled circuit and proving key.
  Node-local cache state, resource histories, logs, request data and proof outputs
  remain isolated per attempt.
- **One full profiling request per selected setup condition** by default
  (`PROFILE_REPEATS=1`). A request covers every partition, not one selected job.
- No cold primer and no second warm bootstrap. Cold and warm profiling share the
  bootstrap artifacts/SRS; cold execution deliberately regenerates its own setup.
- Default measured runs: **warm setup**, N=1 and N=4, fixed-8/whole-job/stage,
  one trial, for MNIST GAN, NanoGPT (4L), then MobileNetV2.
- Cold comparisons, memory sensitivity, fixed-k sweeps and repetitions are
  commented optional lines. Poisson/mixed/granularity sweeps are in run_full.sh.

This gives 18 measured cells plus three setup-only bootstraps and three isolated
warm profiling requests. It is a priority list, not a promise that all work fits
within one day. Actual proof time depends on the backend and thread settings.
The script has a 23-hour execution budget per invocation by default; a cutoff
can interrupt the active cell, which is retained as an incomplete attempt.

## 1. Update and use a fresh campaign directory

Stop the old script with Ctrl-C and wait for its child processes to finish before
pulling changes. Do not run both campaigns concurrently on the experimental host.
Keep old outputs; this update does not delete or relabel the old cold-primer runs.

From your repository root, in the environment with your working custom EZKL wheel:

```bash
git switch sacdad27
git pull --ff-only
tmux new -s zkinfer-eval
# Activate your existing environment if needed.
export PYTHON=python
export EVAL_ROOT="$PWD/evaluation/results/paper-day-v2"
export PROFILE_REPEATS=1
export PYTHONPATH=/home/pwatters/project/zkinfer:$PYTHONPATH
```

The explicit export matters if your shell still has PROFILE_REPEATS=5 from the
old instructions. Use the SAME thread configuration you intend to evaluate for
setup/profiling/measurement. The script still requires RAYON_NUM_THREADS explicitly;
eight below is an example choice, not an optimum:

```bash
export RAYON_NUM_THREADS=128
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

If you already chose different settings, keep those instead. Do not change them
between profiles and measured runs. Preflight records versions, CPU/RAM,
model/input hashes and these environment variables; record custom wheel identity
separately. Source/config hashes also protect against mixing incompatible runs.

## 2. Preview, then run

```bash
bash evaluation/run_all.sh --plan
bash evaluation/run_all.sh --execute
```

The script finishes each model's profiling and six measured cells before moving
to the next model, so useful results arrive early. MobileNetV2 is last because
earlier measurements were slower, not because of its measured improvement.
Report all attempted configurations, including failures and time-budget cutoffs.

To start with MNIST only:

```bash
ONLY='*mnist_gan*' bash evaluation/run_all.sh --execute
```

Or run preparation and the first comparison separately:

```bash
SECTION=profiles ONLY='mnist_gan-g1-warm' bash evaluation/run_all.sh --execute
SECTION=serving ONLY='burst-mnist_gan-warm-*-n1-r1' bash evaluation/run_all.sh --execute
```

You can comment out any `prep` or `cell` line in run_all.sh. Completed cells skip
automatically. Failed/interrupted cells get a new attempt directory; existing
reports are retained. Bootstrap resumes completed partitions separately.

## 3. Where the artifacts and profiles live

For MNIST, relative to EVAL_ROOT:

```text
bootstrap/mnist_gan-g1/artifacts/setup/<setup_key>/
    model.onnx
    settings.json
    network.compiled
    pk.json
    vk.json

profiles/mnist_gan-g1-warm/attempt-001/
    setup_source.json
    observation-01/
    snapshot/node/
    snapshot/coordinator/

runs/<ID>/attempt-001/
    setup_source.json        # points to the shared bootstrap setup
    run/
        cache/               # fresh node-local cache for this attempt
        node-state/
        coordinator-profiles/
        requests/
        traffic/
```

This setup-only path retains **uncompressed pk.json**, which the worker's
existing restore path accepts. It avoids the additional bootstrap compression
pass. `setup_source.json` records the shared bootstrap location; measured warm
trials do not create `trial-setup` copies anymore.

The distinction is deliberate:

- **Bootstrap setup artifacts** are immutable reusable inputs for matching warm
  trials and may be shared across those trials.
- **Node cache state, resource histories, request data, logs and proof outputs**
  belong to one attempt and remain separate.

A warm worker still restores the setup into its own fresh node cache. This means
local restore/decompression cost remains part of the measured run, while the
campaign no longer spends another 288+ GiB merely duplicating the shared setup
before the run starts.

Bootstrap creates **no coordinator/node resource profiles**. The one full warm
profiling request creates both. Proving must execute at least once to observe
proving-stage memory; setup-only execution cannot supply that information.
Warm profiling checks proving-key reuse and positive measured memory for every
setup key. Node and coordinator histories start empty, independent of bootstrap.

For optional cold profiling, the SAME bootstrap is retained for the campaign,
but its directory is not passed as reusable setup to cold jobs. Those jobs
regenerate setup and prove once to collect cold history. Their generated state
lives in that run's own node cache/request output and is not used to replace the
retained bootstrap.

## 4. Inspect the first measured run

Under `runs/<ID>/attempt-001/run/`:

- `plan.json`: commands, overrides, commit, thread settings.
- `run_status.json` and `traffic/summary.json`: completion/failure counts and timings.
- `traffic/requests.jsonl`: individual request outcomes, latency and submission lag.
- `requests/**/reports/job_report.csv`: setup hits, stage memory and admission waits.
- `requests/**/reports/request_report.csv`: request and verification timings.
- `node.log`, `coordinator.log`, `traffic.log`: failures and diagnostics.

Runtime verification remains enabled for measured/profile runs. Bootstrap
generates no proofs to verify; correctness is checked on the subsequent requests.
Check verifier manifests and intended partition-boundary semantics as well.

Warm trials start from the same shared bootstrap setup but with a fresh worker
cache. Local restoration remains part of measured execution. They are not a
prefilled-worker-cache experiment. Cold/warm do not describe OS page-cache state.
All local policies receive the same accounting budget; hardware CPU capacity stays
fixed. This is runtime accounting, not a physical/cgroup RAM limit.

## 5. Defaults and time limits

| Setting | Targeted default | Meaning |
|---|---:|---|
| PROFILE_REPEATS | 1 | Full profiling requests per workload/setup |
| CACHE_GIB | 500 | Worker disk-cache budget, not RAM |
| SAMPLING | 0.1 | Memory/CPU sample interval in seconds |
| TIMEOUT | 21600 | Six-hour limit for an individual run/bootstrap |
| CAMPAIGN_SECONDS | 82800 | 23-hour wall-clock budget per script invocation |
| DISCARD_COPIES | 1 | Remove successful attempts' disposable worker caches |
| RAYON_NUM_THREADS | Explicitly required | Backend thread setting, not job concurrency |

Previously exported values override shell defaults. To select the time limits
explicitly:

```bash
export TIMEOUT=21600
export CAMPAIGN_SECONDS=82800
```

The campaign deadline sends an interrupt and allows cleanup. A deadline-aborted
run is incomplete; do not classify it as an OOM or silently drop it from reports.
Resuming starts a new wall-clock allowance and skips successful cells. Use a new
ID/root when changing the settings of completed cells. Set CAMPAIGN_SECONDS=0
only if you intentionally want no overall limit.

Default cleanup retains bootstrap seeds, profile snapshots, raw reports, proof
outputs, plans and logs. It removes only successful attempts' disposable worker
caches. It never deletes the bootstrap setup referenced by `setup_source.json`.
Bootstrap artifacts and retained failed attempts can still consume substantial
disk space.

## 6. Extend only if time remains

Uncomment optional cold/profile and N=4 memory-sensitivity lines after completing
the warm comparisons. Repetitions r2/r3 are more useful for quantifying variation
than claiming precision from one trial. Use the exhaustive list explicitly:

```bash
SECTION=profiles,serving bash evaluation/run_full.sh --plan
```

run_full.sh still contains the larger three-trial protocol, now using one initial
profiling request by default and the shared setup-only bootstrap. It is NOT the
one-day default. Keep a consistent EVAL_ROOT if intentionally reusing compatible
profiles; default full/day roots are separate.

AWS scale-out remains in run_cluster.sh and requires your physical deployment
hooks. TinyBERT still needs its workload YAML. Worker failure injection,
controlled estimate perturbation and fine-grained scheduler/check overhead need
their dedicated instrumentation; this update does not pretend those experiments
have run.

## What to say in the paper

One initial profile is a sparse-history admission experiment, not evidence of
repeatability. Describe it as one prior isolated execution per job and condition.
The first-pass measured cells have ONE trial: do not show trial-range error bars
or say they are three-run averages. Repeat selected cells later if possible.
The earlier repeated-input memory microbenchmark remains a separate experiment.

## Validation

```bash
python -m unittest discover -s evaluation -p 'test_*.py' -v
bash -n evaluation/run_all.sh evaluation/run_full.sh evaluation/run_cluster.sh
```

Tests check setup-only stage selection, shared cold/warm bootstrap, profile
isolation, checkpoint/resume and failure accounting. No real proof benchmark or
AWS workload was executed while authoring this update.
