# Run the evaluation step by step

**Yes: collect profiles before running the admission comparisons.** This file
is the executable walkthrough; [00_setup/README.md](00_setup/README.md) explains
the measurement protocol. You do not execute Markdown. Execute `run_all.sh`.

`run_all.sh` contains a long, explicit list of preparation and experiment calls.
Comment out any line you do not want. It defaults to printing a plan. Adding
`--execute` launches real proofs and can take days for the full matrix.

## 1. Use the evaluation server and your working backend environment

Run from your existing checkout on the dual-EPYC server, not this chat sandbox.
Use the virtual environment/custom EZKL wheel that already proves your models.
Do not replace that wheel with a generic pip release just to run this checklist.

```bash
git switch sacdad27
git pull --ff-only
tmux new -s zkinfer-eval
# Activate your existing zkInfer virtual environment here.
export PYTHON=python
export EVAL_ROOT="$PWD/evaluation/results/paper-v1"
```

Choose and fix the actual thread settings you want to evaluate. For example,
the following chooses eight Rayon threads per prover (a protocol choice, not a
claim that eight is optimal). Keep the settings identical for profiles and runs:

```bash
export RAYON_NUM_THREADS=8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PROFILE_REPEATS=5
export CACHE_GIB=500
export SAMPLING=0.1
export TIMEOUT=172800
```

Record your custom EZKL wheel/commit separately. `preflight.py` records Python,
package versions, model/input hashes, CPU/RAM and thread environment. It rejects
reuse of a campaign directory after those recorded properties change. The
campaign also checks runtime/config source hashes. Changing the Bash checklist
to comment out lines is fine; changing backend, inputs, thread settings, proving
configuration or runtime source calls for a new campaign root and new profiles.

## 2. Preview the checklist; start with one workload

```bash
SECTION=profiles ONLY='mnist_gan-g1-*' bash evaluation/run_all.sh --plan
SECTION=serving ONLY='burst-mnist_gan-warm-*-n1-r1' bash evaluation/run_all.sh --plan
```

No proofs or output directories are created in plan mode. The local script is
for Linux/GNU Bash and GNU `cp` (uses copy-on-write when available). Plans assume
755 GiB target RAM (`PLAN_HOST_GIB` overrides this); execution detects actual RAM.

## 3. Collect MNIST profiles and setup artifacts

```bash
SECTION=profiles ONLY='mnist_gan-g1-*' bash evaluation/run_all.sh --execute
```

For each setup condition this runs **one request at a time, with one active proof
job per node**, admission disabled, and nearly full host memory (8 GiB reserved):

1. **Cold:** run an unmeasured primer to fetch required SRS/initialize the backend;
   discard its histories. Then run five independent requests, forcing setup
   regeneration each time and accumulating only their resource observations.
2. **Warm:** first run an unmeasured bootstrap that generates reusable artifacts.
   Discard both of its histories, since it is a cache-miss execution. Then run five
   independent requests that reuse the artifacts. Their fresh node caches load
   artifacts from local files. Require proving-key cache-hit metrics for every job.
3. Save independent coordinator and node history snapshots. Five requests cover
   every proof job in the model, not just the three selected repeatability jobs.
   These profiling requests are excluded from serving results.

Both histories matter: the coordinator uses job-level estimates for placement;
the node uses stage estimates for admission. A node snapshot alone is insufficient.
Warm/cold describe artifact availability, not OS page-cache state. These scripts
do not flush OS caches. Preparation uses the workload's configured input fixture;
it is not the separate varying-input repeatability microbenchmark.

Files are under:

```text
paper-v1/profiles/mnist_gan-g1-cold/DONE.json
paper-v1/profiles/mnist_gan-g1-cold/attempt-001/snapshot/node/
paper-v1/profiles/mnist_gan-g1-cold/attempt-001/snapshot/coordinator/
paper-v1/profiles/mnist_gan-g1-warm/attempt-001/snapshot/
paper-v1/profiles/mnist_gan-g1-warm/attempt-001/setup/
```

`DONE.json` identifies the successful attempt. Do not rename warm histories to
cold or use bootstrap histories as warm observations. Preparation fails if any
request fails, if reports/profiles are absent, or if warm key reuse is not observed.
Inspect actual stage boundaries in those reports; preparation does not split a
backend call into independently controllable stages.

## 4. Run the first real comparison and inspect it

```bash
SECTION=serving ONLY='burst-mnist_gan-warm-*-n1-r1' bash evaluation/run_all.sh --execute
```

This runs fixed-8, whole-job and stage admission, one request each, with a
256 GiB accounting budget. The script calculates reserve = host RAM - budget;
do not hardcode 755 minus budget if the host reports a different total.
Every run enables runtime proof verification and keeps its cost in the completion
boundary. It has fresh coordinator/node state and independent initial profiles.
Warm artifacts are copied into trial-owned storage before timing; job-level local
cache restoration remains inside the measured pipeline.

Inspect `runs/<ID>/attempt-001/run/`:

- `plan.json`: exact commands, settings, commit and thread environment.
- `run_status.json` and `traffic/summary.json`: all requested completions, failures,
  median latency and batch makespan. A success marker requires all requests complete.
- `traffic/requests.jsonl`: individual timings, status, submission lag and workload.
- `requests/**/reports/job_report.csv`: setup hits, actual memory, stage sampling,
  predicted memory and admission waiting. Confirm the expected 12 MNIST jobs.
- `requests/**/reports/request_report.csv`: request timings and verification time.
- `coordinator.log`, `node.log`, `traffic.log`: failures and process diagnostics.

Check that the backend actually exposes the stages claimed by the paper. For
correctness claims, also check the verifier/manifest outputs; verification enabled
alone does not imply every desired cross-partition semantic check is configured.

## 5. Collect the remaining histories, then run the chosen blocks

```bash
SECTION=profiles bash evaluation/run_all.sh --execute
SECTION=serving ONLY='burst-mnist_gan-warm-*' bash evaluation/run_all.sh --execute
```

Completed MNIST preparation automatically skips. The other models get separate
snapshots; mixed-workload snapshots combine the three prepared models without
running a new training workload. After checking the first workload, run:

```bash
SECTION=serving bash evaluation/run_all.sh --execute
SECTION=memory bash evaluation/run_all.sh --execute
SECTION=scheduling bash evaluation/run_all.sh --execute
SECTION=arrival bash evaluation/run_all.sh --execute
SECTION=granularity bash evaluation/run_all.sh --execute
# Optional expensive reruns; preserve OOM results rather than hiding them:
SECTION=decomposition bash evaluation/run_all.sh --execute
```

Or run the whole editable checklist from top to bottom:

```bash
bash evaluation/run_all.sh --execute
```

There are hundreds of measured cells in the exhaustive list. It is a checklist,
not a recommendation to run everything before looking at the first result.
Use `SECTION`/`ONLY`, or comment out lines. Section names can be comma-separated.

| Block | Runs | Paper use |
|---|---|---|
| profiles | Isolated cold/warm histories for all primary jobs | Preparation only |
| serving | 1/4/8/16 requests; fixed-8/whole-job/stage; 3 trials; cold/warm; individual and mixed models | Main burst results |
| memory | 64/128 GiB; N=16; whole-job/stage; cold/warm | Admission mechanism; reuse 256 GiB serving runs |
| scheduling | FIFO/LPT under stage admission; fixed-1/2/4/16; N=16 warm | Scheduling and fixed-concurrency sensitivity; reuse stage/fixed-8 serving runs |
| arrival | Independent whole-job pilot; 64 Poisson arrivals at 0.25/0.5/0.75/1/1.25 times pilot rate | Offered-load performance |
| granularity | Warm profiles for g=2/4/8, then stage N=1/16 | Partition sensitivity; g=1 comes from serving |
| decomposition | Monolithic profiles, then mono/partitioned-sequential cold/warm N=1 | Optional matched refresh of legacy results |

Reuse results use stage N=1/8 cold/warm serving runs. Ordinary component timing
uses their reports; no duplicate experiment is necessary. For stochastic traffic,
each repetition shares seed/model sequence across policies. Baseline order rotates
between trials rather than always executing the same policy first.

Poisson rates are requests/second, derived from an independent whole-job batch
pilot, and are identical across policies. This pilot rate is only a reference:
inspect backlog and arrival-window completions before claiming sustainable capacity.
The existing summary rate includes the final drain. Long schedules can exceed
TIMEOUT: raise it for the campaign before starting, or use a new ID/root when
changing completed-cell settings. Do not silently count unsubmitted requests as
successful throughput.

## 6. Detach, restart, and skip failures

Detach tmux with **Ctrl-b then d**; reconnect with:

```bash
tmux attach -t zkinfer-eval
```

Stop the active local run with Ctrl-C and wait for cleanup. Run the same command
again: successful task IDs skip, while interrupted/failed tasks get `attempt-002`,
etc. Existing attempts are never overwritten. Preparation retries the whole
preparation task rather than adopting partially collected histories. If a cell
is genuinely infeasible/OOM, retain its failure record, comment out that line,
and resume. Do not create DONE.json manually or summarize successful retries
while suppressing failed attempts.

Default `DISCARD_COPIES=1` removes only a successful measured run's copied setup
directory and node cache, and successful preparation runs' disposable node caches.
It retains original setup seeds, updated histories,
proof outputs, raw request reports and logs. Set `DISCARD_COPIES=0` to retain
those copies too. Copy-on-write is used where supported; otherwise copying large
keys needs substantial disk space. Raw proofs and failed attempts also accumulate.
No global caches are deleted. Do not run independent campaigns simultaneously on
the same experimental host; the lock protects only the selected campaign root.

## 7. AWS scale-out uses physical deployments, not the local runner

Preview the full editable cloud list:

```bash
bash evaluation/run_cluster.sh --plan
```

`run_cluster.sh` contains whole-job/stage, 1/2/4/8/10 physical nodes, N=1/16,
three models and three repetitions. It drives `traffic.py` only after an explicit
deployment hook has prepared the requested cluster. We cannot fill in your SSH
hosts, S3 bucket/prefix, remote environment or per-worker profile paths from the
information available here. It intentionally refuses --execute without hooks.

Hook interface (both executables receive these five arguments):

```text
BASELINE NODE_COUNT WORKLOAD TRIAL OUTPUT_DIRECTORY
```

`CLUSTER_PREPARE` must start/reset exactly that many distinct physical workers,
use a unique node_id on each, configure the requested stage/whole-job mode and
common per-worker budget/thread settings, initialize matching **AWS-collected**
profiles and warm S3 setup state, enable verification, and wait for registration.
It must write `OUTPUT_DIRECTORY/deployment.json` recording actual hosts/instance
IDs, commit/backend, overrides, budget, artifact/profile sources, coordinator
placement and verification setting. EPYC histories are not assumed to match AWS.
`CLUSTER_STOP` must stop only the runtime processes for that trial and collect
remote reports. It need not terminate EC2 instances. It is called on exit, even
when preparation/traffic fails. Supply absolute executable paths:

```bash
export CLUSTER_TARGET='YOUR_COORDINATOR_HOST:50051'
export CLUSTER_PREPARE='/absolute/path/to/your/prepare_cluster.sh'
export CLUSTER_STOP='/absolute/path/to/your/stop_and_collect_cluster.sh'
export EVAL_ROOT="$PWD/evaluation/results/aws-v1"
bash evaluation/run_cluster.sh --execute
```

Cloud DONE markers are simple completion checks, not a deployment fingerprint.
Use a new EVAL_ROOT when changing hooks, infrastructure, profiles or configuration.
The prepare hook is responsible for validating the actual node count and readiness.

## What is not silently automated

- TinyBERT: no workload YAML on this branch. Add its verified ONNX/input paths first.
  NanoGPT 10L has example commented checklist entries.
- Worker failure injection/recovery: needs the real cloud deployment and trigger.
- Exact scheduling/admission-check CPU overhead: requires instrumentation beyond
  the existing component/admission-wait reports; waiting time is not check overhead.
- Controlled 0.9x/1.1x estimate perturbation: needs an explicit estimator test hook.
- Second-backend results: the current runner is EZKL-specific.

These remain optional experiments, not fabricated results. No real proof or AWS
workload was executed while authoring this automation; run and inspect step 4
before expanding. Tests cover orchestration and state isolation, not backend speed.

```bash
python -m unittest discover -s evaluation -p 'test_*.py' -v
bash -n evaluation/run_all.sh evaluation/run_cluster.sh
```
