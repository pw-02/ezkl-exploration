# Section 3: zkML Proof Execution Characterization

Section 3 should establish the execution properties, opportunities, and challenges that motivate the zkInfer design. It should not present the final runtime policy as the answer before Section 4 introduces the system.

The intended argument is:

1. model partitioning exposes independent proof jobs, but those jobs are heterogeneous;
2. memory demand changes substantially over the lifetime of a proof job;
3. repeated executions of the same circuit may make that demand predictable;
4. setup work is expensive and reusable, which makes placement stateful;
5. concurrent inference requests expose enough independent work for these properties to matter in a serving system.

Section 4 can then introduce the corresponding mechanisms: global job placement, historical resource profiles, node-local stage admission, and setup-aware placement. Admission ON/OFF throughput comparisons and scheduler ablations belong in the system evaluation.

## Questions Section 3 must answer

### 1. What does partitioning expose?

Measure how partitioning changes the number of proof jobs, aggregate proving cost, longest-job duration, and peak process memory. Show the distribution across partitions rather than only averages.

The important question is whether fine-grained partitioning creates useful parallel work while also producing heterogeneous jobs that should not be treated as interchangeable.

### 2. How does resource demand change during a proof job?

Measure calibration, compilation, SRS preparation, witness generation, key generation, proving, and the instrumented Halo2 proving stages.

For memory accounting use the following terminology throughout the analysis:

- **Job peak**: assume an active proof job occupies its measured maximum memory for its full execution.
- **Current stage**: use the measured peak of the EZKL stage that is currently executing.
- **Fine-grained proving**: use the current EZKL-stage peak outside proving and the current instrumented Halo2 stage peak inside proving.

These are counterfactual memory-demand models, not runtime policies. They quantify how much information is lost when a scheduler reasons only about a job-level maximum.

### 3. Is resource demand repeatable?

Re-run the same setup key with different inputs and quantify variation in memory, CPU, and duration. The most important measurements are peak-memory CV, prediction error, underprediction frequency, and underprediction magnitude.

The design question is whether prior executions provide sufficiently stable estimates for later proofs of the same circuit.

### 4. How important is reusable setup state?

Compare cold and cached-setup executions. Report the time and storage associated with settings, compilation, key generation, setup restoration, and proving.

The design question is whether a scheduler should consider where reusable setup state already exists rather than treating every node as equivalent.

### 5. What changes when multiple inference requests coexist?

Multi-request execution should be a first-class part of the characterization because proof serving is not only a single-request batch problem. Submit multiple independent requests and observe the mixture of jobs and execution stages available at the same time.

Do not use Section 3 to claim that zkInfer's final admission policy improves throughput. Instead establish whether concurrent requests expose enough heterogeneous work to create a meaningful packing and scheduling problem.

For interactive workloads, report request latency and, where useful, SLO miss rate. For offline/batch workloads, report makespan, throughput, and utilization. The evaluation can later show how zkInfer trades these objectives under different policies.

## Experimental discipline

Use the same machine configuration for comparisons. Record the exact zkInfer, EZKL, and Halo2 commits, CPU model, memory capacity, OS, Python version, sampling interval, partitioning configuration, and coordinator scheduling policy with the results.

For isolated characterization runs use:

```yaml
execution:
  partitioning_enabled: true
  partition_size: 1
node:
  max_concurrent_jobs: 1
  resource_sampling_interval_sec: 0.25
  stage_admission:
    enabled: false
```

Admission is disabled because these runs measure the workload rather than zkInfer's control policy. `max_concurrent_jobs: 1` prevents unrelated jobs from overlapping and contaminating per-job resource traces.

## Recommended experiment order

### E1. Fine-grained job population

Run MobileNetV2 and NanoGPT-4L with `partition_size: 1` and one concurrent proof job at a time.

Report:

- number of proof jobs;
- aggregate proving time;
- longest proof job;
- per-job runtime distribution;
- per-job peak process-memory distribution;
- setup/key sizes where relevant.

Use these complete populations to select representative small, median, p90, and largest jobs systematically. Do not hand-pick visually interesting partitions.

### E2. Isolated execution shape

For each representative partition collect one cold and one cached-setup execution.

Retain:

- `resource_usage.csv`;
- `stage_transitions.csv`;
- `halo2_stage_transitions.csv`;
- `halo2_prover_cpu.csv`;
- `admission_profile_before.json`;
- `job_report.csv` and `request_report.csv`.

`job_report.csv` is the canonical derived per-job table. Raw timing and resource traces remain under the per-job artifact directory.

Primary outputs:

- stage duration;
- stage peak process RSS;
- average CPU;
- fraction of runtime spent near the job peak;
- differences between key generation and proving;
- variation among Halo2 proving stages.

A useful negative result is possible: if proving stays close to the job peak for most of execution, finer-grained memory accounting has limited value for that workload.

### E3. Repeatability and prediction

For 10--20 setup keys per model, execute at least 10 runs with different inputs that preserve the same circuit/setup identity. A smaller pilot such as 5 setup keys per model is reasonable before committing to the full campaign.

Report:

- coefficient of variation (CV);
- median, p90, and maximum;
- rolling prediction error from runs 1..k to run k+1;
- underprediction rate;
- p95 underprediction magnitude.

For cached-setup steady-state runs, witness generation and proving are naturally repeated. To measure key-generation repeatability, use a fresh setup-cache namespace for each repetition while preserving the logical setup identity in experiment metadata.

### E4. Memory-packing opportunity

Replay the isolated traces using the three memory-demand models:

1. **Job peak**;
2. **Current stage**;
3. **Fine-grained proving**.

Report:

- time-weighted accounted memory;
- unused capacity in GB-seconds;
- normalized unused-capacity fraction;
- achievable overlap under explicit node-memory budgets;
- the minimum makespan for small workloads where the exact oracle completes.

This is an opportunity analysis. It does not claim that the production scheduler attains the oracle schedule.

### E5. Concurrent-request execution shape

Submit 1, 4, 8, 16, and, only if useful, 32 requests with different inputs. Repeat for both models where runtime permits.

For workload characterization keep admission disabled and remove the count cap:

```yaml
node:
  max_concurrent_jobs: null
  stage_admission:
    enabled: false
```

Report:

- active proof jobs over time;
- execution-stage mixture over time;
- peak process/system memory;
- fraction of time with multiple independently runnable jobs;
- cold-to-cached setup transition across the request stream.

The system evaluation can later replay comparable request streams with alternative runtime policies.

## Characterization analysis

After collecting runs:

```bash
python analysis/section_3/analyze_characterization.py \
  --runs-dir experiments/runs \
  --output-dir analysis/section_3/generated \
  --memory-budget-gb 760
```

Use the actual usable node-memory budget, not simply physical DIMM capacity.

The characterization script produces per-job and per-stage tables, repeatability statistics, prediction accuracy, cold/cached comparisons, artifact validation, and memory-opportunity summaries.

## Trace-driven packing and scheduling simulator

`simulate_scheduling.py` replays measured traces without rerunning proofs. It serves two different purposes.

### A. Find the best packing opportunity

For a small all-at-once workload, the `--oracle` mode performs an exact search for the minimum makespan subject to the measured stage ordering and the configured memory budget. Exact search is intentionally bounded by job count, stage count, and wall-clock search time because the problem is combinatorial.

### B. Compare scheduling heuristics

The same simulator can compare work-conserving policies such as FIFO, LPT, SPT, memory-fit, and request-fair ordering. These are offline simulation heuristics and are deliberately separate from claims about the production coordinator implementation.

The simulator supports `--max-concurrent-jobs 1` for a sequential-node baseline and zero for no count-based limit.

## Interpretation rules

- Do not call a stage lightweight without measured evidence.
- Keep cold and cached-setup executions separate.
- Do not infer memory peaks from timing logs alone; align them with process-memory traces.
- Report when proving remains close to the job peak for most of a cached run; that is evidence that finer-grained accounting has limited opportunity.
- Treat CPU as characterization data rather than an admission constraint. Memory is the primary safety and packing resource.
- If finer-grained proving captures little additional opportunity beyond current-stage accounting, stop at EZKL-stage admission in the final design.
- Separate offline objectives (makespan/throughput) from serving objectives (request latency/SLO attainment). A policy that is good for one is not automatically good for the other.
