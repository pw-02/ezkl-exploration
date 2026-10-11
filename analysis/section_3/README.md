# Section 3: zkML Proof Execution Characterization

This directory contains the experiment plan and analysis used to motivate the
zkInfer runtime design. Section 3 should establish workload properties and
resource-management opportunities. It should **not** use the final admission
policy's performance as evidence; admission ON/OFF comparisons belong in the
system evaluation.

## Questions Section 3 must answer

1. **What does a proof job spend time and memory on?**
   Measure calibration, compilation, SRS preparation, witness generation, key
   generation, proving, and the instrumented Halo2 proving subphases.
2. **How much does resource demand vary during a job?**
   Compare whole-job peak memory with EZKL-stage and Halo2-subphase memory.
3. **Is stage demand repeatable?**
   Re-run the same setup key with different inputs and quantify variation in
   memory, CPU, and duration.
4. **How much capacity does whole-job peak reservation leave unused?**
   Replay isolated traces under whole-job, EZKL-stage, and Halo2-subphase
   reservation models.
5. **Does concurrent request execution expose useful packing opportunities?**
   Measure how many independent proof jobs and phases coexist under sustained
   multi-request load.
6. **How important is reusable setup state?**
   Compare cold and cached-setup executions and report the setup work removed by
   cache hits.

## Experimental discipline

Use the same machine configuration for comparisons. Record the exact zkInfer,
EZKL, and Halo2 commits, CPU model, memory capacity, OS, Python version, sampling
interval, partitioning configuration, and coordinator scheduling policy with the
results.

For isolated characterization runs use:

```yaml
execution:
  partitioning:
    mode: fixed_size
    operators_per_partition: 1
node:
  max_active_jobs: 1
  resource_monitor_interval_sec: 0.25
  phase_admission:
    enabled: false
```

Admission is disabled because Section 3 is measuring the workload rather than
measuring zkInfer's control policy. `max_active_jobs: 1` also prevents unrelated
proof jobs from overlapping during isolated measurements. Avoid unrelated
compute or memory-intensive processes on the machine.

## Experiment matrix

### E1. Isolated phase/resource shape

For both MobileNetV2 and NanoGPT-4L select partitions representing approximately
small, median, p90, and largest observed proof jobs. Use a systematic rule based
on a prior complete run; do not select visually convenient examples.

For each selected partition collect:

- one cold execution with no reusable setup artifacts;
- one cached-setup execution;
- `resource_usage.csv`;
- `stage_transitions.csv`;
- `halo2_stage_transitions.csv` (end rows now include `duration_us` and `duration_s`);
- `halo2_prover_cpu.csv`;
- `admission_profile_before.json`;
- the request `job_report.csv` and `request_report.csv`.

`job_report.csv` is the canonical derived per-job table. The old duplicate
`perf_metrics.csv` is no longer produced. Raw traces remain under the per-job
artifact directory.

Primary outputs: per-stage duration, peak RSS, average CPU, whole-job peak, and
memory-over-time traces.

### E2. Repeatability

For 10--20 setup keys per model, execute at least 10 runs with different inputs
that preserve the same circuit/setup identity. Report memory, CPU, and duration
variation for each stage.

For cached-setup steady-state runs, witness generation and proving are naturally
repeated. To measure key-generation repeatability, use a fresh setup-cache
namespace for each repetition while preserving the logical setup key in the
experiment metadata.

Primary statistics:

- coefficient of variation (CV);
- median, p90, and maximum;
- rolling prediction error from runs 1..k to run k+1;
- underprediction rate and p95 underprediction magnitude.

### E3. Reservation opportunity

Replay the isolated process-memory traces using three accounting granularities:

1. whole-job peak throughout the job;
2. current EZKL stage peak;
3. current Halo2 subphase peak while inside proving.

Report:

- average reserved memory;
- reserved-but-unused GB-seconds;
- normalized reservation waste;
- reduction in reservation waste from finer granularity;
- pairwise overlap opportunities under the target node memory budget.

This experiment characterizes opportunity only. It does not claim that a
particular runtime policy achieves the replayed optimum.

### E4. Multi-request execution shape

Submit 1, 4, 8, 16, and 32 same-model requests with different inputs. Repeat for
both models if runtime permits. Keep admission disabled for the Section 3
characterization and remove the whole-job cap (`max_active_jobs: null`) when the
purpose of the experiment is to observe natural phase overlap.

Report:

- active proof jobs over time;
- phase mixture over time;
- peak process/system memory;
- fraction of time with two or more independently runnable jobs;
- cold-to-cached setup transition across the batch.

The system evaluation can later repeat these loads with admission disabled and
enabled.

## Analysis command

After collecting runs:

```bash
python analysis/section_3/analyze_characterization.py \
  --runs-dir experiments/runs \
  --output-dir analysis/section_3/generated \
  --memory-budget-gb 760
```

Use the actual usable node memory budget for `--memory-budget-gb`, not simply the
physical DIMM capacity. The script discovers per-job artifacts recursively and
writes tables, figures, a run-validation report, and a JSON summary.

## Expected tables

The analysis script produces:

- `job_characterization.csv`: one row per proof job;
- `stage_characterization.csv`: one row per job/stage;
- `repeatability.csv`: variation grouped by setup key and stage;
- `reservation_opportunity.csv`: whole-job vs stage/subphase accounting;
- `prediction_accuracy.csv`: historical prediction vs measured stage demand;
- `pairwise_packing.csv`: pairs rejected by whole-job peaks but compatible in at
  least one finer-grained stage combination;
- `cold_warm_summary.csv`: setup and proof behavior split by cache state;
- `artifact_validation.csv`: missing/available profiling artifacts for every job.

## Expected figures

The script produces publication-oriented source figures (PNG; regenerate as PDF
for the final paper if required):

1. `fig_ezkl_stage_memory.png` -- stage peak-memory distribution;
2. `fig_ezkl_stage_duration.png` -- stage-duration distribution;
3. `fig_halo2_subphase_memory.png` -- peak memory inside proving;
4. `fig_reservation_waste.png` -- reservation waste at the three granularities;
5. `fig_repeatability_memory.png` -- per-stage memory CV across repeated setup
   keys;
6. `fig_prediction_accuracy.png` -- predicted vs measured stage memory when
   historical predictions are present;
7. `fig_node_activity.png` -- active jobs/reserved memory over time for the
   highest-concurrency observed run.

Do not put every figure in the paper. The likely core Section 3 set is the
existing granularity table/figure plus stage memory, repeatability, and
reservation-waste results. Halo2 subphase data should be promoted to the main
paper only if it exposes materially more resource variation than EZKL-level
stages.

## Interpretation rules

- Do not call a stage lightweight without measured evidence.
- Keep cold and cached-setup executions separate.
- Do not infer memory peaks from Halo2 timing logs alone; use aligned RSS traces.
- Report when proving remains close to whole-job peak for most of a cached run;
  that is evidence that finer admission has limited opportunity in that regime.
- Treat CPU results as secondary unless they are stable enough to support a
  clear scheduling claim. Memory is the primary safety and packing resource.
- If Halo2 subphase granularity reduces reservation waste only marginally over
  EZKL-stage granularity, stop at EZKL stages in the final runtime design.
