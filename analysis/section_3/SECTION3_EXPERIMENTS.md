# Section 3: Characterizing zkML Proof Execution

This directory contains the experiment plan and analysis specification for Section 3. Section 3 reports workload properties and *potential* resource-packing opportunities; actual admission-on/off performance belongs in evaluation.

## Experiment matrix

| ID | Question | Runs | Required outputs |
| --- | --- | --- | --- |
| G | How does granularity change proof cost? | MobileNetV2 and NanoGPT4L; operator groups 1,2,4,8,full and profile-guided grouping; repeat runs | job counts, aggregate proving time, longest job, maximum process memory |
| H | How heterogeneous are partitions? | All one-operator partitions, isolated; cold and warm clearly labelled | per-job stage durations, peak process memory, quantiles and sorted distributions |
| P | How do resources vary by stage? | Isolated representative smallest, median, p90 and largest partitions by baseline peak memory; both models; cold and warm | stage duration, max RSS, CPU, fractions of runtime |
| R | Are per-stage profiles repeatable? | 10–20 setup keys, >=10 repetitions per key with varied inputs but identical circuit identity | CV, p90/max, leave-one-run-forward prediction, underprediction frequency and magnitude |
| O | How much memory could stage-aware reservation save? | Replay isolated stage traces, comparing constant whole-job peak with stage-specific peak | reserved GB-seconds, unused GB-seconds, relative reduction, feasible pair fractions at specified node memory budgets |
| M | Does a multi-request backlog expose stage overlap? | 1,4,8,16,32 concurrent requests; same model/different inputs; separately cold and warm | concurrent jobs, phase mix over time, setup reuse, request arrival/departure timeline |

Record the hardware, available RAM, reserve, CPU count, EZKL/Halo2 commit, zkInfer commit, model/partition hash, input shape, setup key, cache state, resource sampling interval, and run repetition. Do not mix cold and warm observations. Warm runs typically skip key generation; repeat cold key generation in a fresh setup-cache namespace while retaining a stable analysis setup identity.

## Input artifacts

Each prover job should retain:
- `status.txt`, `stage_transitions.csv`, `resource_usage.csv`
- `admission_profile_before.json`
- `halo2_stage_transitions.csv` and Halo2 timing CSV where available
- request-level `job_report.csv` and `request_report.csv`
- batch `batch_summary.json` when using the batch launcher

`job_report.csv` is the canonical derived per-job scalar table. Do not create a second duplicate per-job summary. `resource_profile.json` belongs to the persistent setup cache and is runtime scheduling history, not a run report.

Use the exact stage-transition records for duration. Use resource samples for RSS and CPU maxima/averages. Report that sub-second stages may be missed by sampled RSS. Do not equate system-wide RSS with a job's RSS; distinguish process-tree memory from host memory. Use microsecond timestamps to align Halo2 subphases with the outer proof stage. Halo2 end-transition rows include `duration_us` and `duration_s`; start rows intentionally leave duration blank so incomplete regions remain visible.

## Analysis outputs to implement and publish

1. `table_granularity.csv`: model, grouping, number of jobs, aggregate proving time, longest job, peak job RSS, and repetition count. Keep setup excluded for direct comparison with the existing Section 3 table; report setup separately.
2. `figure_job_heterogeneity.pdf`: sorted job proving times and peak process RSS, one panel per metric and model; include p50/p90/p99 markers.
3. `table_phase_characterization.csv`: model, setup key, cold/warm, stage, duration, duration fraction, peak RSS, mean CPU, n; include p50/p90 across jobs.
4. `figure_phase_profiles.pdf`: stage-duration composition and per-stage peak RSS for representative partitions. Do not assume calibration is cheap.
5. `table_repeatability.csv`: setup key, stage, n, mean, standard deviation, CV, p90, max for memory/CPU/duration; aggregate across setup keys with median and p90 CV.
6. `figure_prediction_accuracy.pdf`: predicted versus observed stage peak RSS (identity line); predictions must use *earlier* repetitions only, never the run being predicted. `table_prediction_errors.csv`: underprediction fraction, p95 underprediction GB, median absolute percentage error, prediction source and history size.
7. `table_reservation_opportunity.csv`: whole-job reserved GB-seconds, stage-reserved GB-seconds, unused GB-seconds, percent reduction and pairwise feasible-overlap fractions at explicit memory budgets. Compare like-for-like process RSS measurements and use only stages with valid boundaries.
8. `figure_request_phase_mix.pdf`: stage occupancy over time under 1/4/8/16/32 requests; `table_multi_request.csv`: number of requests/jobs, maximum concurrent jobs, stage overlap fraction, cold/warm mix, and cache hit rate.

**Reservation accounting:** For each isolated job, whole-job reservation is its measured maximum process-tree RSS multiplied by its elapsed execution time. Stage reservation is the sum of each stage's measured peak process-tree RSS multiplied by that stage's duration. Difference is reserved-but-unused capacity attributable to whole-job peak accounting, not measured throughput improvement. A pair is *potentially* co-runnable at a given stage combination if summed stage peaks fit the memory budget while summed whole-job peaks do not. Report budgets and phase weights explicitly.

**Prediction accounting:** Sort repetitions by recorded execution time within setup key; train only on earlier successful runs, predict the next. Distinguish an absent estimate from zero demand. A positive error (prediction minus actual) is overprediction; negative is underprediction. Report underprediction separately because it is relevant to memory safety.

## Quality gates

- Fail analysis loudly when mandatory columns or transition files are absent; do not silently substitute zeros.
- Do not count `WAITING_FOR_*` as stage execution time.
- Treat fused cold keygen/prove as two measured phases for characterization, but as one non-preemptible admission interval.
- Do not merge runs across different circuits or hardware classes.
- Report the number of excluded/failed runs and missing stages.
- Preserve raw inputs and emit a manifest of included runs, configuration, and analysis version.
- Avoid claims that a memory-packing opportunity necessarily translates to throughput.

## Suggested Section 3 layout

3.1 Experimental Setup
3.2 Effect of Proving Granularity (existing table)
3.3 Heterogeneity Across Proving Jobs (existing figure)
3.4 Resource Demand Across Proof Stages
3.5 Repeatability of Stage Resource Demand
3.6 Multi-Request Packing Opportunity
3.7 Implications for Runtime Design

Do not replace existing Section 3 granularity measurements with new numbers until the measurement definitions and hardware match.
