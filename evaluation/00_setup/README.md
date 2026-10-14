# Setup and measurement protocol

For exact commands, profile collection and the restartable tmux checklist, start
with [evaluation/setup.md](../setup.md) and [run_all.sh](../run_all.sh).
This page describes the controls; it is not itself a script.

## Hardware and software record

Record commit SHA (including backend/wheel), OS, physical/logical cores, RAM,
storage medium, network, node count, prover-process cap, prover thread environment,
sampling interval, partition size, proving settings and model/input versions.
Use the same prover thread configuration across arms; CPU utilization is diagnostic,
not a fixed admission requirement. Do not infer thread allocation from CPU percent.

## Profiles, cold/warm setup, and trial independence

1. Collect historical profiles in isolated runs before evaluation, under matching
   inputs/settings/setup conditions. Never train on future measured executions.
2. Snapshot the resource_profiles directory. Give each policy/trial the same initial
   snapshot; record online profile updates. Do not let one baseline train the next.
   Also snapshot coordinator profiles: <setup_key>/cold_resource_profile.json for
   forced-cold execution, <setup_key>/resource_profile.json for reuse-enabled
   execution. Profiles can live independently of artifacts. Node profiles alone
   do not populate coordinator job estimates.
3. Cold runs use no reusable artifacts. Warm runs have matching setup already
   prepared. Specify whether copies reside locally on every worker or in shared
   storage. Artifact warmth does not mean OS page-cache warmth.
4. Warm up outside measurement and verify cache-hit metrics. Use equivalent fresh
   copies of setup/cache state per arm. Keep SRS availability consistent.
5. Use new output and runtime-state directories. The runner never deletes caches.
6. Run at least three independent pilot trials, randomizing policy order. Increase
   repetitions only if variability leaves the conclusion unresolved. Reuse arrival
   seeds across policies within each repetition; change them across repetitions.

## Metrics

- Request latency: from client RPC start to observed completed status, including
  submission processing and queueing. Also record from scheduled arrival to completion.
- Burst makespan: first client submission start to last completion. Report actual
  submission spread; requests are not literally simultaneous if injection is slow.
- Achieved throughput: successful completions per second with a declared window.
  Separate steady-state throughput from total completions divided by drain-inclusive
  trial time. Backlog growth and offered rate must accompany saturation claims.
- Report failures/timeouts separately and retain their records. Do not calculate
  apparently improved latency by silently discarding unsuccessful requests.
- Tail percentiles require enough independent observations. With small batches,
  show all points/range rather than presenting an unstable p95 as definitive.
- Peak actual node memory, OOM events, admitted/reserved memory, queueing and stage
  wait times, CPU utilization, setup hits/misses and transfer time explain results.
- In decomposition tests include total proof bytes and verification time when
  claiming an end-to-end advantage; check boundary commitment semantics are equal.

## Memory budgets

For controlled budget B on a host with measured total RAM H, set reserve H-B
using `--reserve-gib`. B must exceed each eligible job's estimated peak. This
restricts accounted proof memory without changing CPU count; it is not a physical
machine with only B GiB of RAM. Preserve headroom for OS/runtime. Record B and H
separately. Do not blindly substitute a cgroup limit: the controller reads host RAM.
Use identical reserves across fixed, whole-job and stage variants.

## Before trusting a run

For controlled cold runs, supply both --profile-dir (node history) and
--coordinator-profile-dir (matching cold history). The runtime reads those
estimates with reusable setup disabled, while workers still regenerate setup.
The runner copies only condition-matched coordinator profile files, not artifacts.

Check config, stage boundaries, setup hits, profile source, job count, verification
success, expected node count, nonzero admission waits where applicable, reservation
release, submission lag, process failures, and all terminal request outcomes.
Do not merge old and new measurements with different circuits/backend settings
into a single speedup ratio. Old figures can be retained with clearly stated setup,
or rerun on the current implementation.
