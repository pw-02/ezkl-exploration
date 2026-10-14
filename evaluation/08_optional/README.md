# Optional experiments, driven by claims
Do not expand the paper with experiments unrelated to its contributions.

## Recovery (if claimed)
Hold baseline/configuration fixed, run a warm concurrent batch, then terminate one
worker at a predefined phase/time. Compare against a no-failure run. Report
completed/failed requests, added latency, work repeated, and successful validation.
Use job/lease logs to check completed work is not unnecessarily recomputed.
Failure injection and restart are manual deployment operations, not automated here.

## Runtime overhead
Measure partition-input preparation, scheduling/RPC overhead, artifact handling,
and final verification as fractions of request time. Vary job count or partition
size to identify scaling overhead. Do not subtract overhead from end-to-end latency.

## Backend portability
Retain the old second-backend result if compatibility is current and settings are
documented. It demonstrates portability, not the generality of admission speedups.
The provided runners target the existing EZKL runtime; they do not install or
adapt the other backend.

## Thread allocation (optional follow-up)
Only if real concurrent results indicate CPU contention: hold total cores fixed
and sweep processes versus threads per prover. Measure throughput and latency,
not only CPU percent. No thread-control mechanism is assumed by this plan.
