# Distributed scale-out
Question: can additional nodes improve individual-request latency and serving capacity?

## Run
Compare **whole-job** and **stage** at each node count, initially 1, 2, 4, 8
(or the established old 1-to-10-node sweep). Keep instance type, per-node reserve,
backend settings and cache policy fixed. Run:
1. One request: fixed-work single-request speedup.
2. A fixed 16-request batch across all node counts: strong scaling under concurrency.
3. Optional sustained load scaled with node count: capacity scaling, clearly
   separate from fixed-work speedup.

Warm primary, cold companion where needed. Specify whether setup is replicated
everywhere or shared and cached. The scripts do not provision/deploy machines.
Use run.py's printed coordinator/node commands as templates, setting a unique
node_id and local state/cache directory on each host, reachable coordinator host,
and shared artifact/input paths (or an explicitly configured S3 backend).
Do not launch multiple local supervisors and label them separate physical nodes.

After deploying the cluster, drive it with:
```bash
python evaluation/traffic.py --target COORDINATOR:50051 --workload mnist_gan --requests 16 --output evaluation/results/cluster-stage-n4-r1
```

## Report
Latency and throughput versus physical node count, speedup relative to the
one-node partitioned configuration, and node-seconds (node count times elapsed
time) or monetary cost with documented pricing. Report failures and utilization.
One straggling job can limit single-request scaling even when throughput scales.

## Reuse old work
Retain old single-request distributed curves and one-node process-count sweep
if configuration matches or is explicitly separated. New whole-job comparisons
and concurrent capacity curves support the revised resource-management claims.
