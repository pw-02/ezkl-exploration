# Impact of partition granularity
Question: how does the partitioning choice change the workload and system behavior?

## Run
Use **stage** at partition sizes 1, 2, 4, 8 on fixed hardware, with one request.
Keep model, inputs, backend/proving settings, threads and node count fixed.
Collect matching profiles and warm artifacts separately for each partition size:
a profile or key from size 1 is not valid automatically for size 4.
Use partitioned-sequential as an optional diagnostic to separate aggregate work
from parallel execution. The monolithic endpoint is already in decomposition.

```bash
python evaluation/run.py --baseline stage --workload mobilenet_v2 --partition-size 4 --requests 1 --profile-dir /absolute/size4/resource_profiles
```

## Report
Aggregate prover time, longest job duration, maximum per-job memory, actual
request latency, total proof size and verification time. Clarify whether aggregate
time excludes setup. A lower summed isolated time does not guarantee lower
distributed request latency. Plot latency and peak job memory versus size; use a
compact secondary table for total work and boundary costs.

## Reuse old work
Move the old partitioning/granularity analysis from motivation here.
The old operator-count versus profile-guided grouping study can be a secondary
result or appendix. The current experiment interface supports fixed operator
counts; this runner does not implement the old profile-guided splitter.
Keep only a brief partitioning rationale/configuration in motivation.
