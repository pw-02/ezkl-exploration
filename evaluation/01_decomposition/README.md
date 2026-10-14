# Decomposition benefit
Question: what changes when the same inference is proved as smaller circuits,
without gaining extra parallel hardware?

## Run
Compare **monolithic** versus **partitioned-sequential**, one node, one request,
one active job. Run MNIST GAN, MobileNetV2, NanoGPT under cold and warm conditions.
Keep backend, prover threads, arithmetic settings, input, and verification semantics
identical. Collect at least three trials per feasible cell.

```bash
python evaluation/run.py --baseline monolithic --workload mnist_gan --requests 1
python evaluation/run.py --baseline partitioned-sequential --workload mnist_gan --requests 1
```

These commands print plans. Add --execute and a unique --output to run. Warm
variants also need --setup warm and --setup-dir pointing at matching artifacts.

## Report
End-to-end request latency, aggregate prover time, maximum process memory,
total proving-key size, proof bytes and total verification time. Mark OOM as OOM;
do not assign an arbitrary runtime or calculate a finite speedup against failure.
Plot paired latency/memory bars or a compact table, cold and warm distinguished.

## Reuse old work
Old Section 6.2 single-worker comparison, online key-loading/proving breakdown,
and larger-model feasibility result belong here. Recheck versions/units before
reusing numbers. This establishes decomposition, not stage admission.
