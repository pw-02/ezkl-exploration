# Setup reuse
Question: how much work is avoided by retaining circuit-specific artifacts?

## Run
Compare **stage cold** versus **stage warm**, with the same partitioning, policy,
hardware and request count. Start at one request; add eight to show the effect
under serving load. Use separate cold/warm memory profiles and document that warm
execution skips setup stages, so differences include both avoided work and the
resulting execution profile.

Cold: reusable setup disabled. Warm: matching reusable artifacts exist locally
before timing. Keep SRS availability the same. Local reuse means loading local
files, not retaining native proving objects in memory.

Supply matching coordinator and node profiles for each condition. Loading cold
profiles does not enable artifact reuse. The runner isolates profile copies per
trial. For an isolated reuse diagnostic, use partitioned-sequential for both arms.

```bash
python evaluation/run.py --baseline stage --workload mnist_gan --setup cold --profile-dir /absolute/cold/resource_profiles --coordinator-profile-dir /absolute/cold-coordinator
python evaluation/run.py --baseline stage --workload mnist_gan --setup warm --setup-dir /absolute/trial/setup --profile-dir /absolute/warm/resource_profiles
```

## Report
Setup generation/loading time, complete request latency, total proving-key storage,
p90 per-job key size and cache hits. Sum of per-job setup times is not request
latency. Reuse enabled does not guarantee local hits: verify reports.

## Existing measurements to reconcile/reuse
| Model | Aggregate generation s | Local reuse s | Saved % | Total keys GiB | p90 GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST GAN | 1056.14 | 38.57 | 96.35 | 30.93 | 1.63 |
| MobileNetV2 | 14608.66 | 390.99 | 97.32 | 287.47 | 6.08 |
| NanoGPT (4L) | 1812.34 | 67.09 | 96.30 | 59.10 | 0.75 |

These are supplied characterization values, not outputs from this runner.
Confirm source reports/configurations before publication. The corrected NanoGPT
reuse time is 67.09 seconds. Keep the compact motivation table; avoid repeating
all of it in evaluation if the new result is request-level benefit.

## Locality
No cloud retrieval number is assumed. If claiming placement benefit, measure
controlled local versus absent-local setup and loaded versus idle workers.
Large artifact sizes motivate this question but do not answer it.
