# Proof-job repeatability microbenchmark

This benchmark measures isolated repeated execution of selected partitions. It
does not start a coordinator or supervisor, enable admission, flush the OS page
cache, or change existing setup stores. Each proof runs in a fresh subprocess.

## Bash launchers for synthetic inputs

Activate your usual EZKL Python environment, then run:

```bash
# Pilot: median-duration MNIST GAN job, 3 cold + 3 warm measured repetitions.
bash experiments/microbench/run_pilot.sh --source /path/to/runs.tar.gz

# Main: all 9 selected jobs, 10 cold + 10 warm repetitions each.
bash experiments/microbench/run_main.sh --source /path/to/runs.tar.gz
```

Each launcher executes preparation, proofs, and analysis in order, verifies proofs,
and uses seed 42 with 0.1-second sampling. Warm groups include an additional
unmeasured setup-creation run. Results and launcher logs go into a new timestamped
directory under `experiments/microbench_outputs`; its path is printed at startup.
No JSON input configuration is needed with the default model paths. An archive
or extracted directory with one characterization report per selected model is
required. The scripts do not activate or install a Python environment for you.

Run the main experiment for models individually:

```bash
bash experiments/microbench/run_main.sh --source /path/to/runs.tar.gz --model mnist_gan
bash experiments/microbench/run_main.sh --source /path/to/runs.tar.gz --model mobilenet_v2
bash experiments/microbench/run_main.sh --source /path/to/runs.tar.gz --model nanogpt
```

Run these sequentially on the same machine. Launching them concurrently would
invalidate the isolated-execution condition. Pilot case selection is derived
from the actual prepared manifest, not a hard-coded partition number.

Optional parameters include `--repetitions`, `--seed`, `--interval`, `--srs-dir`,
`--output-root`, `--noise-scale`, and `--config` for model/baseline path overrides.
For integer model inputs, provide a valid `--integer-upper-bound`; do not guess
the vocabulary size. `PYTHON_BIN=/path/to/venv/bin/python` selects an interpreter.
Use `--help` for the full option list. These remain synthetic-input experiments
and should be described that way in the paper.

## Selection

From one completed `job_report.csv` per model, select the jobs closest to the
median and p90 duration, plus the maximum sampled process-memory job. Percentiles
use linear interpolation; equal-distance ties choose the lower partition index.
Duplicate selections are included once. The October 12 characterization archive
selects:

| Model | Median-duration partition | p90-duration partition | Maximum-memory partition |
| --- | ---: | ---: | ---: |
| MNIST GAN | 2 | 10 | 7 |
| MobileNetV2 | 43 | 15 | 96 |
| NanoGPT (4L) | 25 | 172 | 63 |

Selection reads either extracted runs or a `.tar.gz` archive, without extracting
untrusted archive paths. It uses reported job durations, not final stage elapsed
timestamps. It rejects duplicate IDs, failed jobs, and invalid measurements.

Run all commands from the repository root using the Python environment with the
repository's requirements and the custom `pw-02/ezkl:sacdad27` wheel installed.
The cold fused path must provide detailed stage events. ONNX model files and
baseline inputs must match the archived run. Default paths are the three model
paths used in that archive; override them with `--config` if needed.

```bash
python -m experiments.microbench.repeatability select \
  --source experiments/runs \
  --output /tmp/repeatability-selection.json
```

Point `--source` at a narrower directory/archive if it contains multiple runs of
the same model. Use `--models mnist_gan` (or either other model name) for a subset.

## Prepare distinct inputs outside the measured runs

The recommended paper experiment uses real full-model inputs in the standard
EZKL JSON format (`{"input_data": [...]}`). Supply paths in a JSON config:

```json
{
  "mnist_gan": {
    "model": "experiments/models/mnist_gan/mnist_gan.onnx",
    "baseline_input": "experiments/models/mnist_gan/input.json",
    "inputs": ["data/gan/input_01.json", "data/gan/input_02.json", "data/gan/input_03.json"]
  }
}
```

Provide at least as many distinct input files as `--repetitions` for every
selected model. Paths are absolute or relative to the repository root.

```bash
python -m experiments.microbench.repeatability prepare \
  --source experiments/runs \
  --config experiments/microbench/inputs.json \
  --models mnist_gan \
  --repetitions 10 \
  --output experiments/microbench_outputs/prepared
```

For a synthetic pilot, floating-point full-model inputs can instead be perturbed
with seeded Gaussian noise, default standard deviation 1% of their RMS magnitude:

```bash
python -m experiments.microbench.repeatability prepare \
  --source experiments/runs \
  --input-mode perturb \
  --repetitions 10 \
  --output experiments/microbench_outputs/prepared
```

Synthetic inputs are explicitly recorded as such, are not clipped or assumed to
follow a realistic request distribution, and should not be described as real
client traffic. Integer inputs need real files or an explicitly valid
`--integer-upper-bound` (exclusive vocabulary bound). Attention masks and token
type IDs are preserved. Do not guess a vocabulary bound.

Preparation evaluates each full-model input and uses the runtime's splitter to
materialize the selected partition boundary inputs. It checks that each selected
ONNX component matches its archived setup key and remains byte-identical across
inputs. If model, serialization, or splitter changes cause a mismatch, regenerate
the characterization baseline rather than silently treating it as the same
circuit. It also rejects repeated partition inputs, including partitions whose
inputs remain invariant despite varying full-model inputs. This work is not
included in proof measurements.

## Collect cold and warm measurements

```bash
python -m experiments.microbench.repeatability run \
  --prepared experiments/microbench_outputs/prepared \
  --output experiments/microbench_outputs/measurements \
  --interval 0.1 \
  --srs-dir srs \
  --verify
```

For a smaller pilot add `--case mnist_gan:2` (repeat the flag for more cases).
Use `--conditions warm` or `--conditions cold` to run one condition. Output
directories must be new; existing results are never overwritten. Runs stop on
failure, retaining failure logs and any completed measurements for analysis.

- Cold repetitions each start with an empty, benchmark-owned circuit setup
  directory, generating settings, compilation, and keys afresh. Shared SRS files
  remain available and are not deleted or counted as regenerated circuit setup.
- Warm groups first execute a separate, unmeasured setup-creation proof. Measured
  repetitions reuse these local disk artifacts but run in fresh processes, so
  keys are loaded afresh rather than retained in the preceding process's memory.
- Calibration always uses the same baseline partition input, while measured
  proof inputs vary. The first calibration establishes reference proving settings.
  Every cold repetition still executes calibration, but compilation, key generation
  and proving use that fixed reference, even if calibration chooses a different
  candidate on a later run. Candidate settings and changes are recorded in
  `calibration_candidate.json` and `calibration_record.json`; the actual settings
  are stored in `settings_snapshot.json`. This explicitly controls the proving
  configuration rather than assuming calibration is deterministic. Warm cached
  settings must match the reference or the run fails; cached keys are never paired
  with replacement settings. Document this controlled calibration policy in the
  paper. These are cold artifact-generation experiments under fixed proving
  settings, not unconstrained recalibration experiments.
- Actual settings/compiled/key cache flags are checked. A warm miss fails instead
  of silently regenerating setup. A cold cache hit likewise fails.
- `--verify` checks each proof after resource profiling ends, with verification
  time recorded separately from proof execution. Without the flag, completion
  reports execution success, not independently verified correctness.
- Sampling defaults to 0.1 seconds. Even this may miss very short stages, notably
  compilation and witness generation. Sample counts remain explicit.
- Setup and published proof/witness intermediates are deleted after they are no
  longer needed, while all measurement logs and settings snapshots remain. Warm
  caches are removed after their group by default; `--keep-setup` retains them.
  Cleanup never touches the source inputs, shared SRS, or runtime caches.
- No root access or page-cache dropping is used. "Cold" and "warm" describe
  circuit artifact availability, not a guarantee about OS disk-cache state.

The campaign metadata records Git revision and tracked changes, Python/platform,
CPU/RAM, EZKL version and native-library SHA256, prepared manifest hash, sampling
interval, conditions, and verification policy. Every execution records its input
hash, result, settings fingerprint, and task paths. Raw traces include stage
transitions, process-tree RSS/CPU samples, and the available Halo2 profiling CSVs.

## Analyze

```bash
python -m experiments.microbench.analysis \
  --runs experiments/microbench_outputs/measurements
```

Outputs under `measurements/analysis`:

| File | Contents |
| --- | --- |
| `stage_measurements.csv` | Per-run durations, sampled memory peaks, CPU, and sample counts |
| `repeatability.csv` | Per-circuit/condition/stage mean, sample CV, p90, maximum, excluded counts |
| `predictions.csv` | Every causal prediction, history size, error, underprediction and shortfall |
| `prediction_by_history.csv` | Prediction summaries by available history size |
| `failures.csv` | Failed/incomplete measured executions |

For each metric and stage, prediction starts after two valid earlier observations.
With 2–4 observations, use their maximum; with at least five, use their empirical
p90 (linear interpolation). Missing samples do not enter history as zeros. Cold
and warm groups, circuit IDs, stages, and settings fingerprints are kept separate.
The estimator is evaluated as an estimate, not guaranteed to upper-bound memory.

Memory includes only stage executions with at least one valid process-memory
sample. Raw sample counts and excluded counts are reported; one sample is not
proof that the true stage peak was captured. CV uses sample standard deviation.
For repeated Halo2 phase names, durations are summed and observed peaks are taken
over their intervals. Native timestamp intervals are aligned to resource samples
using UTC wall-clock times, rather than assuming the two elapsed clocks match.

Diagnostic Halo2 stages and the combined `FUSED_KEYGEN_PROVE_ENVELOPE` are both
reported. The latter reflects the runtime's actual admission region, including
key/proof persistence within the fused call. Detailed subphase timestamps do not
imply independently controllable admission boundaries. Warm cached preparation
checks are named as cache lookups rather than regeneration.

For publication, aggregate prediction counts across cases while retaining the
history dimension. Ten runs leave only eight predictions per sampled stage; do
not interpret a zero underprediction count as a validated safety guarantee.
