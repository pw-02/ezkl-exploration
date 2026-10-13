#!/usr/bin/env bash
# Shared sequential synthetic-input campaign launcher.
set -euo pipefail

mode=${1:?Expected pilot or main}
shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
python_bin=${PYTHON_BIN:-python}
source_path=""
model=all
repetitions=10
if [[ $mode == pilot ]]; then
    model=mnist_gan
    repetitions=3
elif [[ $mode != main ]]; then
    printf 'Unknown mode: %s\n' "$mode" >&2
    exit 2
fi
seed=42
interval=0.1
noise_scale=0.01
srs_dir="$repo_root/srs"
output_root="$repo_root/experiments/microbench_outputs"
config=""
integer_bound=""

usage() {
    cat <<EOF
Usage: bash experiments/microbench/run_${mode}.sh --source PATH [options]

  --source PATH                Characterization archive or extracted runs (required)
  --model NAME                 all, mnist_gan, mobilenet_v2, or nanogpt
  --repetitions N              Measured runs per job per condition (default: $repetitions)
  --seed N                     Synthetic full-model input seed (default: 42)
  --interval SECONDS           Resource sampling interval (default: 0.1)
  --noise-scale VALUE          Floating-input perturbation scale (default: 0.01)
  --srs-dir PATH               Existing shared SRS directory
  --output-root PATH           Parent for a new timestamped campaign directory
  --config PATH                Optional model/baseline path overrides in JSON
  --integer-upper-bound N      Valid exclusive bound for synthetic integer inputs
  --help                      Show this message

Pilot runs the median-duration selection only (one per selected model).
Main runs all selected jobs. Both run cold and warm conditions, verify proofs,
and analyze results. Models/jobs execute sequentially, not concurrently.
Set PYTHON_BIN to the Python interpreter in your installed EZKL environment.
EOF
}

while (($#)); do
    case $1 in
        --help|-h) usage; exit 0 ;;
        --source|--model|--repetitions|--seed|--interval|--noise-scale|--srs-dir|--output-root|--config|--integer-upper-bound)
            if (($# < 2)) || [[ $2 == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2; exit 2
            fi
            case $1 in
                --source) source_path=$2 ;;
                --model) model=$2 ;;
                --repetitions) repetitions=$2 ;;
                --seed) seed=$2 ;;
                --interval) interval=$2 ;;
                --noise-scale) noise_scale=$2 ;;
                --srs-dir) srs_dir=$2 ;;
                --output-root) output_root=$2 ;;
                --config) config=$2 ;;
                --integer-upper-bound) integer_bound=$2 ;;
            esac
            shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z $source_path || ! -e $source_path ]]; then
    printf 'Provide --source pointing to an existing archive or runs directory.\n' >&2
    exit 2
fi
if ! [[ $repetitions =~ ^[0-9]+$ ]] || ((10#$repetitions < 3)); then
    printf 'Repetitions must be an integer of at least three.\n' >&2; exit 2
fi
repetitions=$((10#$repetitions))
case $model in
    all) models=(mnist_gan mobilenet_v2 nano_gpt_4_layers_64_embd) ;;
    mnist_gan|mobilenet_v2) models=("$model") ;;
    nanogpt|nano_gpt_4_layers_64_embd) models=(nano_gpt_4_layers_64_embd) ;;
    *) printf 'Unknown model: %s\n' "$model" >&2; exit 2 ;;
esac
if [[ -n $config && ! -f $config ]]; then
    printf 'Config does not exist: %s\n' "$config" >&2; exit 2
fi

# Resolve user paths before changing working directory to the repository root.
absolute_path() {
    "$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}
source_path=$(absolute_path "$source_path")
srs_dir=$(absolute_path "$srs_dir")
output_root=$(absolute_path "$output_root")
if [[ -n $config ]]; then config=$(absolute_path "$config"); fi
cd -- "$repo_root"

# Separate invocation directories prevent accidental result replacement.
mkdir -p -- "$output_root"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
campaign=$(mktemp -d "$output_root/${mode}_${model}_${stamp}_XXXXXX")
prepared="$campaign/prepared"
results="$campaign/results"
printf 'Campaign: %s\n' "$campaign"
trap 'code=$?; if ((code)); then printf "Campaign stopped (exit %s). Logs: %s\n" "$code" "$campaign" >&2; fi' EXIT

prepare_args=(
    -m experiments.microbench.repeatability prepare
    --source "$source_path" --models "${models[@]}"
    --input-mode perturb --seed "$seed" --repetitions "$repetitions"
    --noise-scale "$noise_scale" --output "$prepared"
)
if [[ -n $config ]]; then prepare_args+=(--config "$config"); fi
if [[ -n $integer_bound ]]; then prepare_args+=(--integer-upper-bound "$integer_bound"); fi
"$python_bin" "${prepare_args[@]}" 2>&1 | tee "$campaign/prepare.log"

run_args=(
    -m experiments.microbench.repeatability run
    --prepared "$prepared" --output "$results"
    --interval "$interval" --srs-dir "$srs_dir" --verify
)
if [[ $mode == pilot ]]; then
    # Derive the pilot case from the actual selection, rather than hard-coding
    # partition 2 for arbitrary future characterization traces.
    "$python_bin" - "$prepared/manifest.json" > "$campaign/pilot_cases.txt" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
for case in manifest['cases']:
    if 'median_duration' in case['criteria']:
        print(f"{case['model']}:{case['partition']}")
PY
    while IFS= read -r selected_case; do
        run_args+=(--case "$selected_case")
    done < "$campaign/pilot_cases.txt"
    if [[ ! -s $campaign/pilot_cases.txt ]]; then
        printf 'No pilot cases were selected.\n' >&2; exit 2
    fi
fi
"$python_bin" "${run_args[@]}" 2>&1 | tee "$campaign/run.log"
"$python_bin" -m experiments.microbench.analysis --runs "$results" \
    2>&1 | tee "$campaign/analysis.log"
printf '\nFinished. CSV results: %s/analysis\n' "$results"
