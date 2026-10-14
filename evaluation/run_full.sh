#!/usr/bin/env bash
# Editable experiment checklist. Comment out any prep/cell line below.
# Default is PLAN ONLY. See evaluation/setup.md before --execute.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
EVAL_ROOT="${EVAL_ROOT:-$PWD/evaluation/results/paper-full-v2}"
SECTION="${SECTION:-all}"
ONLY="${ONLY:-*}"
TIMEOUT="${TIMEOUT:-172800}"
PROFILE_REPEATS="${PROFILE_REPEATS:-1}"
CACHE_GIB="${CACHE_GIB:-500}"
SAMPLING="${SAMPLING:-0.1}"
PORT="${PORT:-50051}"
DISCARD_COPIES="${DISCARD_COPIES:-1}"
MODE="${1:---plan}"
if [[ "$MODE" != --plan && "$MODE" != --execute ]]; then
  echo "Usage: SECTION=profiles|serving|memory|scheduling|arrival|granularity|decomposition|all bash evaluation/run_all.sh [--plan|--execute]" >&2
  exit 2
fi
if [[ "$MODE" == --execute ]]; then
  : "${RAYON_NUM_THREADS:?Set RAYON_NUM_THREADS explicitly; keep it fixed across profiling and trials}"
  "$PYTHON" evaluation/preflight.py --output "$EVAL_ROOT"
fi
if [[ "$MODE" == --plan ]]; then
  TOTAL_GIB="${PLAN_HOST_GIB:-755}"
  echo "PLAN assumes $TOTAL_GIB GiB target RAM; execution always detects actual host RAM."
else
  TOTAL_GIB="$("$PYTHON" -c 'import os; print(os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")/1024**3)')"
fi
reserve_for() {
  "$PYTHON" -c 'import sys; h,b=map(float,sys.argv[1:]); assert b>0 and h-b>=8, "Budget must leave at least 8 GiB on this host"; print(h-b)' "$TOTAL_GIB" "$1"
}
selected() {
  local group="$1" id="$2"
  [[ "$id" == $ONLY ]] && [[ "$SECTION" == all || ",$SECTION," == *",$group,"* ]]
}
invoke() {
  local -a options=(--root "$EVAL_ROOT" --timeout "$TIMEOUT" --cache-gib "$CACHE_GIB" --sampling "$SAMPLING" --port "$PORT")
  [[ "$MODE" == --execute ]] && options+=(--execute)
  [[ "$DISCARD_COPIES" == 1 ]] && options+=(--discard-copies)
  "$PYTHON" -u evaluation/campaign.py "$@" "${options[@]}"
}
prep() {
  local group="$1" model="$2" g="$3" condition="$4"
  local id="$model-g$g-$condition"
  selected "$group" "$id" || return 0
  invoke prepare --id "$id" --workload "$model" --partition "$g" --condition "$condition" --reserve 8 --repetitions "$PROFILE_REPEATS"
}
cell() {
  local group="$1" id="$2" model="$3" g="$4" condition="$5" baseline="$6" n="$7" budget="$8" trial="$9"
  shift 9
  selected "$group" "$id" || return 0
  invoke run --id "$id" --workload "$model" --partition "$g" --condition "$condition" --baseline "$baseline" --requests "$n" --reserve "$(reserve_for "$budget")" --seed "$((4100+trial))" "$@"
}
# Columns: group ID workload granularity setup baseline requests budget_GiB trial
# Each line is one resumable task. DONE.json is the automatic checkmark.
# Three repetitions use rotated baseline order and paired arrival seeds.
# Stop on first failure; failed attempts remain on disk. Re-running retries them.


# 1. PREPARATION: one isolated observation per condition; shared setup-only bootstrap.
prep profiles mnist_gan 1 cold
prep profiles mnist_gan 1 warm
prep profiles mobilenet_v2 1 cold
prep profiles mobilenet_v2 1 warm
prep profiles nano_gpt_4_layers_64_embd 1 cold
prep profiles nano_gpt_4_layers_64_embd 1 warm
prep profiles mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold
prep profiles mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm

# 2. MAIN SERVING: fixed-8 / whole-job / stage, bursts; warm and cold separate.
cell serving burst-mnist_gan-warm-fixed-8-n1-r1 mnist_gan 1 warm fixed-8 1 256 1
cell serving burst-mnist_gan-warm-whole-job-n1-r1 mnist_gan 1 warm whole-job 1 256 1
cell serving burst-mnist_gan-warm-stage-n1-r1 mnist_gan 1 warm stage 1 256 1
cell serving burst-mnist_gan-warm-fixed-8-n4-r1 mnist_gan 1 warm fixed-8 4 256 1
cell serving burst-mnist_gan-warm-whole-job-n4-r1 mnist_gan 1 warm whole-job 4 256 1
cell serving burst-mnist_gan-warm-stage-n4-r1 mnist_gan 1 warm stage 4 256 1
cell serving burst-mnist_gan-warm-fixed-8-n8-r1 mnist_gan 1 warm fixed-8 8 256 1
cell serving burst-mnist_gan-warm-whole-job-n8-r1 mnist_gan 1 warm whole-job 8 256 1
cell serving burst-mnist_gan-warm-stage-n8-r1 mnist_gan 1 warm stage 8 256 1
cell serving burst-mnist_gan-warm-fixed-8-n16-r1 mnist_gan 1 warm fixed-8 16 256 1
cell serving burst-mnist_gan-warm-whole-job-n16-r1 mnist_gan 1 warm whole-job 16 256 1
cell serving burst-mnist_gan-warm-stage-n16-r1 mnist_gan 1 warm stage 16 256 1
cell serving burst-mobilenet_v2-warm-fixed-8-n1-r1 mobilenet_v2 1 warm fixed-8 1 256 1
cell serving burst-mobilenet_v2-warm-whole-job-n1-r1 mobilenet_v2 1 warm whole-job 1 256 1
cell serving burst-mobilenet_v2-warm-stage-n1-r1 mobilenet_v2 1 warm stage 1 256 1
cell serving burst-mobilenet_v2-warm-fixed-8-n4-r1 mobilenet_v2 1 warm fixed-8 4 256 1
cell serving burst-mobilenet_v2-warm-whole-job-n4-r1 mobilenet_v2 1 warm whole-job 4 256 1
cell serving burst-mobilenet_v2-warm-stage-n4-r1 mobilenet_v2 1 warm stage 4 256 1
cell serving burst-mobilenet_v2-warm-fixed-8-n8-r1 mobilenet_v2 1 warm fixed-8 8 256 1
cell serving burst-mobilenet_v2-warm-whole-job-n8-r1 mobilenet_v2 1 warm whole-job 8 256 1
cell serving burst-mobilenet_v2-warm-stage-n8-r1 mobilenet_v2 1 warm stage 8 256 1
cell serving burst-mobilenet_v2-warm-fixed-8-n16-r1 mobilenet_v2 1 warm fixed-8 16 256 1
cell serving burst-mobilenet_v2-warm-whole-job-n16-r1 mobilenet_v2 1 warm whole-job 16 256 1
cell serving burst-mobilenet_v2-warm-stage-n16-r1 mobilenet_v2 1 warm stage 16 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n1-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 1 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n1-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 1 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n1-r1 nano_gpt_4_layers_64_embd 1 warm stage 1 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n4-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 4 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n4-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 4 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n4-r1 nano_gpt_4_layers_64_embd 1 warm stage 4 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n8-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 8 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n8-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 8 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n8-r1 nano_gpt_4_layers_64_embd 1 warm stage 8 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n16-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 16 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n16-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 16 256 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n16-r1 nano_gpt_4_layers_64_embd 1 warm stage 16 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n1-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 1 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n1-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 1 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n1-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 1 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n4-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 4 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n4-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 4 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n4-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 4 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n8-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 8 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n8-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 8 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n8-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 8 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n16-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 16 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n16-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 16 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n16-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 16 256 1
cell serving burst-mnist_gan-cold-fixed-8-n1-r1 mnist_gan 1 cold fixed-8 1 256 1
cell serving burst-mnist_gan-cold-whole-job-n1-r1 mnist_gan 1 cold whole-job 1 256 1
cell serving burst-mnist_gan-cold-stage-n1-r1 mnist_gan 1 cold stage 1 256 1
cell serving burst-mnist_gan-cold-fixed-8-n4-r1 mnist_gan 1 cold fixed-8 4 256 1
cell serving burst-mnist_gan-cold-whole-job-n4-r1 mnist_gan 1 cold whole-job 4 256 1
cell serving burst-mnist_gan-cold-stage-n4-r1 mnist_gan 1 cold stage 4 256 1
cell serving burst-mnist_gan-cold-fixed-8-n8-r1 mnist_gan 1 cold fixed-8 8 256 1
cell serving burst-mnist_gan-cold-whole-job-n8-r1 mnist_gan 1 cold whole-job 8 256 1
cell serving burst-mnist_gan-cold-stage-n8-r1 mnist_gan 1 cold stage 8 256 1
cell serving burst-mnist_gan-cold-fixed-8-n16-r1 mnist_gan 1 cold fixed-8 16 256 1
cell serving burst-mnist_gan-cold-whole-job-n16-r1 mnist_gan 1 cold whole-job 16 256 1
cell serving burst-mnist_gan-cold-stage-n16-r1 mnist_gan 1 cold stage 16 256 1
cell serving burst-mobilenet_v2-cold-fixed-8-n1-r1 mobilenet_v2 1 cold fixed-8 1 256 1
cell serving burst-mobilenet_v2-cold-whole-job-n1-r1 mobilenet_v2 1 cold whole-job 1 256 1
cell serving burst-mobilenet_v2-cold-stage-n1-r1 mobilenet_v2 1 cold stage 1 256 1
cell serving burst-mobilenet_v2-cold-fixed-8-n4-r1 mobilenet_v2 1 cold fixed-8 4 256 1
cell serving burst-mobilenet_v2-cold-whole-job-n4-r1 mobilenet_v2 1 cold whole-job 4 256 1
cell serving burst-mobilenet_v2-cold-stage-n4-r1 mobilenet_v2 1 cold stage 4 256 1
cell serving burst-mobilenet_v2-cold-fixed-8-n8-r1 mobilenet_v2 1 cold fixed-8 8 256 1
cell serving burst-mobilenet_v2-cold-whole-job-n8-r1 mobilenet_v2 1 cold whole-job 8 256 1
cell serving burst-mobilenet_v2-cold-stage-n8-r1 mobilenet_v2 1 cold stage 8 256 1
cell serving burst-mobilenet_v2-cold-fixed-8-n16-r1 mobilenet_v2 1 cold fixed-8 16 256 1
cell serving burst-mobilenet_v2-cold-whole-job-n16-r1 mobilenet_v2 1 cold whole-job 16 256 1
cell serving burst-mobilenet_v2-cold-stage-n16-r1 mobilenet_v2 1 cold stage 16 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n1-r1 nano_gpt_4_layers_64_embd 1 cold fixed-8 1 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n1-r1 nano_gpt_4_layers_64_embd 1 cold whole-job 1 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n1-r1 nano_gpt_4_layers_64_embd 1 cold stage 1 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n4-r1 nano_gpt_4_layers_64_embd 1 cold fixed-8 4 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n4-r1 nano_gpt_4_layers_64_embd 1 cold whole-job 4 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n4-r1 nano_gpt_4_layers_64_embd 1 cold stage 4 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n8-r1 nano_gpt_4_layers_64_embd 1 cold fixed-8 8 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n8-r1 nano_gpt_4_layers_64_embd 1 cold whole-job 8 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n8-r1 nano_gpt_4_layers_64_embd 1 cold stage 8 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n16-r1 nano_gpt_4_layers_64_embd 1 cold fixed-8 16 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n16-r1 nano_gpt_4_layers_64_embd 1 cold whole-job 16 256 1
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n16-r1 nano_gpt_4_layers_64_embd 1 cold stage 16 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n1-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 1 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n1-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 1 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n1-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 1 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n4-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 4 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n4-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 4 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n4-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 4 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n8-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 8 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n8-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 8 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n8-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 8 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n16-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 16 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n16-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 16 256 1
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n16-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 16 256 1
cell serving burst-mnist_gan-warm-whole-job-n1-r2 mnist_gan 1 warm whole-job 1 256 2
cell serving burst-mnist_gan-warm-stage-n1-r2 mnist_gan 1 warm stage 1 256 2
cell serving burst-mnist_gan-warm-fixed-8-n1-r2 mnist_gan 1 warm fixed-8 1 256 2
cell serving burst-mnist_gan-warm-whole-job-n4-r2 mnist_gan 1 warm whole-job 4 256 2
cell serving burst-mnist_gan-warm-stage-n4-r2 mnist_gan 1 warm stage 4 256 2
cell serving burst-mnist_gan-warm-fixed-8-n4-r2 mnist_gan 1 warm fixed-8 4 256 2
cell serving burst-mnist_gan-warm-whole-job-n8-r2 mnist_gan 1 warm whole-job 8 256 2
cell serving burst-mnist_gan-warm-stage-n8-r2 mnist_gan 1 warm stage 8 256 2
cell serving burst-mnist_gan-warm-fixed-8-n8-r2 mnist_gan 1 warm fixed-8 8 256 2
cell serving burst-mnist_gan-warm-whole-job-n16-r2 mnist_gan 1 warm whole-job 16 256 2
cell serving burst-mnist_gan-warm-stage-n16-r2 mnist_gan 1 warm stage 16 256 2
cell serving burst-mnist_gan-warm-fixed-8-n16-r2 mnist_gan 1 warm fixed-8 16 256 2
cell serving burst-mobilenet_v2-warm-whole-job-n1-r2 mobilenet_v2 1 warm whole-job 1 256 2
cell serving burst-mobilenet_v2-warm-stage-n1-r2 mobilenet_v2 1 warm stage 1 256 2
cell serving burst-mobilenet_v2-warm-fixed-8-n1-r2 mobilenet_v2 1 warm fixed-8 1 256 2
cell serving burst-mobilenet_v2-warm-whole-job-n4-r2 mobilenet_v2 1 warm whole-job 4 256 2
cell serving burst-mobilenet_v2-warm-stage-n4-r2 mobilenet_v2 1 warm stage 4 256 2
cell serving burst-mobilenet_v2-warm-fixed-8-n4-r2 mobilenet_v2 1 warm fixed-8 4 256 2
cell serving burst-mobilenet_v2-warm-whole-job-n8-r2 mobilenet_v2 1 warm whole-job 8 256 2
cell serving burst-mobilenet_v2-warm-stage-n8-r2 mobilenet_v2 1 warm stage 8 256 2
cell serving burst-mobilenet_v2-warm-fixed-8-n8-r2 mobilenet_v2 1 warm fixed-8 8 256 2
cell serving burst-mobilenet_v2-warm-whole-job-n16-r2 mobilenet_v2 1 warm whole-job 16 256 2
cell serving burst-mobilenet_v2-warm-stage-n16-r2 mobilenet_v2 1 warm stage 16 256 2
cell serving burst-mobilenet_v2-warm-fixed-8-n16-r2 mobilenet_v2 1 warm fixed-8 16 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n1-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 1 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n1-r2 nano_gpt_4_layers_64_embd 1 warm stage 1 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n1-r2 nano_gpt_4_layers_64_embd 1 warm fixed-8 1 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n4-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 4 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n4-r2 nano_gpt_4_layers_64_embd 1 warm stage 4 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n4-r2 nano_gpt_4_layers_64_embd 1 warm fixed-8 4 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n8-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 8 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n8-r2 nano_gpt_4_layers_64_embd 1 warm stage 8 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n8-r2 nano_gpt_4_layers_64_embd 1 warm fixed-8 8 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n16-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 16 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n16-r2 nano_gpt_4_layers_64_embd 1 warm stage 16 256 2
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n16-r2 nano_gpt_4_layers_64_embd 1 warm fixed-8 16 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n1-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 1 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n1-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 1 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n1-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 1 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n4-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 4 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n4-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 4 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n4-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 4 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n8-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 8 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n8-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 8 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n8-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 8 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n16-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 16 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n16-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 16 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n16-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 16 256 2
cell serving burst-mnist_gan-cold-whole-job-n1-r2 mnist_gan 1 cold whole-job 1 256 2
cell serving burst-mnist_gan-cold-stage-n1-r2 mnist_gan 1 cold stage 1 256 2
cell serving burst-mnist_gan-cold-fixed-8-n1-r2 mnist_gan 1 cold fixed-8 1 256 2
cell serving burst-mnist_gan-cold-whole-job-n4-r2 mnist_gan 1 cold whole-job 4 256 2
cell serving burst-mnist_gan-cold-stage-n4-r2 mnist_gan 1 cold stage 4 256 2
cell serving burst-mnist_gan-cold-fixed-8-n4-r2 mnist_gan 1 cold fixed-8 4 256 2
cell serving burst-mnist_gan-cold-whole-job-n8-r2 mnist_gan 1 cold whole-job 8 256 2
cell serving burst-mnist_gan-cold-stage-n8-r2 mnist_gan 1 cold stage 8 256 2
cell serving burst-mnist_gan-cold-fixed-8-n8-r2 mnist_gan 1 cold fixed-8 8 256 2
cell serving burst-mnist_gan-cold-whole-job-n16-r2 mnist_gan 1 cold whole-job 16 256 2
cell serving burst-mnist_gan-cold-stage-n16-r2 mnist_gan 1 cold stage 16 256 2
cell serving burst-mnist_gan-cold-fixed-8-n16-r2 mnist_gan 1 cold fixed-8 16 256 2
cell serving burst-mobilenet_v2-cold-whole-job-n1-r2 mobilenet_v2 1 cold whole-job 1 256 2
cell serving burst-mobilenet_v2-cold-stage-n1-r2 mobilenet_v2 1 cold stage 1 256 2
cell serving burst-mobilenet_v2-cold-fixed-8-n1-r2 mobilenet_v2 1 cold fixed-8 1 256 2
cell serving burst-mobilenet_v2-cold-whole-job-n4-r2 mobilenet_v2 1 cold whole-job 4 256 2
cell serving burst-mobilenet_v2-cold-stage-n4-r2 mobilenet_v2 1 cold stage 4 256 2
cell serving burst-mobilenet_v2-cold-fixed-8-n4-r2 mobilenet_v2 1 cold fixed-8 4 256 2
cell serving burst-mobilenet_v2-cold-whole-job-n8-r2 mobilenet_v2 1 cold whole-job 8 256 2
cell serving burst-mobilenet_v2-cold-stage-n8-r2 mobilenet_v2 1 cold stage 8 256 2
cell serving burst-mobilenet_v2-cold-fixed-8-n8-r2 mobilenet_v2 1 cold fixed-8 8 256 2
cell serving burst-mobilenet_v2-cold-whole-job-n16-r2 mobilenet_v2 1 cold whole-job 16 256 2
cell serving burst-mobilenet_v2-cold-stage-n16-r2 mobilenet_v2 1 cold stage 16 256 2
cell serving burst-mobilenet_v2-cold-fixed-8-n16-r2 mobilenet_v2 1 cold fixed-8 16 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n1-r2 nano_gpt_4_layers_64_embd 1 cold whole-job 1 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n1-r2 nano_gpt_4_layers_64_embd 1 cold stage 1 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n1-r2 nano_gpt_4_layers_64_embd 1 cold fixed-8 1 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n4-r2 nano_gpt_4_layers_64_embd 1 cold whole-job 4 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n4-r2 nano_gpt_4_layers_64_embd 1 cold stage 4 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n4-r2 nano_gpt_4_layers_64_embd 1 cold fixed-8 4 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n8-r2 nano_gpt_4_layers_64_embd 1 cold whole-job 8 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n8-r2 nano_gpt_4_layers_64_embd 1 cold stage 8 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n8-r2 nano_gpt_4_layers_64_embd 1 cold fixed-8 8 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n16-r2 nano_gpt_4_layers_64_embd 1 cold whole-job 16 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n16-r2 nano_gpt_4_layers_64_embd 1 cold stage 16 256 2
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n16-r2 nano_gpt_4_layers_64_embd 1 cold fixed-8 16 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n1-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 1 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n1-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 1 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n1-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 1 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n4-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 4 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n4-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 4 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n4-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 4 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n8-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 8 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n8-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 8 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n8-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 8 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n16-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 16 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n16-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 16 256 2
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n16-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 16 256 2
cell serving burst-mnist_gan-warm-stage-n1-r3 mnist_gan 1 warm stage 1 256 3
cell serving burst-mnist_gan-warm-fixed-8-n1-r3 mnist_gan 1 warm fixed-8 1 256 3
cell serving burst-mnist_gan-warm-whole-job-n1-r3 mnist_gan 1 warm whole-job 1 256 3
cell serving burst-mnist_gan-warm-stage-n4-r3 mnist_gan 1 warm stage 4 256 3
cell serving burst-mnist_gan-warm-fixed-8-n4-r3 mnist_gan 1 warm fixed-8 4 256 3
cell serving burst-mnist_gan-warm-whole-job-n4-r3 mnist_gan 1 warm whole-job 4 256 3
cell serving burst-mnist_gan-warm-stage-n8-r3 mnist_gan 1 warm stage 8 256 3
cell serving burst-mnist_gan-warm-fixed-8-n8-r3 mnist_gan 1 warm fixed-8 8 256 3
cell serving burst-mnist_gan-warm-whole-job-n8-r3 mnist_gan 1 warm whole-job 8 256 3
cell serving burst-mnist_gan-warm-stage-n16-r3 mnist_gan 1 warm stage 16 256 3
cell serving burst-mnist_gan-warm-fixed-8-n16-r3 mnist_gan 1 warm fixed-8 16 256 3
cell serving burst-mnist_gan-warm-whole-job-n16-r3 mnist_gan 1 warm whole-job 16 256 3
cell serving burst-mobilenet_v2-warm-stage-n1-r3 mobilenet_v2 1 warm stage 1 256 3
cell serving burst-mobilenet_v2-warm-fixed-8-n1-r3 mobilenet_v2 1 warm fixed-8 1 256 3
cell serving burst-mobilenet_v2-warm-whole-job-n1-r3 mobilenet_v2 1 warm whole-job 1 256 3
cell serving burst-mobilenet_v2-warm-stage-n4-r3 mobilenet_v2 1 warm stage 4 256 3
cell serving burst-mobilenet_v2-warm-fixed-8-n4-r3 mobilenet_v2 1 warm fixed-8 4 256 3
cell serving burst-mobilenet_v2-warm-whole-job-n4-r3 mobilenet_v2 1 warm whole-job 4 256 3
cell serving burst-mobilenet_v2-warm-stage-n8-r3 mobilenet_v2 1 warm stage 8 256 3
cell serving burst-mobilenet_v2-warm-fixed-8-n8-r3 mobilenet_v2 1 warm fixed-8 8 256 3
cell serving burst-mobilenet_v2-warm-whole-job-n8-r3 mobilenet_v2 1 warm whole-job 8 256 3
cell serving burst-mobilenet_v2-warm-stage-n16-r3 mobilenet_v2 1 warm stage 16 256 3
cell serving burst-mobilenet_v2-warm-fixed-8-n16-r3 mobilenet_v2 1 warm fixed-8 16 256 3
cell serving burst-mobilenet_v2-warm-whole-job-n16-r3 mobilenet_v2 1 warm whole-job 16 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n1-r3 nano_gpt_4_layers_64_embd 1 warm stage 1 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n1-r3 nano_gpt_4_layers_64_embd 1 warm fixed-8 1 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n1-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 1 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n4-r3 nano_gpt_4_layers_64_embd 1 warm stage 4 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n4-r3 nano_gpt_4_layers_64_embd 1 warm fixed-8 4 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n4-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 4 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n8-r3 nano_gpt_4_layers_64_embd 1 warm stage 8 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n8-r3 nano_gpt_4_layers_64_embd 1 warm fixed-8 8 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n8-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 8 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n16-r3 nano_gpt_4_layers_64_embd 1 warm stage 16 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n16-r3 nano_gpt_4_layers_64_embd 1 warm fixed-8 16 256 3
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n16-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 16 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n1-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 1 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n1-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 1 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n1-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 1 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n4-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 4 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n4-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 4 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n4-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 4 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n8-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 8 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n8-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 8 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n8-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 8 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-stage-n16-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 16 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-fixed-8-n16-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 16 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm-whole-job-n16-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 16 256 3
cell serving burst-mnist_gan-cold-stage-n1-r3 mnist_gan 1 cold stage 1 256 3
cell serving burst-mnist_gan-cold-fixed-8-n1-r3 mnist_gan 1 cold fixed-8 1 256 3
cell serving burst-mnist_gan-cold-whole-job-n1-r3 mnist_gan 1 cold whole-job 1 256 3
cell serving burst-mnist_gan-cold-stage-n4-r3 mnist_gan 1 cold stage 4 256 3
cell serving burst-mnist_gan-cold-fixed-8-n4-r3 mnist_gan 1 cold fixed-8 4 256 3
cell serving burst-mnist_gan-cold-whole-job-n4-r3 mnist_gan 1 cold whole-job 4 256 3
cell serving burst-mnist_gan-cold-stage-n8-r3 mnist_gan 1 cold stage 8 256 3
cell serving burst-mnist_gan-cold-fixed-8-n8-r3 mnist_gan 1 cold fixed-8 8 256 3
cell serving burst-mnist_gan-cold-whole-job-n8-r3 mnist_gan 1 cold whole-job 8 256 3
cell serving burst-mnist_gan-cold-stage-n16-r3 mnist_gan 1 cold stage 16 256 3
cell serving burst-mnist_gan-cold-fixed-8-n16-r3 mnist_gan 1 cold fixed-8 16 256 3
cell serving burst-mnist_gan-cold-whole-job-n16-r3 mnist_gan 1 cold whole-job 16 256 3
cell serving burst-mobilenet_v2-cold-stage-n1-r3 mobilenet_v2 1 cold stage 1 256 3
cell serving burst-mobilenet_v2-cold-fixed-8-n1-r3 mobilenet_v2 1 cold fixed-8 1 256 3
cell serving burst-mobilenet_v2-cold-whole-job-n1-r3 mobilenet_v2 1 cold whole-job 1 256 3
cell serving burst-mobilenet_v2-cold-stage-n4-r3 mobilenet_v2 1 cold stage 4 256 3
cell serving burst-mobilenet_v2-cold-fixed-8-n4-r3 mobilenet_v2 1 cold fixed-8 4 256 3
cell serving burst-mobilenet_v2-cold-whole-job-n4-r3 mobilenet_v2 1 cold whole-job 4 256 3
cell serving burst-mobilenet_v2-cold-stage-n8-r3 mobilenet_v2 1 cold stage 8 256 3
cell serving burst-mobilenet_v2-cold-fixed-8-n8-r3 mobilenet_v2 1 cold fixed-8 8 256 3
cell serving burst-mobilenet_v2-cold-whole-job-n8-r3 mobilenet_v2 1 cold whole-job 8 256 3
cell serving burst-mobilenet_v2-cold-stage-n16-r3 mobilenet_v2 1 cold stage 16 256 3
cell serving burst-mobilenet_v2-cold-fixed-8-n16-r3 mobilenet_v2 1 cold fixed-8 16 256 3
cell serving burst-mobilenet_v2-cold-whole-job-n16-r3 mobilenet_v2 1 cold whole-job 16 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n1-r3 nano_gpt_4_layers_64_embd 1 cold stage 1 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n1-r3 nano_gpt_4_layers_64_embd 1 cold fixed-8 1 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n1-r3 nano_gpt_4_layers_64_embd 1 cold whole-job 1 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n4-r3 nano_gpt_4_layers_64_embd 1 cold stage 4 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n4-r3 nano_gpt_4_layers_64_embd 1 cold fixed-8 4 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n4-r3 nano_gpt_4_layers_64_embd 1 cold whole-job 4 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n8-r3 nano_gpt_4_layers_64_embd 1 cold stage 8 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n8-r3 nano_gpt_4_layers_64_embd 1 cold fixed-8 8 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n8-r3 nano_gpt_4_layers_64_embd 1 cold whole-job 8 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-stage-n16-r3 nano_gpt_4_layers_64_embd 1 cold stage 16 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-fixed-8-n16-r3 nano_gpt_4_layers_64_embd 1 cold fixed-8 16 256 3
cell serving burst-nano_gpt_4_layers_64_embd-cold-whole-job-n16-r3 nano_gpt_4_layers_64_embd 1 cold whole-job 16 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n1-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 1 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n1-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 1 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n1-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 1 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n4-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 4 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n4-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 4 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n4-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 4 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n8-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 8 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n8-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 8 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n8-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 8 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-stage-n16-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold stage 16 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-fixed-8-n16-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold fixed-8 16 256 3
cell serving burst-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-cold-whole-job-n16-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 cold whole-job 16 256 3

# 3. MEMORY ADMISSION: N=16. Budget 256 reuses the main burst runs.
cell memory memory-mnist_gan-warm-whole-job-b64-r1 mnist_gan 1 warm whole-job 16 64 1
cell memory memory-mnist_gan-warm-stage-b64-r1 mnist_gan 1 warm stage 16 64 1
cell memory memory-mnist_gan-warm-whole-job-b128-r1 mnist_gan 1 warm whole-job 16 128 1
cell memory memory-mnist_gan-warm-stage-b128-r1 mnist_gan 1 warm stage 16 128 1
cell memory memory-mobilenet_v2-warm-whole-job-b64-r1 mobilenet_v2 1 warm whole-job 16 64 1
cell memory memory-mobilenet_v2-warm-stage-b64-r1 mobilenet_v2 1 warm stage 16 64 1
cell memory memory-mobilenet_v2-warm-whole-job-b128-r1 mobilenet_v2 1 warm whole-job 16 128 1
cell memory memory-mobilenet_v2-warm-stage-b128-r1 mobilenet_v2 1 warm stage 16 128 1
cell memory memory-nano_gpt_4_layers_64_embd-warm-whole-job-b64-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 16 64 1
cell memory memory-nano_gpt_4_layers_64_embd-warm-stage-b64-r1 nano_gpt_4_layers_64_embd 1 warm stage 16 64 1
cell memory memory-nano_gpt_4_layers_64_embd-warm-whole-job-b128-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 16 128 1
cell memory memory-nano_gpt_4_layers_64_embd-warm-stage-b128-r1 nano_gpt_4_layers_64_embd 1 warm stage 16 128 1
cell memory memory-mnist_gan-cold-whole-job-b64-r1 mnist_gan 1 cold whole-job 16 64 1
cell memory memory-mnist_gan-cold-stage-b64-r1 mnist_gan 1 cold stage 16 64 1
cell memory memory-mnist_gan-cold-whole-job-b128-r1 mnist_gan 1 cold whole-job 16 128 1
cell memory memory-mnist_gan-cold-stage-b128-r1 mnist_gan 1 cold stage 16 128 1
cell memory memory-mobilenet_v2-cold-whole-job-b64-r1 mobilenet_v2 1 cold whole-job 16 64 1
cell memory memory-mobilenet_v2-cold-stage-b64-r1 mobilenet_v2 1 cold stage 16 64 1
cell memory memory-mobilenet_v2-cold-whole-job-b128-r1 mobilenet_v2 1 cold whole-job 16 128 1
cell memory memory-mobilenet_v2-cold-stage-b128-r1 mobilenet_v2 1 cold stage 16 128 1
cell memory memory-nano_gpt_4_layers_64_embd-cold-whole-job-b64-r1 nano_gpt_4_layers_64_embd 1 cold whole-job 16 64 1
cell memory memory-nano_gpt_4_layers_64_embd-cold-stage-b64-r1 nano_gpt_4_layers_64_embd 1 cold stage 16 64 1
cell memory memory-nano_gpt_4_layers_64_embd-cold-whole-job-b128-r1 nano_gpt_4_layers_64_embd 1 cold whole-job 16 128 1
cell memory memory-nano_gpt_4_layers_64_embd-cold-stage-b128-r1 nano_gpt_4_layers_64_embd 1 cold stage 16 128 1
cell memory memory-mnist_gan-warm-stage-b64-r2 mnist_gan 1 warm stage 16 64 2
cell memory memory-mnist_gan-warm-whole-job-b64-r2 mnist_gan 1 warm whole-job 16 64 2
cell memory memory-mnist_gan-warm-stage-b128-r2 mnist_gan 1 warm stage 16 128 2
cell memory memory-mnist_gan-warm-whole-job-b128-r2 mnist_gan 1 warm whole-job 16 128 2
cell memory memory-mobilenet_v2-warm-stage-b64-r2 mobilenet_v2 1 warm stage 16 64 2
cell memory memory-mobilenet_v2-warm-whole-job-b64-r2 mobilenet_v2 1 warm whole-job 16 64 2
cell memory memory-mobilenet_v2-warm-stage-b128-r2 mobilenet_v2 1 warm stage 16 128 2
cell memory memory-mobilenet_v2-warm-whole-job-b128-r2 mobilenet_v2 1 warm whole-job 16 128 2
cell memory memory-nano_gpt_4_layers_64_embd-warm-stage-b64-r2 nano_gpt_4_layers_64_embd 1 warm stage 16 64 2
cell memory memory-nano_gpt_4_layers_64_embd-warm-whole-job-b64-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 16 64 2
cell memory memory-nano_gpt_4_layers_64_embd-warm-stage-b128-r2 nano_gpt_4_layers_64_embd 1 warm stage 16 128 2
cell memory memory-nano_gpt_4_layers_64_embd-warm-whole-job-b128-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 16 128 2
cell memory memory-mnist_gan-cold-stage-b64-r2 mnist_gan 1 cold stage 16 64 2
cell memory memory-mnist_gan-cold-whole-job-b64-r2 mnist_gan 1 cold whole-job 16 64 2
cell memory memory-mnist_gan-cold-stage-b128-r2 mnist_gan 1 cold stage 16 128 2
cell memory memory-mnist_gan-cold-whole-job-b128-r2 mnist_gan 1 cold whole-job 16 128 2
cell memory memory-mobilenet_v2-cold-stage-b64-r2 mobilenet_v2 1 cold stage 16 64 2
cell memory memory-mobilenet_v2-cold-whole-job-b64-r2 mobilenet_v2 1 cold whole-job 16 64 2
cell memory memory-mobilenet_v2-cold-stage-b128-r2 mobilenet_v2 1 cold stage 16 128 2
cell memory memory-mobilenet_v2-cold-whole-job-b128-r2 mobilenet_v2 1 cold whole-job 16 128 2
cell memory memory-nano_gpt_4_layers_64_embd-cold-stage-b64-r2 nano_gpt_4_layers_64_embd 1 cold stage 16 64 2
cell memory memory-nano_gpt_4_layers_64_embd-cold-whole-job-b64-r2 nano_gpt_4_layers_64_embd 1 cold whole-job 16 64 2
cell memory memory-nano_gpt_4_layers_64_embd-cold-stage-b128-r2 nano_gpt_4_layers_64_embd 1 cold stage 16 128 2
cell memory memory-nano_gpt_4_layers_64_embd-cold-whole-job-b128-r2 nano_gpt_4_layers_64_embd 1 cold whole-job 16 128 2
cell memory memory-mnist_gan-warm-whole-job-b64-r3 mnist_gan 1 warm whole-job 16 64 3
cell memory memory-mnist_gan-warm-stage-b64-r3 mnist_gan 1 warm stage 16 64 3
cell memory memory-mnist_gan-warm-whole-job-b128-r3 mnist_gan 1 warm whole-job 16 128 3
cell memory memory-mnist_gan-warm-stage-b128-r3 mnist_gan 1 warm stage 16 128 3
cell memory memory-mobilenet_v2-warm-whole-job-b64-r3 mobilenet_v2 1 warm whole-job 16 64 3
cell memory memory-mobilenet_v2-warm-stage-b64-r3 mobilenet_v2 1 warm stage 16 64 3
cell memory memory-mobilenet_v2-warm-whole-job-b128-r3 mobilenet_v2 1 warm whole-job 16 128 3
cell memory memory-mobilenet_v2-warm-stage-b128-r3 mobilenet_v2 1 warm stage 16 128 3
cell memory memory-nano_gpt_4_layers_64_embd-warm-whole-job-b64-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 16 64 3
cell memory memory-nano_gpt_4_layers_64_embd-warm-stage-b64-r3 nano_gpt_4_layers_64_embd 1 warm stage 16 64 3
cell memory memory-nano_gpt_4_layers_64_embd-warm-whole-job-b128-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 16 128 3
cell memory memory-nano_gpt_4_layers_64_embd-warm-stage-b128-r3 nano_gpt_4_layers_64_embd 1 warm stage 16 128 3
cell memory memory-mnist_gan-cold-whole-job-b64-r3 mnist_gan 1 cold whole-job 16 64 3
cell memory memory-mnist_gan-cold-stage-b64-r3 mnist_gan 1 cold stage 16 64 3
cell memory memory-mnist_gan-cold-whole-job-b128-r3 mnist_gan 1 cold whole-job 16 128 3
cell memory memory-mnist_gan-cold-stage-b128-r3 mnist_gan 1 cold stage 16 128 3
cell memory memory-mobilenet_v2-cold-whole-job-b64-r3 mobilenet_v2 1 cold whole-job 16 64 3
cell memory memory-mobilenet_v2-cold-stage-b64-r3 mobilenet_v2 1 cold stage 16 64 3
cell memory memory-mobilenet_v2-cold-whole-job-b128-r3 mobilenet_v2 1 cold whole-job 16 128 3
cell memory memory-mobilenet_v2-cold-stage-b128-r3 mobilenet_v2 1 cold stage 16 128 3
cell memory memory-nano_gpt_4_layers_64_embd-cold-whole-job-b64-r3 nano_gpt_4_layers_64_embd 1 cold whole-job 16 64 3
cell memory memory-nano_gpt_4_layers_64_embd-cold-stage-b64-r3 nano_gpt_4_layers_64_embd 1 cold stage 16 64 3
cell memory memory-nano_gpt_4_layers_64_embd-cold-whole-job-b128-r3 nano_gpt_4_layers_64_embd 1 cold whole-job 16 128 3
cell memory memory-nano_gpt_4_layers_64_embd-cold-stage-b128-r3 nano_gpt_4_layers_64_embd 1 cold stage 16 128 3

# 4. SCHEDULING + FIXED-CONCURRENCY SENSITIVITY: main stage/fixed-8 are references.
cell scheduling policy-mnist_gan-stage-fifo-r1 mnist_gan 1 warm stage-fifo 16 256 1
cell scheduling policy-mnist_gan-stage-lpt-r1 mnist_gan 1 warm stage-lpt 16 256 1
cell scheduling policy-mnist_gan-fixed-1-r1 mnist_gan 1 warm fixed-1 16 256 1
cell scheduling policy-mnist_gan-fixed-2-r1 mnist_gan 1 warm fixed-2 16 256 1
cell scheduling policy-mnist_gan-fixed-4-r1 mnist_gan 1 warm fixed-4 16 256 1
cell scheduling policy-mnist_gan-fixed-16-r1 mnist_gan 1 warm fixed-16 16 256 1
cell scheduling policy-mobilenet_v2-stage-fifo-r1 mobilenet_v2 1 warm stage-fifo 16 256 1
cell scheduling policy-mobilenet_v2-stage-lpt-r1 mobilenet_v2 1 warm stage-lpt 16 256 1
cell scheduling policy-mobilenet_v2-fixed-1-r1 mobilenet_v2 1 warm fixed-1 16 256 1
cell scheduling policy-mobilenet_v2-fixed-2-r1 mobilenet_v2 1 warm fixed-2 16 256 1
cell scheduling policy-mobilenet_v2-fixed-4-r1 mobilenet_v2 1 warm fixed-4 16 256 1
cell scheduling policy-mobilenet_v2-fixed-16-r1 mobilenet_v2 1 warm fixed-16 16 256 1
cell scheduling policy-nano_gpt_4_layers_64_embd-stage-fifo-r1 nano_gpt_4_layers_64_embd 1 warm stage-fifo 16 256 1
cell scheduling policy-nano_gpt_4_layers_64_embd-stage-lpt-r1 nano_gpt_4_layers_64_embd 1 warm stage-lpt 16 256 1
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-1-r1 nano_gpt_4_layers_64_embd 1 warm fixed-1 16 256 1
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-2-r1 nano_gpt_4_layers_64_embd 1 warm fixed-2 16 256 1
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-4-r1 nano_gpt_4_layers_64_embd 1 warm fixed-4 16 256 1
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-16-r1 nano_gpt_4_layers_64_embd 1 warm fixed-16 16 256 1
cell scheduling policy-mnist_gan-stage-fifo-r2 mnist_gan 1 warm stage-fifo 16 256 2
cell scheduling policy-mnist_gan-stage-lpt-r2 mnist_gan 1 warm stage-lpt 16 256 2
cell scheduling policy-mnist_gan-fixed-1-r2 mnist_gan 1 warm fixed-1 16 256 2
cell scheduling policy-mnist_gan-fixed-2-r2 mnist_gan 1 warm fixed-2 16 256 2
cell scheduling policy-mnist_gan-fixed-4-r2 mnist_gan 1 warm fixed-4 16 256 2
cell scheduling policy-mnist_gan-fixed-16-r2 mnist_gan 1 warm fixed-16 16 256 2
cell scheduling policy-mobilenet_v2-stage-fifo-r2 mobilenet_v2 1 warm stage-fifo 16 256 2
cell scheduling policy-mobilenet_v2-stage-lpt-r2 mobilenet_v2 1 warm stage-lpt 16 256 2
cell scheduling policy-mobilenet_v2-fixed-1-r2 mobilenet_v2 1 warm fixed-1 16 256 2
cell scheduling policy-mobilenet_v2-fixed-2-r2 mobilenet_v2 1 warm fixed-2 16 256 2
cell scheduling policy-mobilenet_v2-fixed-4-r2 mobilenet_v2 1 warm fixed-4 16 256 2
cell scheduling policy-mobilenet_v2-fixed-16-r2 mobilenet_v2 1 warm fixed-16 16 256 2
cell scheduling policy-nano_gpt_4_layers_64_embd-stage-fifo-r2 nano_gpt_4_layers_64_embd 1 warm stage-fifo 16 256 2
cell scheduling policy-nano_gpt_4_layers_64_embd-stage-lpt-r2 nano_gpt_4_layers_64_embd 1 warm stage-lpt 16 256 2
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-1-r2 nano_gpt_4_layers_64_embd 1 warm fixed-1 16 256 2
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-2-r2 nano_gpt_4_layers_64_embd 1 warm fixed-2 16 256 2
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-4-r2 nano_gpt_4_layers_64_embd 1 warm fixed-4 16 256 2
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-16-r2 nano_gpt_4_layers_64_embd 1 warm fixed-16 16 256 2
cell scheduling policy-mnist_gan-stage-fifo-r3 mnist_gan 1 warm stage-fifo 16 256 3
cell scheduling policy-mnist_gan-stage-lpt-r3 mnist_gan 1 warm stage-lpt 16 256 3
cell scheduling policy-mnist_gan-fixed-1-r3 mnist_gan 1 warm fixed-1 16 256 3
cell scheduling policy-mnist_gan-fixed-2-r3 mnist_gan 1 warm fixed-2 16 256 3
cell scheduling policy-mnist_gan-fixed-4-r3 mnist_gan 1 warm fixed-4 16 256 3
cell scheduling policy-mnist_gan-fixed-16-r3 mnist_gan 1 warm fixed-16 16 256 3
cell scheduling policy-mobilenet_v2-stage-fifo-r3 mobilenet_v2 1 warm stage-fifo 16 256 3
cell scheduling policy-mobilenet_v2-stage-lpt-r3 mobilenet_v2 1 warm stage-lpt 16 256 3
cell scheduling policy-mobilenet_v2-fixed-1-r3 mobilenet_v2 1 warm fixed-1 16 256 3
cell scheduling policy-mobilenet_v2-fixed-2-r3 mobilenet_v2 1 warm fixed-2 16 256 3
cell scheduling policy-mobilenet_v2-fixed-4-r3 mobilenet_v2 1 warm fixed-4 16 256 3
cell scheduling policy-mobilenet_v2-fixed-16-r3 mobilenet_v2 1 warm fixed-16 16 256 3
cell scheduling policy-nano_gpt_4_layers_64_embd-stage-fifo-r3 nano_gpt_4_layers_64_embd 1 warm stage-fifo 16 256 3
cell scheduling policy-nano_gpt_4_layers_64_embd-stage-lpt-r3 nano_gpt_4_layers_64_embd 1 warm stage-lpt 16 256 3
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-1-r3 nano_gpt_4_layers_64_embd 1 warm fixed-1 16 256 3
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-2-r3 nano_gpt_4_layers_64_embd 1 warm fixed-2 16 256 3
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-4-r3 nano_gpt_4_layers_64_embd 1 warm fixed-4 16 256 3
cell scheduling policy-nano_gpt_4_layers_64_embd-fixed-16-r3 nano_gpt_4_layers_64_embd 1 warm fixed-16 16 256 3

# 5. REUSE / OVERHEAD: reuse stage burst N=1/8/16 cold/warm reports.
# No duplicate runs. Fine-grained scheduler/check overhead needs instrumentation.

# 6. ARRIVAL PILOTS: independent warm whole-job 8-request batches.
cell arrival pilot-mnist_gan-warm mnist_gan 1 warm whole-job 8 256 99
cell arrival pilot-mobilenet_v2-warm mobilenet_v2 1 warm whole-job 8 256 99
cell arrival pilot-nano_gpt_4_layers_64_embd-warm nano_gpt_4_layers_64_embd 1 warm whole-job 8 256 99
cell arrival pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 8 256 99
cell arrival arrival-mnist_gan-fixed-8-f0.25-r1 mnist_gan 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.25
cell arrival arrival-mnist_gan-whole-job-f0.25-r1 mnist_gan 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.25
cell arrival arrival-mnist_gan-stage-f0.25-r1 mnist_gan 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.25
cell arrival arrival-mnist_gan-fixed-8-f0.5-r1 mnist_gan 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.5
cell arrival arrival-mnist_gan-whole-job-f0.5-r1 mnist_gan 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.5
cell arrival arrival-mnist_gan-stage-f0.5-r1 mnist_gan 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.5
cell arrival arrival-mnist_gan-fixed-8-f0.75-r1 mnist_gan 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.75
cell arrival arrival-mnist_gan-whole-job-f0.75-r1 mnist_gan 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.75
cell arrival arrival-mnist_gan-stage-f0.75-r1 mnist_gan 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.75
cell arrival arrival-mnist_gan-fixed-8-f1-r1 mnist_gan 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1
cell arrival arrival-mnist_gan-whole-job-f1-r1 mnist_gan 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1
cell arrival arrival-mnist_gan-stage-f1-r1 mnist_gan 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1
cell arrival arrival-mnist_gan-fixed-8-f1.25-r1 mnist_gan 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1.25
cell arrival arrival-mnist_gan-whole-job-f1.25-r1 mnist_gan 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1.25
cell arrival arrival-mnist_gan-stage-f1.25-r1 mnist_gan 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1.25
cell arrival arrival-mobilenet_v2-fixed-8-f0.25-r1 mobilenet_v2 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.25
cell arrival arrival-mobilenet_v2-whole-job-f0.25-r1 mobilenet_v2 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.25
cell arrival arrival-mobilenet_v2-stage-f0.25-r1 mobilenet_v2 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.25
cell arrival arrival-mobilenet_v2-fixed-8-f0.5-r1 mobilenet_v2 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.5
cell arrival arrival-mobilenet_v2-whole-job-f0.5-r1 mobilenet_v2 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.5
cell arrival arrival-mobilenet_v2-stage-f0.5-r1 mobilenet_v2 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.5
cell arrival arrival-mobilenet_v2-fixed-8-f0.75-r1 mobilenet_v2 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.75
cell arrival arrival-mobilenet_v2-whole-job-f0.75-r1 mobilenet_v2 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.75
cell arrival arrival-mobilenet_v2-stage-f0.75-r1 mobilenet_v2 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.75
cell arrival arrival-mobilenet_v2-fixed-8-f1-r1 mobilenet_v2 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1
cell arrival arrival-mobilenet_v2-whole-job-f1-r1 mobilenet_v2 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1
cell arrival arrival-mobilenet_v2-stage-f1-r1 mobilenet_v2 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1
cell arrival arrival-mobilenet_v2-fixed-8-f1.25-r1 mobilenet_v2 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1.25
cell arrival arrival-mobilenet_v2-whole-job-f1.25-r1 mobilenet_v2 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1.25
cell arrival arrival-mobilenet_v2-stage-f1.25-r1 mobilenet_v2 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1.25
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f0.25-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f0.25-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f0.25-r1 nano_gpt_4_layers_64_embd 1 warm stage 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f0.5-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f0.5-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f0.5-r1 nano_gpt_4_layers_64_embd 1 warm stage 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f0.75-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f0.75-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f0.75-r1 nano_gpt_4_layers_64_embd 1 warm stage 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f1-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f1-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f1-r1 nano_gpt_4_layers_64_embd 1 warm stage 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f1.25-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f1.25-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f1.25-r1 nano_gpt_4_layers_64_embd 1 warm stage 64 256 1 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f0.25-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f0.25-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f0.25-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f0.5-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f0.5-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f0.5-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f0.75-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f0.75-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f0.75-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f1-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f1-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f1-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f1.25-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f1.25-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f1.25-r1 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 1 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan-whole-job-f0.25-r2 mnist_gan 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.25
cell arrival arrival-mnist_gan-stage-f0.25-r2 mnist_gan 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.25
cell arrival arrival-mnist_gan-fixed-8-f0.25-r2 mnist_gan 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.25
cell arrival arrival-mnist_gan-whole-job-f0.5-r2 mnist_gan 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.5
cell arrival arrival-mnist_gan-stage-f0.5-r2 mnist_gan 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.5
cell arrival arrival-mnist_gan-fixed-8-f0.5-r2 mnist_gan 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.5
cell arrival arrival-mnist_gan-whole-job-f0.75-r2 mnist_gan 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.75
cell arrival arrival-mnist_gan-stage-f0.75-r2 mnist_gan 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.75
cell arrival arrival-mnist_gan-fixed-8-f0.75-r2 mnist_gan 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.75
cell arrival arrival-mnist_gan-whole-job-f1-r2 mnist_gan 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1
cell arrival arrival-mnist_gan-stage-f1-r2 mnist_gan 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1
cell arrival arrival-mnist_gan-fixed-8-f1-r2 mnist_gan 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1
cell arrival arrival-mnist_gan-whole-job-f1.25-r2 mnist_gan 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1.25
cell arrival arrival-mnist_gan-stage-f1.25-r2 mnist_gan 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1.25
cell arrival arrival-mnist_gan-fixed-8-f1.25-r2 mnist_gan 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1.25
cell arrival arrival-mobilenet_v2-whole-job-f0.25-r2 mobilenet_v2 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.25
cell arrival arrival-mobilenet_v2-stage-f0.25-r2 mobilenet_v2 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.25
cell arrival arrival-mobilenet_v2-fixed-8-f0.25-r2 mobilenet_v2 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.25
cell arrival arrival-mobilenet_v2-whole-job-f0.5-r2 mobilenet_v2 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.5
cell arrival arrival-mobilenet_v2-stage-f0.5-r2 mobilenet_v2 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.5
cell arrival arrival-mobilenet_v2-fixed-8-f0.5-r2 mobilenet_v2 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.5
cell arrival arrival-mobilenet_v2-whole-job-f0.75-r2 mobilenet_v2 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.75
cell arrival arrival-mobilenet_v2-stage-f0.75-r2 mobilenet_v2 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.75
cell arrival arrival-mobilenet_v2-fixed-8-f0.75-r2 mobilenet_v2 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.75
cell arrival arrival-mobilenet_v2-whole-job-f1-r2 mobilenet_v2 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1
cell arrival arrival-mobilenet_v2-stage-f1-r2 mobilenet_v2 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1
cell arrival arrival-mobilenet_v2-fixed-8-f1-r2 mobilenet_v2 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1
cell arrival arrival-mobilenet_v2-whole-job-f1.25-r2 mobilenet_v2 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1.25
cell arrival arrival-mobilenet_v2-stage-f1.25-r2 mobilenet_v2 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1.25
cell arrival arrival-mobilenet_v2-fixed-8-f1.25-r2 mobilenet_v2 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1.25
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f0.25-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f0.25-r2 nano_gpt_4_layers_64_embd 1 warm stage 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f0.25-r2 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f0.5-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f0.5-r2 nano_gpt_4_layers_64_embd 1 warm stage 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f0.5-r2 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f0.75-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f0.75-r2 nano_gpt_4_layers_64_embd 1 warm stage 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f0.75-r2 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f1-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f1-r2 nano_gpt_4_layers_64_embd 1 warm stage 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f1-r2 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f1.25-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f1.25-r2 nano_gpt_4_layers_64_embd 1 warm stage 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f1.25-r2 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f0.25-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f0.25-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f0.25-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f0.5-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f0.5-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f0.5-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f0.75-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f0.75-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f0.75-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f1-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f1-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f1-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f1.25-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f1.25-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f1.25-r2 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 2 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan-stage-f0.25-r3 mnist_gan 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.25
cell arrival arrival-mnist_gan-fixed-8-f0.25-r3 mnist_gan 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.25
cell arrival arrival-mnist_gan-whole-job-f0.25-r3 mnist_gan 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.25
cell arrival arrival-mnist_gan-stage-f0.5-r3 mnist_gan 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.5
cell arrival arrival-mnist_gan-fixed-8-f0.5-r3 mnist_gan 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.5
cell arrival arrival-mnist_gan-whole-job-f0.5-r3 mnist_gan 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.5
cell arrival arrival-mnist_gan-stage-f0.75-r3 mnist_gan 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.75
cell arrival arrival-mnist_gan-fixed-8-f0.75-r3 mnist_gan 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.75
cell arrival arrival-mnist_gan-whole-job-f0.75-r3 mnist_gan 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 0.75
cell arrival arrival-mnist_gan-stage-f1-r3 mnist_gan 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1
cell arrival arrival-mnist_gan-fixed-8-f1-r3 mnist_gan 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1
cell arrival arrival-mnist_gan-whole-job-f1-r3 mnist_gan 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1
cell arrival arrival-mnist_gan-stage-f1.25-r3 mnist_gan 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1.25
cell arrival arrival-mnist_gan-fixed-8-f1.25-r3 mnist_gan 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1.25
cell arrival arrival-mnist_gan-whole-job-f1.25-r3 mnist_gan 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mnist_gan-warm --rate-factor 1.25
cell arrival arrival-mobilenet_v2-stage-f0.25-r3 mobilenet_v2 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.25
cell arrival arrival-mobilenet_v2-fixed-8-f0.25-r3 mobilenet_v2 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.25
cell arrival arrival-mobilenet_v2-whole-job-f0.25-r3 mobilenet_v2 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.25
cell arrival arrival-mobilenet_v2-stage-f0.5-r3 mobilenet_v2 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.5
cell arrival arrival-mobilenet_v2-fixed-8-f0.5-r3 mobilenet_v2 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.5
cell arrival arrival-mobilenet_v2-whole-job-f0.5-r3 mobilenet_v2 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.5
cell arrival arrival-mobilenet_v2-stage-f0.75-r3 mobilenet_v2 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.75
cell arrival arrival-mobilenet_v2-fixed-8-f0.75-r3 mobilenet_v2 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.75
cell arrival arrival-mobilenet_v2-whole-job-f0.75-r3 mobilenet_v2 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 0.75
cell arrival arrival-mobilenet_v2-stage-f1-r3 mobilenet_v2 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1
cell arrival arrival-mobilenet_v2-fixed-8-f1-r3 mobilenet_v2 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1
cell arrival arrival-mobilenet_v2-whole-job-f1-r3 mobilenet_v2 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1
cell arrival arrival-mobilenet_v2-stage-f1.25-r3 mobilenet_v2 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1.25
cell arrival arrival-mobilenet_v2-fixed-8-f1.25-r3 mobilenet_v2 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1.25
cell arrival arrival-mobilenet_v2-whole-job-f1.25-r3 mobilenet_v2 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mobilenet_v2-warm --rate-factor 1.25
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f0.25-r3 nano_gpt_4_layers_64_embd 1 warm stage 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f0.25-r3 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f0.25-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f0.5-r3 nano_gpt_4_layers_64_embd 1 warm stage 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f0.5-r3 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f0.5-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f0.75-r3 nano_gpt_4_layers_64_embd 1 warm stage 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f0.75-r3 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f0.75-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f1-r3 nano_gpt_4_layers_64_embd 1 warm stage 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f1-r3 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f1-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-nano_gpt_4_layers_64_embd-stage-f1.25-r3 nano_gpt_4_layers_64_embd 1 warm stage 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-nano_gpt_4_layers_64_embd-fixed-8-f1.25-r3 nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-nano_gpt_4_layers_64_embd-whole-job-f1.25-r3 nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f0.25-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f0.25-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f0.25-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f0.5-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f0.5-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f0.5-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.5
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f0.75-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f0.75-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f0.75-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 0.75
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f1-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f1-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f1-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-stage-f1.25-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm stage 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-fixed-8-f1.25-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm fixed-8 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1.25
cell arrival arrival-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-whole-job-f1.25-r3 mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd 1 warm whole-job 64 256 3 --arrival poisson --pilot pilot-mnist_gan,mobilenet_v2,nano_gpt_4_layers_64_embd-warm --rate-factor 1.25

# 7. GRANULARITY: new circuits require separate histories and setup.
prep granularity mnist_gan 2 warm
prep granularity mnist_gan 4 warm
prep granularity mnist_gan 8 warm
prep granularity mobilenet_v2 2 warm
prep granularity mobilenet_v2 4 warm
prep granularity mobilenet_v2 8 warm
prep granularity nano_gpt_4_layers_64_embd 2 warm
prep granularity nano_gpt_4_layers_64_embd 4 warm
prep granularity nano_gpt_4_layers_64_embd 8 warm
cell granularity gran-mnist_gan-g2-n1-r1 mnist_gan 2 warm stage 1 256 1
cell granularity gran-mnist_gan-g2-n16-r1 mnist_gan 2 warm stage 16 256 1
cell granularity gran-mnist_gan-g4-n1-r1 mnist_gan 4 warm stage 1 256 1
cell granularity gran-mnist_gan-g4-n16-r1 mnist_gan 4 warm stage 16 256 1
cell granularity gran-mnist_gan-g8-n1-r1 mnist_gan 8 warm stage 1 256 1
cell granularity gran-mnist_gan-g8-n16-r1 mnist_gan 8 warm stage 16 256 1
cell granularity gran-mobilenet_v2-g2-n1-r1 mobilenet_v2 2 warm stage 1 256 1
cell granularity gran-mobilenet_v2-g2-n16-r1 mobilenet_v2 2 warm stage 16 256 1
cell granularity gran-mobilenet_v2-g4-n1-r1 mobilenet_v2 4 warm stage 1 256 1
cell granularity gran-mobilenet_v2-g4-n16-r1 mobilenet_v2 4 warm stage 16 256 1
cell granularity gran-mobilenet_v2-g8-n1-r1 mobilenet_v2 8 warm stage 1 256 1
cell granularity gran-mobilenet_v2-g8-n16-r1 mobilenet_v2 8 warm stage 16 256 1
cell granularity gran-nano_gpt_4_layers_64_embd-g2-n1-r1 nano_gpt_4_layers_64_embd 2 warm stage 1 256 1
cell granularity gran-nano_gpt_4_layers_64_embd-g2-n16-r1 nano_gpt_4_layers_64_embd 2 warm stage 16 256 1
cell granularity gran-nano_gpt_4_layers_64_embd-g4-n1-r1 nano_gpt_4_layers_64_embd 4 warm stage 1 256 1
cell granularity gran-nano_gpt_4_layers_64_embd-g4-n16-r1 nano_gpt_4_layers_64_embd 4 warm stage 16 256 1
cell granularity gran-nano_gpt_4_layers_64_embd-g8-n1-r1 nano_gpt_4_layers_64_embd 8 warm stage 1 256 1
cell granularity gran-nano_gpt_4_layers_64_embd-g8-n16-r1 nano_gpt_4_layers_64_embd 8 warm stage 16 256 1
cell granularity gran-mnist_gan-g2-n1-r2 mnist_gan 2 warm stage 1 256 2
cell granularity gran-mnist_gan-g2-n16-r2 mnist_gan 2 warm stage 16 256 2
cell granularity gran-mnist_gan-g4-n1-r2 mnist_gan 4 warm stage 1 256 2
cell granularity gran-mnist_gan-g4-n16-r2 mnist_gan 4 warm stage 16 256 2
cell granularity gran-mnist_gan-g8-n1-r2 mnist_gan 8 warm stage 1 256 2
cell granularity gran-mnist_gan-g8-n16-r2 mnist_gan 8 warm stage 16 256 2
cell granularity gran-mobilenet_v2-g2-n1-r2 mobilenet_v2 2 warm stage 1 256 2
cell granularity gran-mobilenet_v2-g2-n16-r2 mobilenet_v2 2 warm stage 16 256 2
cell granularity gran-mobilenet_v2-g4-n1-r2 mobilenet_v2 4 warm stage 1 256 2
cell granularity gran-mobilenet_v2-g4-n16-r2 mobilenet_v2 4 warm stage 16 256 2
cell granularity gran-mobilenet_v2-g8-n1-r2 mobilenet_v2 8 warm stage 1 256 2
cell granularity gran-mobilenet_v2-g8-n16-r2 mobilenet_v2 8 warm stage 16 256 2
cell granularity gran-nano_gpt_4_layers_64_embd-g2-n1-r2 nano_gpt_4_layers_64_embd 2 warm stage 1 256 2
cell granularity gran-nano_gpt_4_layers_64_embd-g2-n16-r2 nano_gpt_4_layers_64_embd 2 warm stage 16 256 2
cell granularity gran-nano_gpt_4_layers_64_embd-g4-n1-r2 nano_gpt_4_layers_64_embd 4 warm stage 1 256 2
cell granularity gran-nano_gpt_4_layers_64_embd-g4-n16-r2 nano_gpt_4_layers_64_embd 4 warm stage 16 256 2
cell granularity gran-nano_gpt_4_layers_64_embd-g8-n1-r2 nano_gpt_4_layers_64_embd 8 warm stage 1 256 2
cell granularity gran-nano_gpt_4_layers_64_embd-g8-n16-r2 nano_gpt_4_layers_64_embd 8 warm stage 16 256 2
cell granularity gran-mnist_gan-g2-n1-r3 mnist_gan 2 warm stage 1 256 3
cell granularity gran-mnist_gan-g2-n16-r3 mnist_gan 2 warm stage 16 256 3
cell granularity gran-mnist_gan-g4-n1-r3 mnist_gan 4 warm stage 1 256 3
cell granularity gran-mnist_gan-g4-n16-r3 mnist_gan 4 warm stage 16 256 3
cell granularity gran-mnist_gan-g8-n1-r3 mnist_gan 8 warm stage 1 256 3
cell granularity gran-mnist_gan-g8-n16-r3 mnist_gan 8 warm stage 16 256 3
cell granularity gran-mobilenet_v2-g2-n1-r3 mobilenet_v2 2 warm stage 1 256 3
cell granularity gran-mobilenet_v2-g2-n16-r3 mobilenet_v2 2 warm stage 16 256 3
cell granularity gran-mobilenet_v2-g4-n1-r3 mobilenet_v2 4 warm stage 1 256 3
cell granularity gran-mobilenet_v2-g4-n16-r3 mobilenet_v2 4 warm stage 16 256 3
cell granularity gran-mobilenet_v2-g8-n1-r3 mobilenet_v2 8 warm stage 1 256 3
cell granularity gran-mobilenet_v2-g8-n16-r3 mobilenet_v2 8 warm stage 16 256 3
cell granularity gran-nano_gpt_4_layers_64_embd-g2-n1-r3 nano_gpt_4_layers_64_embd 2 warm stage 1 256 3
cell granularity gran-nano_gpt_4_layers_64_embd-g2-n16-r3 nano_gpt_4_layers_64_embd 2 warm stage 16 256 3
cell granularity gran-nano_gpt_4_layers_64_embd-g4-n1-r3 nano_gpt_4_layers_64_embd 4 warm stage 1 256 3
cell granularity gran-nano_gpt_4_layers_64_embd-g4-n16-r3 nano_gpt_4_layers_64_embd 4 warm stage 16 256 3
cell granularity gran-nano_gpt_4_layers_64_embd-g8-n1-r3 nano_gpt_4_layers_64_embd 8 warm stage 1 256 3
cell granularity gran-nano_gpt_4_layers_64_embd-g8-n16-r3 nano_gpt_4_layers_64_embd 8 warm stage 16 256 3
# g=1 uses serving runs; isolated profile reports provide summed job time/peak memory.

# 8. DECOMPOSITION: costly. g=0 prepares monolithic profiles.
# 700 GiB budget assumes the 755 GiB host. A monolithic OOM is a result:
# preserve logs, comment out that task if needed, then resume.
prep decomposition mnist_gan 0 cold
prep decomposition mnist_gan 0 warm
prep decomposition mobilenet_v2 0 cold
prep decomposition mobilenet_v2 0 warm
prep decomposition nano_gpt_4_layers_64_embd 0 cold
prep decomposition nano_gpt_4_layers_64_embd 0 warm
cell decomposition decomp-mnist_gan-cold-monolithic-r1 mnist_gan 0 cold monolithic 1 700 1
cell decomposition decomp-mnist_gan-cold-partitioned-sequential-r1 mnist_gan 1 cold partitioned-sequential 1 700 1
cell decomposition decomp-mnist_gan-warm-monolithic-r1 mnist_gan 0 warm monolithic 1 700 1
cell decomposition decomp-mnist_gan-warm-partitioned-sequential-r1 mnist_gan 1 warm partitioned-sequential 1 700 1
cell decomposition decomp-mobilenet_v2-cold-monolithic-r1 mobilenet_v2 0 cold monolithic 1 700 1
cell decomposition decomp-mobilenet_v2-cold-partitioned-sequential-r1 mobilenet_v2 1 cold partitioned-sequential 1 700 1
cell decomposition decomp-mobilenet_v2-warm-monolithic-r1 mobilenet_v2 0 warm monolithic 1 700 1
cell decomposition decomp-mobilenet_v2-warm-partitioned-sequential-r1 mobilenet_v2 1 warm partitioned-sequential 1 700 1
cell decomposition decomp-nano_gpt_4_layers_64_embd-cold-monolithic-r1 nano_gpt_4_layers_64_embd 0 cold monolithic 1 700 1
cell decomposition decomp-nano_gpt_4_layers_64_embd-cold-partitioned-sequential-r1 nano_gpt_4_layers_64_embd 1 cold partitioned-sequential 1 700 1
cell decomposition decomp-nano_gpt_4_layers_64_embd-warm-monolithic-r1 nano_gpt_4_layers_64_embd 0 warm monolithic 1 700 1
cell decomposition decomp-nano_gpt_4_layers_64_embd-warm-partitioned-sequential-r1 nano_gpt_4_layers_64_embd 1 warm partitioned-sequential 1 700 1
cell decomposition decomp-mnist_gan-cold-monolithic-r2 mnist_gan 0 cold monolithic 1 700 2
cell decomposition decomp-mnist_gan-cold-partitioned-sequential-r2 mnist_gan 1 cold partitioned-sequential 1 700 2
cell decomposition decomp-mnist_gan-warm-monolithic-r2 mnist_gan 0 warm monolithic 1 700 2
cell decomposition decomp-mnist_gan-warm-partitioned-sequential-r2 mnist_gan 1 warm partitioned-sequential 1 700 2
cell decomposition decomp-mobilenet_v2-cold-monolithic-r2 mobilenet_v2 0 cold monolithic 1 700 2
cell decomposition decomp-mobilenet_v2-cold-partitioned-sequential-r2 mobilenet_v2 1 cold partitioned-sequential 1 700 2
cell decomposition decomp-mobilenet_v2-warm-monolithic-r2 mobilenet_v2 0 warm monolithic 1 700 2
cell decomposition decomp-mobilenet_v2-warm-partitioned-sequential-r2 mobilenet_v2 1 warm partitioned-sequential 1 700 2
cell decomposition decomp-nano_gpt_4_layers_64_embd-cold-monolithic-r2 nano_gpt_4_layers_64_embd 0 cold monolithic 1 700 2
cell decomposition decomp-nano_gpt_4_layers_64_embd-cold-partitioned-sequential-r2 nano_gpt_4_layers_64_embd 1 cold partitioned-sequential 1 700 2
cell decomposition decomp-nano_gpt_4_layers_64_embd-warm-monolithic-r2 nano_gpt_4_layers_64_embd 0 warm monolithic 1 700 2
cell decomposition decomp-nano_gpt_4_layers_64_embd-warm-partitioned-sequential-r2 nano_gpt_4_layers_64_embd 1 warm partitioned-sequential 1 700 2
cell decomposition decomp-mnist_gan-cold-monolithic-r3 mnist_gan 0 cold monolithic 1 700 3
cell decomposition decomp-mnist_gan-cold-partitioned-sequential-r3 mnist_gan 1 cold partitioned-sequential 1 700 3
cell decomposition decomp-mnist_gan-warm-monolithic-r3 mnist_gan 0 warm monolithic 1 700 3
cell decomposition decomp-mnist_gan-warm-partitioned-sequential-r3 mnist_gan 1 warm partitioned-sequential 1 700 3
cell decomposition decomp-mobilenet_v2-cold-monolithic-r3 mobilenet_v2 0 cold monolithic 1 700 3
cell decomposition decomp-mobilenet_v2-cold-partitioned-sequential-r3 mobilenet_v2 1 cold partitioned-sequential 1 700 3
cell decomposition decomp-mobilenet_v2-warm-monolithic-r3 mobilenet_v2 0 warm monolithic 1 700 3
cell decomposition decomp-mobilenet_v2-warm-partitioned-sequential-r3 mobilenet_v2 1 warm partitioned-sequential 1 700 3
cell decomposition decomp-nano_gpt_4_layers_64_embd-cold-monolithic-r3 nano_gpt_4_layers_64_embd 0 cold monolithic 1 700 3
cell decomposition decomp-nano_gpt_4_layers_64_embd-cold-partitioned-sequential-r3 nano_gpt_4_layers_64_embd 1 cold partitioned-sequential 1 700 3
cell decomposition decomp-nano_gpt_4_layers_64_embd-warm-monolithic-r3 nano_gpt_4_layers_64_embd 0 warm monolithic 1 700 3
cell decomposition decomp-nano_gpt_4_layers_64_embd-warm-partitioned-sequential-r3 nano_gpt_4_layers_64_embd 1 warm partitioned-sequential 1 700 3

# Optional larger model. Uncomment profiling and corresponding measured cells together.
# prep supplementary nano_gpt_10_layers_64_embd 1 cold
# prep supplementary nano_gpt_10_layers_64_embd 1 warm
# cell supplementary nano10-stage-n1-r1 nano_gpt_10_layers_64_embd 1 warm stage 1 256 1
# TinyBERT has no YAML on this branch: add the correct model/input config first.
# AWS scale-out: use evaluation/run_cluster.sh on an explicitly prepared deployment.
# Recovery and second-backend tests are not simulated by this checklist.
