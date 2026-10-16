#!/usr/bin/env bash
# Editable experiment checklist. Comment out any prep/cell line below.
# Default is PLAN ONLY. See evaluation/setup.md before --execute.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
EVAL_ROOT="${EVAL_ROOT:-$PWD/evaluation/results/paper-day-v2}"
SECTION="${SECTION:-all}"
ONLY="${ONLY:-*}"
TIMEOUT="${TIMEOUT:-21600}"
CAMPAIGN_SECONDS="${CAMPAIGN_SECONDS:-82800}" # 23 hours per invocation; 0 disables
campaign_started=$SECONDS
PROFILE_REPEATS="${PROFILE_REPEATS:-1}"
CACHE_GIB="${CACHE_GIB:-500}"
SAMPLING="${SAMPLING:-0.1}"
PORT="${PORT:-50051}"
DISCARD_COPIES="${DISCARD_COPIES:-1}"
PRUNE_RUN_SETUP="${PRUNE_RUN_SETUP:-1}" # after successful verification, outside timing
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
  if [[ "$MODE" == --execute && "$CAMPAIGN_SECONDS" -gt 0 ]]; then
    local remaining=$((CAMPAIGN_SECONDS - SECONDS + campaign_started))
    if (( remaining <= 0 )); then echo "Campaign time budget reached; resume later." >&2; exit 124; fi
    timeout --foreground --signal=INT --kill-after=45s "${remaining}s" "$PYTHON" -u evaluation/campaign.py "$@" "${options[@]}"
  else
    "$PYTHON" -u evaluation/campaign.py "$@" "${options[@]}"
  fi
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
  if [[ "$MODE" == --execute && "$PRUNE_RUN_SETUP" == 1 ]]; then
    "$PYTHON" evaluation/maintenance/prune_run_setup.py --root "$EVAL_ROOT" --run-id "$id" --execute
  fi
}
# Columns: group ID workload granularity setup baseline requests budget_GiB trial
# Each line is one resumable task. DONE.json is the automatic checkmark.
# Targeted pass: one measured trial; no error bars or repeatability claims.
# Stop on first failure; failed attempts remain on disk. Re-running retries them.



# TARGETED WARM SERVING: finish each model before preparing the next.
# One setup-only bootstrap + ONE full warm profiling request per model.
# No cold primer. No bootstrap proof or witness. N=1, 4, 8 and 16; one measured trial at 128 GiB.
# MobileNetV2 is last because its earlier runs were much longer; report all models
# actually completed, and do not select models based on whether they show gains.

prep profiles mnist_gan 1 warm
cell serving burst-mnist_gan-warm-whole-job-n1-b128-r1 mnist_gan 1 warm whole-job 1 128 1
cell serving burst-mnist_gan-warm-stage-n1-b128-r1 mnist_gan 1 warm stage 1 128 1
cell serving burst-mnist_gan-warm-fixed-8-n1-b128-r1 mnist_gan 1 warm fixed-8 1 128 1
cell serving burst-mnist_gan-warm-whole-job-n4-b128-r1 mnist_gan 1 warm whole-job 4 128 1
cell serving burst-mnist_gan-warm-whole-job-n8-b128-r1 mnist_gan 1 warm whole-job 8 128 1
cell serving burst-mnist_gan-warm-whole-job-n16-b128-r1 mnist_gan 1 warm whole-job 16 128 1
cell serving burst-mnist_gan-warm-stage-n4-b128-r1 mnist_gan 1 warm stage 4 128 1
cell serving burst-mnist_gan-warm-stage-n8-b128-r1 mnist_gan 1 warm stage 8 128 1
cell serving burst-mnist_gan-warm-stage-n16-b128-r1 mnist_gan 1 warm stage 16 128 1
cell serving burst-mnist_gan-warm-fixed-8-n4-b128-r1 mnist_gan 1 warm fixed-8 4 128 1
cell serving burst-mnist_gan-warm-fixed-8-n8-b128-r1 mnist_gan 1 warm fixed-8 8 128 1
cell serving burst-mnist_gan-warm-fixed-8-n16-b128-r1 mnist_gan 1 warm fixed-8 16 128 1

prep profiles nano_gpt_4_layers_64_embd 1 warm
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n1-b128-r1 nano_gpt_4_layers_64_embd 1 warm stage 1 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n1-b128-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 1 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n1-b128-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 1 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n4-b128-r1 nano_gpt_4_layers_64_embd 1 warm stage 4 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n8-b128-r1 nano_gpt_4_layers_64_embd 1 warm stage 8 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-stage-n16-b128-r1 nano_gpt_4_layers_64_embd 1 warm stage 16 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n4-b128-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 4 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n8-b128-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 8 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n16-b128-r1 nano_gpt_4_layers_64_embd 1 warm fixed-8 16 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n4-b128-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 4 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n8-b128-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 8 128 1
cell serving burst-nano_gpt_4_layers_64_embd-warm-whole-job-n16-b128-r1 nano_gpt_4_layers_64_embd 1 warm whole-job 16 128 1

# TinyBERT uses the bert.yaml configuration stem; its model name is tiny_bert.
prep profiles bert 1 warm
cell serving burst-bert-warm-whole-job-n1-b128-r1 bert 1 warm whole-job 1 128 1
cell serving burst-bert-warm-stage-n1-b128-r1 bert 1 warm stage 1 128 1
cell serving burst-bert-warm-fixed-8-n1-b128-r1 bert 1 warm fixed-8 1 128 1
cell serving burst-bert-warm-whole-job-n4-b128-r1 bert 1 warm whole-job 4 128 1
cell serving burst-bert-warm-whole-job-n8-b128-r1 bert 1 warm whole-job 8 128 1
cell serving burst-bert-warm-whole-job-n16-b128-r1 bert 1 warm whole-job 16 128 1
cell serving burst-bert-warm-stage-n4-b128-r1 bert 1 warm stage 4 128 1
cell serving burst-bert-warm-stage-n8-b128-r1 bert 1 warm stage 8 128 1
cell serving burst-bert-warm-stage-n16-b128-r1 bert 1 warm stage 16 128 1
cell serving burst-bert-warm-fixed-8-n4-b128-r1 bert 1 warm fixed-8 4 128 1
cell serving burst-bert-warm-fixed-8-n8-b128-r1 bert 1 warm fixed-8 8 128 1
cell serving burst-bert-warm-fixed-8-n16-b128-r1 bert 1 warm fixed-8 16 128 1

prep profiles mobilenet_v2 1 warm
cell serving burst-mobilenet_v2-warm-fixed-8-n1-b128-r1 mobilenet_v2 1 warm fixed-8 1 128 1
cell serving burst-mobilenet_v2-warm-whole-job-n1-b128-r1 mobilenet_v2 1 warm whole-job 1 128 1
cell serving burst-mobilenet_v2-warm-stage-n1-b128-r1 mobilenet_v2 1 warm stage 1 128 1
cell serving burst-mobilenet_v2-warm-fixed-8-n4-b128-r1 mobilenet_v2 1 warm fixed-8 4 128 1
cell serving burst-mobilenet_v2-warm-fixed-8-n8-b128-r1 mobilenet_v2 1 warm fixed-8 8 128 1
cell serving burst-mobilenet_v2-warm-fixed-8-n16-b128-r1 mobilenet_v2 1 warm fixed-8 16 128 1
cell serving burst-mobilenet_v2-warm-whole-job-n4-b128-r1 mobilenet_v2 1 warm whole-job 4 128 1
cell serving burst-mobilenet_v2-warm-whole-job-n8-b128-r1 mobilenet_v2 1 warm whole-job 8 128 1
cell serving burst-mobilenet_v2-warm-whole-job-n16-b128-r1 mobilenet_v2 1 warm whole-job 16 128 1
cell serving burst-mobilenet_v2-warm-stage-n4-b128-r1 mobilenet_v2 1 warm stage 4 128 1
cell serving burst-mobilenet_v2-warm-stage-n8-b128-r1 mobilenet_v2 1 warm stage 8 128 1
cell serving burst-mobilenet_v2-warm-stage-n16-b128-r1 mobilenet_v2 1 warm stage 16 128 1

# OPTIONAL: enable only after the primary warm comparison finishes.
# Cold profiles reuse the SAME bootstrap, but run full forced-cold proving once.
# prep profiles mnist_gan 1 cold
# cell cold burst-mnist_gan-cold-whole-job-n4-b128-r1 mnist_gan 1 cold whole-job 4 128 1
# cell cold burst-mnist_gan-cold-stage-n4-b128-r1 mnist_gan 1 cold stage 4 128 1
# cell memory memory-mnist_gan-warm-whole-job-b128-r1 mnist_gan 1 warm whole-job 4 128 1
# cell memory memory-mnist_gan-warm-stage-b128-r1 mnist_gan 1 warm stage 4 128 1
# cell scheduling fixed1-mnist-b128-r1 mnist_gan 1 warm fixed-1 4 128 1
# cell scheduling fixed4-mnist-b128-r1 mnist_gan 1 warm fixed-4 4 128 1
# cell scheduling fixed16-mnist-b128-r1 mnist_gan 1 warm fixed-16 4 128 1
# OPTIONAL REPLICATION: same N=4 cells with trial 2, then trial 3 if time remains.
# cell repeats burst-mnist_gan-warm-stage-n4-b128-r2 mnist_gan 1 warm stage 4 128 2
# cell repeats burst-mnist_gan-warm-fixed-8-n4-b128-r2 mnist_gan 1 warm fixed-8 4 128 2
# cell repeats burst-mnist_gan-warm-whole-job-n4-b128-r2 mnist_gan 1 warm whole-job 4 128 2
# cell repeats burst-nano_gpt_4_layers_64_embd-warm-stage-n4-b128-r2 nano_gpt_4_layers_64_embd 1 warm stage 4 128 2
# cell repeats burst-nano_gpt_4_layers_64_embd-warm-fixed-8-n4-b128-r2 nano_gpt_4_layers_64_embd 1 warm fixed-8 4 128 2
# cell repeats burst-nano_gpt_4_layers_64_embd-warm-whole-job-n4-b128-r2 nano_gpt_4_layers_64_embd 1 warm whole-job 4 128 2
# cell repeats burst-bert-warm-stage-n4-b128-r2 bert 1 warm stage 4 128 2
# cell repeats burst-bert-warm-fixed-8-n4-b128-r2 bert 1 warm fixed-8 4 128 2
# cell repeats burst-bert-warm-whole-job-n4-b128-r2 bert 1 warm whole-job 4 128 2
# cell repeats burst-mobilenet_v2-warm-stage-n4-b128-r2 mobilenet_v2 1 warm stage 4 128 2
# cell repeats burst-mobilenet_v2-warm-fixed-8-n4-b128-r2 mobilenet_v2 1 warm fixed-8 4 128 2
# cell repeats burst-mobilenet_v2-warm-whole-job-n4-b128-r2 mobilenet_v2 1 warm whole-job 4 128 2

# Exhaustive Poisson/mixed/scaling/granularity sweeps: run_full.sh / run_cluster.sh.
# They are intentionally NOT part of the one-day default.

