#!/usr/bin/env bash
# Physical AWS scale-out checklist; deployment hooks are intentionally explicit.
# They must deploy/reset actual nodes, not start multiple local supervisors.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
EVAL_ROOT="${EVAL_ROOT:-$PWD/evaluation/results/aws-v1}"
ONLY="${ONLY:-*}"
MODE="${1:---plan}"
[[ "$MODE" == --plan || "$MODE" == --execute ]] || exit 2
cluster() {
  local model="$1" baseline="$2" nodes="$3" requests="$4" trial="$5"
  local id="$model-$baseline-nodes$nodes-requests$requests-r$trial"
  [[ "$id" == $ONLY ]] || return 0
  if [[ "$MODE" == --plan ]]; then echo "PLAN cluster $id"; return 0; fi
  : "${CLUSTER_TARGET:?Set host:port for the coordinator}"
  : "${CLUSTER_PREPARE:?Set executable hook to deploy/reset the requested physical cluster}"
  : "${CLUSTER_STOP:?Set executable hook to stop only this trial runtime}"
  [[ -x "$CLUSTER_PREPARE" && -x "$CLUSTER_STOP" ]] || { echo 'Hooks must be executable' >&2; return 2; }
  local dir="$EVAL_ROOT/$id" attempt=1
  if [[ -f "$dir/DONE" ]]; then echo "SKIP $id"; return 0; fi
  while [[ -e "$dir/attempt-$attempt" ]]; do attempt=$((attempt+1)); done
  dir="$dir/attempt-$attempt"; mkdir -p "$dir"
  # Hook interface: baseline node_count workload trial output_directory.
  # Preparation must return only when all requested nodes are registered and ready.
  # See setup.md for required profile/artifact resets and metadata.
  (
    trap '"$CLUSTER_STOP" "$baseline" "$nodes" "$model" "$trial" "$dir"' EXIT
    "$CLUSTER_PREPARE" "$baseline" "$nodes" "$model" "$trial" "$dir"
    [[ -s "$dir/deployment.json" ]] || { echo 'Prepare hook must record deployment.json' >&2; exit 2; }
    "$PYTHON" -u evaluation/traffic.py --target "$CLUSTER_TARGET" --workload "$model" \
      --requests "$requests" --seed "$((4100+trial))" --timeout "${TIMEOUT:-172800}" --output "$dir/traffic"
  ) > >(tee "$dir/console.log") 2>&1
  touch "$EVAL_ROOT/$id/DONE"
}
# Explicit list below; comment out completed/unwanted cells.
cluster mnist_gan whole-job 1 1 1
cluster mnist_gan stage 1 1 1
cluster mnist_gan whole-job 1 16 1
cluster mnist_gan stage 1 16 1
cluster mnist_gan whole-job 2 1 1
cluster mnist_gan stage 2 1 1
cluster mnist_gan whole-job 2 16 1
cluster mnist_gan stage 2 16 1
cluster mnist_gan whole-job 4 1 1
cluster mnist_gan stage 4 1 1
cluster mnist_gan whole-job 4 16 1
cluster mnist_gan stage 4 16 1
cluster mnist_gan whole-job 8 1 1
cluster mnist_gan stage 8 1 1
cluster mnist_gan whole-job 8 16 1
cluster mnist_gan stage 8 16 1
cluster mnist_gan whole-job 10 1 1
cluster mnist_gan stage 10 1 1
cluster mnist_gan whole-job 10 16 1
cluster mnist_gan stage 10 16 1
cluster mobilenet_v2 whole-job 1 1 1
cluster mobilenet_v2 stage 1 1 1
cluster mobilenet_v2 whole-job 1 16 1
cluster mobilenet_v2 stage 1 16 1
cluster mobilenet_v2 whole-job 2 1 1
cluster mobilenet_v2 stage 2 1 1
cluster mobilenet_v2 whole-job 2 16 1
cluster mobilenet_v2 stage 2 16 1
cluster mobilenet_v2 whole-job 4 1 1
cluster mobilenet_v2 stage 4 1 1
cluster mobilenet_v2 whole-job 4 16 1
cluster mobilenet_v2 stage 4 16 1
cluster mobilenet_v2 whole-job 8 1 1
cluster mobilenet_v2 stage 8 1 1
cluster mobilenet_v2 whole-job 8 16 1
cluster mobilenet_v2 stage 8 16 1
cluster mobilenet_v2 whole-job 10 1 1
cluster mobilenet_v2 stage 10 1 1
cluster mobilenet_v2 whole-job 10 16 1
cluster mobilenet_v2 stage 10 16 1
cluster nano_gpt_4_layers_64_embd whole-job 1 1 1
cluster nano_gpt_4_layers_64_embd stage 1 1 1
cluster nano_gpt_4_layers_64_embd whole-job 1 16 1
cluster nano_gpt_4_layers_64_embd stage 1 16 1
cluster nano_gpt_4_layers_64_embd whole-job 2 1 1
cluster nano_gpt_4_layers_64_embd stage 2 1 1
cluster nano_gpt_4_layers_64_embd whole-job 2 16 1
cluster nano_gpt_4_layers_64_embd stage 2 16 1
cluster nano_gpt_4_layers_64_embd whole-job 4 1 1
cluster nano_gpt_4_layers_64_embd stage 4 1 1
cluster nano_gpt_4_layers_64_embd whole-job 4 16 1
cluster nano_gpt_4_layers_64_embd stage 4 16 1
cluster nano_gpt_4_layers_64_embd whole-job 8 1 1
cluster nano_gpt_4_layers_64_embd stage 8 1 1
cluster nano_gpt_4_layers_64_embd whole-job 8 16 1
cluster nano_gpt_4_layers_64_embd stage 8 16 1
cluster nano_gpt_4_layers_64_embd whole-job 10 1 1
cluster nano_gpt_4_layers_64_embd stage 10 1 1
cluster nano_gpt_4_layers_64_embd whole-job 10 16 1
cluster nano_gpt_4_layers_64_embd stage 10 16 1
cluster mnist_gan stage 1 1 2
cluster mnist_gan whole-job 1 1 2
cluster mnist_gan stage 1 16 2
cluster mnist_gan whole-job 1 16 2
cluster mnist_gan stage 2 1 2
cluster mnist_gan whole-job 2 1 2
cluster mnist_gan stage 2 16 2
cluster mnist_gan whole-job 2 16 2
cluster mnist_gan stage 4 1 2
cluster mnist_gan whole-job 4 1 2
cluster mnist_gan stage 4 16 2
cluster mnist_gan whole-job 4 16 2
cluster mnist_gan stage 8 1 2
cluster mnist_gan whole-job 8 1 2
cluster mnist_gan stage 8 16 2
cluster mnist_gan whole-job 8 16 2
cluster mnist_gan stage 10 1 2
cluster mnist_gan whole-job 10 1 2
cluster mnist_gan stage 10 16 2
cluster mnist_gan whole-job 10 16 2
cluster mobilenet_v2 stage 1 1 2
cluster mobilenet_v2 whole-job 1 1 2
cluster mobilenet_v2 stage 1 16 2
cluster mobilenet_v2 whole-job 1 16 2
cluster mobilenet_v2 stage 2 1 2
cluster mobilenet_v2 whole-job 2 1 2
cluster mobilenet_v2 stage 2 16 2
cluster mobilenet_v2 whole-job 2 16 2
cluster mobilenet_v2 stage 4 1 2
cluster mobilenet_v2 whole-job 4 1 2
cluster mobilenet_v2 stage 4 16 2
cluster mobilenet_v2 whole-job 4 16 2
cluster mobilenet_v2 stage 8 1 2
cluster mobilenet_v2 whole-job 8 1 2
cluster mobilenet_v2 stage 8 16 2
cluster mobilenet_v2 whole-job 8 16 2
cluster mobilenet_v2 stage 10 1 2
cluster mobilenet_v2 whole-job 10 1 2
cluster mobilenet_v2 stage 10 16 2
cluster mobilenet_v2 whole-job 10 16 2
cluster nano_gpt_4_layers_64_embd stage 1 1 2
cluster nano_gpt_4_layers_64_embd whole-job 1 1 2
cluster nano_gpt_4_layers_64_embd stage 1 16 2
cluster nano_gpt_4_layers_64_embd whole-job 1 16 2
cluster nano_gpt_4_layers_64_embd stage 2 1 2
cluster nano_gpt_4_layers_64_embd whole-job 2 1 2
cluster nano_gpt_4_layers_64_embd stage 2 16 2
cluster nano_gpt_4_layers_64_embd whole-job 2 16 2
cluster nano_gpt_4_layers_64_embd stage 4 1 2
cluster nano_gpt_4_layers_64_embd whole-job 4 1 2
cluster nano_gpt_4_layers_64_embd stage 4 16 2
cluster nano_gpt_4_layers_64_embd whole-job 4 16 2
cluster nano_gpt_4_layers_64_embd stage 8 1 2
cluster nano_gpt_4_layers_64_embd whole-job 8 1 2
cluster nano_gpt_4_layers_64_embd stage 8 16 2
cluster nano_gpt_4_layers_64_embd whole-job 8 16 2
cluster nano_gpt_4_layers_64_embd stage 10 1 2
cluster nano_gpt_4_layers_64_embd whole-job 10 1 2
cluster nano_gpt_4_layers_64_embd stage 10 16 2
cluster nano_gpt_4_layers_64_embd whole-job 10 16 2
cluster mnist_gan whole-job 1 1 3
cluster mnist_gan stage 1 1 3
cluster mnist_gan whole-job 1 16 3
cluster mnist_gan stage 1 16 3
cluster mnist_gan whole-job 2 1 3
cluster mnist_gan stage 2 1 3
cluster mnist_gan whole-job 2 16 3
cluster mnist_gan stage 2 16 3
cluster mnist_gan whole-job 4 1 3
cluster mnist_gan stage 4 1 3
cluster mnist_gan whole-job 4 16 3
cluster mnist_gan stage 4 16 3
cluster mnist_gan whole-job 8 1 3
cluster mnist_gan stage 8 1 3
cluster mnist_gan whole-job 8 16 3
cluster mnist_gan stage 8 16 3
cluster mnist_gan whole-job 10 1 3
cluster mnist_gan stage 10 1 3
cluster mnist_gan whole-job 10 16 3
cluster mnist_gan stage 10 16 3
cluster mobilenet_v2 whole-job 1 1 3
cluster mobilenet_v2 stage 1 1 3
cluster mobilenet_v2 whole-job 1 16 3
cluster mobilenet_v2 stage 1 16 3
cluster mobilenet_v2 whole-job 2 1 3
cluster mobilenet_v2 stage 2 1 3
cluster mobilenet_v2 whole-job 2 16 3
cluster mobilenet_v2 stage 2 16 3
cluster mobilenet_v2 whole-job 4 1 3
cluster mobilenet_v2 stage 4 1 3
cluster mobilenet_v2 whole-job 4 16 3
cluster mobilenet_v2 stage 4 16 3
cluster mobilenet_v2 whole-job 8 1 3
cluster mobilenet_v2 stage 8 1 3
cluster mobilenet_v2 whole-job 8 16 3
cluster mobilenet_v2 stage 8 16 3
cluster mobilenet_v2 whole-job 10 1 3
cluster mobilenet_v2 stage 10 1 3
cluster mobilenet_v2 whole-job 10 16 3
cluster mobilenet_v2 stage 10 16 3
cluster nano_gpt_4_layers_64_embd whole-job 1 1 3
cluster nano_gpt_4_layers_64_embd stage 1 1 3
cluster nano_gpt_4_layers_64_embd whole-job 1 16 3
cluster nano_gpt_4_layers_64_embd stage 1 16 3
cluster nano_gpt_4_layers_64_embd whole-job 2 1 3
cluster nano_gpt_4_layers_64_embd stage 2 1 3
cluster nano_gpt_4_layers_64_embd whole-job 2 16 3
cluster nano_gpt_4_layers_64_embd stage 2 16 3
cluster nano_gpt_4_layers_64_embd whole-job 4 1 3
cluster nano_gpt_4_layers_64_embd stage 4 1 3
cluster nano_gpt_4_layers_64_embd whole-job 4 16 3
cluster nano_gpt_4_layers_64_embd stage 4 16 3
cluster nano_gpt_4_layers_64_embd whole-job 8 1 3
cluster nano_gpt_4_layers_64_embd stage 8 1 3
cluster nano_gpt_4_layers_64_embd whole-job 8 16 3
cluster nano_gpt_4_layers_64_embd stage 8 16 3
cluster nano_gpt_4_layers_64_embd whole-job 10 1 3
cluster nano_gpt_4_layers_64_embd stage 10 1 3
cluster nano_gpt_4_layers_64_embd whole-job 10 16 3
cluster nano_gpt_4_layers_64_embd stage 10 16 3
