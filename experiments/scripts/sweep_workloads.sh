#!/usr/bin/env bash
set -euo pipefail

LOCAL_CONCURRENCY="${1:-1}"
HEAVY_PHASE_LIMIT="${2:-${LOCAL_CONCURRENCY}}"
COORDINATOR_HOST="${3:-127.0.0.1}"
COORDINATOR_PORT="${4:-50051}"

WORKLOADS=(
  mnist_gan
  # mnist_classifier
  # mobilenet_v2
  # nano_gpt_4_layers_64_embd
)

for WORKLOAD in "${WORKLOADS[@]}"; do
  echo "Running workload: ${WORKLOAD}"
  python experiments/run_local_exp.py \
    "suite.workloads=[${WORKLOAD}]" \
    "node.local_concurrency_limit=${LOCAL_CONCURRENCY}" \
    "node.max_heavy_phases=${HEAVY_PHASE_LIMIT}" \
    "launch.coordinator_host=${COORDINATOR_HOST}" \
    "launch.coordinator_port=${COORDINATOR_PORT}"
  echo "Finished workload: ${WORKLOAD}"
done
