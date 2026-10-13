#!/usr/bin/env bash
set -euo pipefail

STAGE_ADMISSION="${1:-true}"
COORDINATOR_HOST="${2:-127.0.0.1}"
COORDINATOR_PORT="${3:-50051}"

if [[ "$STAGE_ADMISSION" != "true" && "$STAGE_ADMISSION" != "false" ]]; then
  echo "Error: stage admission must be true or false."
  exit 1
fi

WORKLOADS=(
  mnist_gan
  # mnist_classifier
  # mobilenet_v2
  # nano_gpt_4_layers_64_embd
)

for WORKLOAD in "${WORKLOADS[@]}"; do
  echo "Running workload: ${WORKLOAD} admission=${STAGE_ADMISSION}"
  python experiments/run_local_exp.py \
    "suite.workloads=[${WORKLOAD}]" \
    "node.stage_admission.enabled=${STAGE_ADMISSION}" \
    "launch.coordinator_host=${COORDINATOR_HOST}" \
    "launch.coordinator_port=${COORDINATOR_PORT}"
  echo "Finished workload: ${WORKLOAD}"
done
