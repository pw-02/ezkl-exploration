#!/usr/bin/env bash
set -euo pipefail

PHASE_ADMISSION="${1:-true}"
COORDINATOR_HOST="${2:-127.0.0.1}"
COORDINATOR_PORT="${3:-50051}"

if [[ "$PHASE_ADMISSION" != "true" && "$PHASE_ADMISSION" != "false" ]]; then
  echo "Error: phase_admission must be true or false."
  exit 1
fi

WORKLOADS=(
  mnist_gan
  # mnist_classifier
  # mobilenet_v2
  # nano_gpt_4_layers_64_embd
)

for WORKLOAD in "${WORKLOADS[@]}"; do
  echo "Running workload: ${WORKLOAD} admission=${PHASE_ADMISSION}"
  python experiments/run_local_exp.py \
    "suite.workloads=[${WORKLOAD}]" \
    "node.phase_admission.enabled=${PHASE_ADMISSION}" \
    "launch.coordinator_host=${COORDINATOR_HOST}" \
    "launch.coordinator_port=${COORDINATOR_PORT}"
  echo "Finished workload: ${WORKLOAD}"
done
