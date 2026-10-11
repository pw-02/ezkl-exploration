#!/usr/bin/env bash

set -euo pipefail

WORKLOAD="${1:-mnist_gan}"
COORDINATOR_HOST="${2:-127.0.0.1}"
COORDINATOR_PORT="${3:-50051}"
PHASE_ADMISSION="${4:-true}"

if [[ "$PHASE_ADMISSION" != "true" && "$PHASE_ADMISSION" != "false" ]]; then
    echo "Error: phase_admission must be true or false."
    exit 1
fi

SESSION="${TMUX_SESSION:-zkexp}"
CONDA_ENV="${CONDA_ENV:-zk}"
ROOT_DIR="$(pwd)"
ADMISSION_TAG="admission-${PHASE_ADMISSION}"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_${WORKLOAD}_${ADMISSION_TAG}"
RUN_DIR="${ROOT_DIR}/experiments/runs/${RUN_ID}"
LOGS_DIR="${RUN_DIR}/logs"
SHARED_DIR="${RUN_DIR}/shared"
CACHE_DIR="${RUN_DIR}/cache"
NODE_CACHE_DIR="${RUN_DIR}/node-cache"
mkdir -p "$LOGS_DIR" "$SHARED_DIR" "$CACHE_DIR" "$NODE_CACHE_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
fi

BASE_CMD="\
cd ${ROOT_DIR} && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate ${CONDA_ENV} && \
export PYTHONPATH=${ROOT_DIR}:\${PYTHONPATH:-}"

RUNTIME_CFG="\
coordinator.host=${COORDINATOR_HOST} \
coordinator.port=${COORDINATOR_PORT} \
coordinator.runs_dir=${RUN_DIR} \
coordinator.logs_dir=${LOGS_DIR} \
node.runs_dir=${RUN_DIR} \
node.logs_dir=${LOGS_DIR} \
node.local_cache_dir=${NODE_CACHE_DIR} \
node.phase_admission.enabled=${PHASE_ADMISSION} \
storage.backend=filesystem \
storage.transfer_prefix=${SHARED_DIR} \
storage.proving_cache_prefix=${CACHE_DIR}"

tmux new-session -d -s "$SESSION" -n "coordinator"
tmux send-keys -t "$SESSION:coordinator" \
    "${BASE_CMD} && python -m zkinfer.runtime.coordinator_grpc ${RUNTIME_CFG} \
    2>&1 | tee ${LOGS_DIR}/coordinator.tmux.log" C-m

echo "Waiting for coordinator at ${COORDINATOR_HOST}:${COORDINATOR_PORT}..."
python - "$COORDINATOR_HOST" "$COORDINATOR_PORT" <<'PY'
import socket
import sys
import time
host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + 60
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            raise SystemExit(0)
    except OSError:
        time.sleep(0.5)
raise SystemExit(f"Coordinator not ready at {host}:{port}")
PY

tmux new-window -t "$SESSION" -n "node"
tmux send-keys -t "$SESSION:node" \
    "${BASE_CMD} && python -m zkinfer.runtime.node.supervisor \
    ${RUNTIME_CFG} node.node_id=local-node \
    2>&1 | tee ${LOGS_DIR}/node.tmux.log" C-m

sleep 2

tmux new-window -t "$SESSION" -n "submit"
tmux send-keys -t "$SESSION:submit" \
    "${BASE_CMD} && \
    echo 'Run directory: ${RUN_DIR}' && \
    echo 'Phase admission: ${PHASE_ADMISSION}' && \
    python experiments/submit_job.py \
    +workload=${WORKLOAD} \
    launch.coordinator_host=${COORDINATOR_HOST} \
    launch.coordinator_port=${COORDINATOR_PORT} \
    2>&1 | tee ${LOGS_DIR}/submit.tmux.log" C-m

echo "Started tmux session: ${SESSION}"
echo "Run directory: ${RUN_DIR}"
echo "Phase admission: ${PHASE_ADMISSION}"
tmux attach -t "$SESSION"
