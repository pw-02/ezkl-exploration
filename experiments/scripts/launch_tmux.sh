#!/usr/bin/env bash

set -euo pipefail

WORKLOAD="${1:-mnist_gan}"
COORDINATOR_HOST="${2:-127.0.0.1}"
COORDINATOR_PORT="${3:-50051}"
PHASE_ADMISSION="${4:-true}"
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-1}"

if [[ "$PHASE_ADMISSION" != "true" && "$PHASE_ADMISSION" != "false" ]]; then
    echo "Error: phase_admission must be true or false."
    exit 1
fi

if [[ "$MAX_ACTIVE_JOBS" != "null" && ! "$MAX_ACTIVE_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_ACTIVE_JOBS must be a positive integer or 'null'."
    exit 1
fi

SESSION="${TMUX_SESSION:-zkexp}"
CONDA_ENV="${CONDA_ENV:-zk}"
ROOT_DIR="$(pwd)"
ADMISSION_TAG="admission-${PHASE_ADMISSION}"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_${WORKLOAD}_${ADMISSION_TAG}"
RUN_DIR="${ROOT_DIR}/experiments/runs/${RUN_ID}"
REQUESTS_DIR="${RUN_DIR}/requests"
LOGS_DIR="${RUN_DIR}/logs"
CONFIG_DIR="${RUN_DIR}/config"
STATE_DB="${RUN_DIR}/coordinator_state.sqlite3"

# Persistent/runtime state is deliberately separate from experiment results.
REQUEST_STORE_DIR="${ROOT_DIR}/storage/requests"
SETUP_STORE_DIR="${ROOT_DIR}/storage/setup"
NODE_CACHE_DIR="${ROOT_DIR}/storage/node/cache"
NODE_RUNTIME_DIR="${ROOT_DIR}/storage/node/runtime"

mkdir -p \
    "$REQUESTS_DIR" \
    "$LOGS_DIR" \
    "$CONFIG_DIR" \
    "$REQUEST_STORE_DIR" \
    "$SETUP_STORE_DIR" \
    "$NODE_CACHE_DIR" \
    "$NODE_RUNTIME_DIR"

cat > "${CONFIG_DIR}/launch.txt" <<EOF
workload=${WORKLOAD}
coordinator_host=${COORDINATOR_HOST}
coordinator_port=${COORDINATOR_PORT}
phase_admission=${PHASE_ADMISSION}
max_active_jobs=${MAX_ACTIVE_JOBS}
state_db=${STATE_DB}
requests_dir=${REQUEST_STORE_DIR}
setup_dir=${SETUP_STORE_DIR}
node_cache_dir=${NODE_CACHE_DIR}
node_runtime_dir=${NODE_RUNTIME_DIR}
EOF

if [[ "$MAX_ACTIVE_JOBS" == "null" && "$PHASE_ADMISSION" == "false" ]]; then
    echo "Warning: unlimited active jobs with phase admission disabled can start many provers at once."
    echo "Use this only for an intentional concurrency characterization run."
fi

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
coordinator.runs_dir=${REQUESTS_DIR} \
coordinator.logs_dir=${LOGS_DIR} \
coordinator.state_db_path=${STATE_DB} \
node.runs_dir=${REQUESTS_DIR} \
node.logs_dir=${LOGS_DIR} \
node.local_cache_dir=${NODE_CACHE_DIR} \
node.runtime_state_dir=${NODE_RUNTIME_DIR} \
node.max_active_jobs=${MAX_ACTIVE_JOBS} \
node.phase_admission.enabled=${PHASE_ADMISSION} \
storage.backend=filesystem \
storage.requests_prefix=${REQUEST_STORE_DIR} \
storage.setup_prefix=${SETUP_STORE_DIR}"

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
    echo 'Max active jobs: ${MAX_ACTIVE_JOBS}' && \
    python experiments/submit_job.py \
    +workload=${WORKLOAD} \
    launch.coordinator_host=${COORDINATOR_HOST} \
    launch.coordinator_port=${COORDINATOR_PORT} \
    2>&1 | tee ${LOGS_DIR}/submit.tmux.log" C-m

echo "Started tmux session: ${SESSION}"
echo "Run directory: ${RUN_DIR}"
echo "Requests: ${REQUESTS_DIR}"
echo "Persistent storage: ${ROOT_DIR}/storage"
echo "Phase admission: ${PHASE_ADMISSION}"
echo "Max active jobs: ${MAX_ACTIVE_JOBS}"
tmux attach -t "$SESSION"

