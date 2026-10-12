#!/usr/bin/env bash

set -euo pipefail

WORKLOAD="${1:-mnist_gan}"
COORDINATOR_HOST="${2:-127.0.0.1}"
COORDINATOR_PORT="${3:-50051}"
OPERATORS_PER_PARTITION="${4:-1}"
PHASE_ADMISSION="${5:-true}"
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-1}"

SESSION="${TMUX_SESSION:-zkexp}"
CONDA_ENV="${CONDA_ENV:-zk}"
S3_BUCKET="${ZKINFER_S3_BUCKET:-zkinfer}"
S3_PREFIX="${ZKINFER_S3_PREFIX:-zkinfer-reviewer}"
ROOT_DIR="$(pwd)"

if [[ ! "$OPERATORS_PER_PARTITION" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: operators_per_partition must be a positive integer."
    exit 1
fi
if [[ "$PHASE_ADMISSION" != "true" && "$PHASE_ADMISSION" != "false" ]]; then
    echo "Error: phase_admission must be true or false."
    exit 1
fi
if [[ "$MAX_ACTIVE_JOBS" != "null" && ! "$MAX_ACTIVE_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_ACTIVE_JOBS must be a positive integer or 'null'."
    exit 1
fi
if [[ -z "$S3_BUCKET" ]]; then
    echo "Error: ZKINFER_S3_BUCKET is not set."
    exit 1
fi
if [[ ! -f "${ROOT_DIR}/experiments/submit_job.py" ]]; then
    echo "Error: run this script from the zkinfer repository root."
    exit 1
fi

RUN_ID="$(date -u +%Y-%m-%d_%H-%M-%S)_${WORKLOAD}_g${OPERATORS_PER_PARTITION}_admission-${PHASE_ADMISSION}"
CAMPAIGN_DIR="${ROOT_DIR}/experiments/runs/${RUN_ID}"
REQUESTS_DIR="${CAMPAIGN_DIR}/requests"
LOGS_DIR="${CAMPAIGN_DIR}/logs"
CONFIG_DIR="${CAMPAIGN_DIR}/config"
STATE_DB="${CAMPAIGN_DIR}/coordinator_state.sqlite3"

# Node-local persistent runtime state is kept outside experiment results.
NODE_CACHE_DIR="${ROOT_DIR}/storage/node/cache"
NODE_RUNTIME_DIR="${ROOT_DIR}/storage/node/runtime"
mkdir -p "$REQUESTS_DIR" "$LOGS_DIR" "$CONFIG_DIR" "$NODE_CACHE_DIR" "$NODE_RUNTIME_DIR"

cat > "${CONFIG_DIR}/launch.txt" <<EOF
workload=${WORKLOAD}
coordinator_host=${COORDINATOR_HOST}
coordinator_port=${COORDINATOR_PORT}
operators_per_partition=${OPERATORS_PER_PARTITION}
phase_admission=${PHASE_ADMISSION}
max_active_jobs=${MAX_ACTIVE_JOBS}
state_db=${STATE_DB}
s3_bucket=${S3_BUCKET}
s3_prefix=${S3_PREFIX}
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
storage.backend=s3 \
storage.s3_bucket=${S3_BUCKET} \
storage.s3_prefix=${S3_PREFIX} \
storage.requests_prefix=requests \
storage.reuse_setup=true \
storage.setup_prefix=setup"

echo "Checking access to S3 bucket: ${S3_BUCKET}"
if command -v aws >/dev/null 2>&1; then
    aws s3api head-bucket --bucket "$S3_BUCKET" >/dev/null
else
    echo "Warning: AWS CLI not found; skipping S3 access check."
fi

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
    echo 'Campaign directory: ${CAMPAIGN_DIR}' && \
    echo 'Workload: ${WORKLOAD}' && \
    echo 'Phase admission: ${PHASE_ADMISSION}' && \
    echo 'Max active jobs: ${MAX_ACTIVE_JOBS}' && \
    echo 'Operators per partition: ${OPERATORS_PER_PARTITION}' && \
    python experiments/submit_job.py \
    +workload=${WORKLOAD} \
    execution.partitioning.mode=fixed_size \
    execution.partitioning.operators_per_partition=${OPERATORS_PER_PARTITION} \
    launch.coordinator_host=${COORDINATOR_HOST} \
    launch.coordinator_port=${COORDINATOR_PORT} \
    2>&1 | tee -a ${LOGS_DIR}/submit.tmux.log" C-m

echo "Started tmux session: ${SESSION}"
echo "Campaign directory: ${CAMPAIGN_DIR}"
echo "Requests: ${REQUESTS_DIR}"
echo "Node runtime storage: ${ROOT_DIR}/storage/node"
echo "S3 location: s3://${S3_BUCKET}/${S3_PREFIX}"
echo "Phase admission: ${PHASE_ADMISSION}"
echo "Max active jobs: ${MAX_ACTIVE_JOBS}"
tmux attach -t "$SESSION"

