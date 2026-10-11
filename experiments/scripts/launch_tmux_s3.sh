#!/usr/bin/env bash

set -euo pipefail

LOCAL_CONCURRENCY="${1:-1}"
WORKLOAD="${2:-mnist_gan}"
COORDINATOR_HOST="${3:-127.0.0.1}"
COORDINATOR_PORT="${4:-50051}"
OPS_PER_CHUNK="${5:-1}"
HEAVY_PHASE_LIMIT="${6:-${LOCAL_CONCURRENCY}}"

SESSION="${TMUX_SESSION:-zkexp}"
CONDA_ENV="${CONDA_ENV:-zk}"
S3_BUCKET="${ZKINFER_S3_BUCKET:-zkinfer}"
S3_PREFIX="${ZKINFER_S3_PREFIX:-zkinfer-reviewer}"
ROOT_DIR="$(pwd)"

for value_name in LOCAL_CONCURRENCY OPS_PER_CHUNK HEAVY_PHASE_LIMIT; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: ${value_name,,} must be a positive integer."
        exit 1
    fi
done
if [[ -z "$S3_BUCKET" ]]; then
    echo "Error: ZKINFER_S3_BUCKET is not set."
    exit 1
fi
if [[ ! -f "${ROOT_DIR}/experiments/submit_job.py" ]]; then
    echo "Error: run this script from the zkinfer repository root."
    exit 1
fi

RUN_ID="$(date -u +%Y-%m-%d_%H-%M-%S)_${WORKLOAD}_g${OPS_PER_CHUNK}_c${LOCAL_CONCURRENCY}_h${HEAVY_PHASE_LIMIT}"
CAMPAIGN_DIR="${ROOT_DIR}/experiments/runs/${RUN_ID}"
REQUESTS_DIR="${CAMPAIGN_DIR}/requests"
LOGS_DIR="${CAMPAIGN_DIR}/logs"
NODE_CACHE_DIR="${CAMPAIGN_DIR}/node-cache"
mkdir -p "$REQUESTS_DIR" "$LOGS_DIR" "$NODE_CACHE_DIR"

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
node.runs_dir=${REQUESTS_DIR} \
node.logs_dir=${LOGS_DIR} \
node.local_cache_dir=${NODE_CACHE_DIR} \
node.local_concurrency_limit=${LOCAL_CONCURRENCY} \
node.max_heavy_phases=${HEAVY_PHASE_LIMIT} \
storage.backend=s3 \
storage.s3_bucket=${S3_BUCKET} \
storage.s3_prefix=${S3_PREFIX} \
storage.transfer_prefix=transfer \
storage.proving_cache_enabled=true \
storage.proving_cache_prefix=cache \
storage.proving_cache_overwrite=false"

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
    echo 'Local concurrency: ${LOCAL_CONCURRENCY}' && \
    echo 'Heavy phase limit: ${HEAVY_PHASE_LIMIT}' && \
    echo 'ops_per_chunk: ${OPS_PER_CHUNK}' && \
    python experiments/submit_job.py \
    +workload=${WORKLOAD} \
    execution.split_mode=fixed \
    execution.ops_per_chunk=${OPS_PER_CHUNK} \
    launch.coordinator_host=${COORDINATOR_HOST} \
    launch.coordinator_port=${COORDINATOR_PORT} \
    2>&1 | tee -a ${LOGS_DIR}/submit.tmux.log" C-m

echo "Started tmux session: ${SESSION}"
echo "Campaign directory: ${CAMPAIGN_DIR}"
echo "S3 location: s3://${S3_BUCKET}/${S3_PREFIX}"
echo "Local concurrency: ${LOCAL_CONCURRENCY}; heavy phase limit: ${HEAVY_PHASE_LIMIT}"
tmux attach -t "$SESSION"
