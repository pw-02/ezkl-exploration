#!/usr/bin/env bash
set -euo pipefail

NUM_WORKERS="${1:-1}"
WORKLOAD="${2:-mnist_classifier}"
COORDINATOR_HOST="${3:-localhost}"
COORDINATOR_PORT="${4:-50051}"

SESSION="zkexp"
CONDA_ENV="${CONDA_ENV:-zk}"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_${WORKLOAD}"
RUN_DIR="runs/${RUN_ID}"

LOGS_DIR="${RUN_DIR}/logs"
REPORTS_DIR="${RUN_DIR}/reports"
ARTIFACTS_DIR="${RUN_DIR}/artifacts"
SHARED_DIR="${RUN_DIR}/shared"
TMP_DIR="${RUN_DIR}/tmp"

mkdir -p "$LOGS_DIR" "$REPORTS_DIR" "$ARTIFACTS_DIR" "$SHARED_DIR" "$TMP_DIR"

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
tmux new-session -d -s "$SESSION"

SETUP_CMD="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate ${CONDA_ENV} && export PYTHONPATH=.:\"\$PYTHONPATH\""

COMMON_CFG="coordinator.host=${COORDINATOR_HOST} coordinator.port=${COORDINATOR_PORT} paths.tmp_dir=${TMP_DIR} paths.logs_dir=${LOGS_DIR} paths.reports_dir=${REPORTS_DIR} paths.artifacts_dir=${ARTIFACTS_DIR} file_transfer.root_dir=${SHARED_DIR}"

tmux rename-window -t "$SESSION:0" "coordinator"

tmux send-keys -t "$SESSION:coordinator" \
  "$SETUP_CMD && python -m zkinfer.runtime.coordinator_grpc ${COMMON_CFG}" \
  C-m

for i in $(seq 1 "$NUM_WORKERS"); do
  tmux new-window -t "$SESSION" -n "worker${i}"

  tmux send-keys -t "$SESSION:worker${i}" \
    "$SETUP_CMD && python -m zkinfer.runtime.worker ${COMMON_CFG} worker.worker_id=worker_${i}" \
    C-m
done

tmux new-window -t "$SESSION" -n "submit"

tmux send-keys -t "$SESSION:submit" \
  "echo 'Run dir: ${RUN_DIR}'; echo 'Submit with:'; echo '$SETUP_CMD && python experiments/submit_job.py workload=${WORKLOAD} launch.num_workers=${NUM_WORKERS} launch.coordinator_host=${COORDINATOR_HOST} launch.coordinator_port=${COORDINATOR_PORT} ${COMMON_CFG}'" \
  C-m

echo "Started tmux session: $SESSION"
echo "Run dir: $RUN_DIR"

tmux attach -t "$SESSION"