#!/usr/bin/env bash
set -euo pipefail

NUM_WORKERS="${1:-1}"
WORKLOAD="${2:-mnist_classifier}"
COORDINATOR_HOST="${3:-127.0.0.1}"
COORDINATOR_PORT="${4:-50051}"

SESSION="zkexp"
CONDA_ENV="${CONDA_ENV:-zk}"
ROOT_DIR="$(pwd)"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_${WORKLOAD}"
RUN_DIR="${ROOT_DIR}/experiments/runs/${RUN_ID}"

LOGS_DIR="${RUN_DIR}/logs"
REPORTS_DIR="${RUN_DIR}/reports"
ARTIFACTS_DIR="${RUN_DIR}/artifacts"
SHARED_DIR="${RUN_DIR}/shared"
TMP_DIR="${RUN_DIR}/tmp"

mkdir -p "$LOGS_DIR" "$REPORTS_DIR" "$ARTIFACTS_DIR" "$SHARED_DIR" "$TMP_DIR"

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

BASE_CMD="cd ${ROOT_DIR} && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && export PYTHONPATH=${ROOT_DIR}:\$PYTHONPATH"

COMMON_CFG="coordinator.host=${COORDINATOR_HOST} coordinator.port=${COORDINATOR_PORT} paths.tmp_dir=${TMP_DIR} paths.logs_dir=${LOGS_DIR} paths.reports_dir=${REPORTS_DIR} paths.artifacts_dir=${ARTIFACTS_DIR} file_transfer.root_dir=${SHARED_DIR}"

tmux new-session -d -s "$SESSION" -n "coordinator"

tmux send-keys -t "$SESSION:coordinator" \
  "${BASE_CMD} && python -m zkinfer.runtime.coordinator_grpc ${COMMON_CFG} 2>&1 | tee ${LOGS_DIR}/coordinator.tmux.log" \
  C-m

tmux new-window -t "$SESSION" -n "wait"
tmux send-keys -t "$SESSION:wait" \
  "${BASE_CMD} && python - <<'PY'
import socket, time, sys
host='${COORDINATOR_HOST}'
port=${COORDINATOR_PORT}
deadline=time.time()+60
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            print(f'Coordinator ready at {host}:{port}')
            sys.exit(0)
    except OSError:
        time.sleep(0.5)
raise SystemExit(f'Coordinator not ready at {host}:{port}')
PY" \
  C-m

sleep 3

for i in $(seq 1 "$NUM_WORKERS"); do
  tmux new-window -t "$SESSION" -n "worker${i}"

  tmux send-keys -t "$SESSION:worker${i}" \
    "${BASE_CMD} && python -m zkinfer.runtime.worker ${COMMON_CFG} worker.worker_id=worker_${i} 2>&1 | tee ${LOGS_DIR}/worker_${i}.tmux.log" \
    C-m
done

tmux new-window -t "$SESSION" -n "submit"

tmux send-keys -t "$SESSION:submit" \
  "${BASE_CMD} && echo 'Run dir: ${RUN_DIR}' && python experiments/submit_job.py workload=${WORKLOAD} launch.num_workers=${NUM_WORKERS} launch.coordinator_host=${COORDINATOR_HOST} launch.coordinator_port=${COORDINATOR_PORT} 2>&1 | tee ${LOGS_DIR}/submit.tmux.log" \
  C-m

echo "Started tmux session: $SESSION"
echo "Run dir: $RUN_DIR"
echo "Logs: $LOGS_DIR"

tmux attach -t "$SESSION"