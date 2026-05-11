#!/usr/bin/env bash
set -euo pipefail

NUM_WORKERS="${1:-1}"
WORKLOAD="${2:-mnist_classifier}"
COORDINATOR_HOST="${3:-localhost}"

SESSION="zkexp"
CONDA_ENV="${CONDA_ENV:-zk}"

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION"

SETUP_CMD="export PYTHONPATH=.:\"\$PYTHONPATH\" && conda activate ${CONDA_ENV}"

#
# Coordinator
#

tmux rename-window -t "$SESSION:0" "coordinator"

tmux send-keys -t "$SESSION:coordinator" \
  "$SETUP_CMD && python -m zkinfer.runtime.coordinator_grpc coordinator.host=${COORDINATOR_HOST}" \
  C-m

#
# Workers
#

for i in $(seq 1 "$NUM_WORKERS"); do
    tmux new-window -t "$SESSION" -n "worker${i}"

    tmux send-keys -t "$SESSION:worker${i}" \
      "$SETUP_CMD && python -m zkinfer.runtime.worker coordinator.host=${COORDINATOR_HOST} worker.worker_id=worker_${i}" \
      C-m
done

#
# Submit workload
#

tmux new-window -t "$SESSION" -n "submit"

tmux send-keys -t "$SESSION:submit" \
  "$SETUP_CMD && sleep 2 && python experiments/submit_job.py workload=${WORKLOAD} launch.num_workers=${NUM_WORKERS} launch.coordinator_host=${COORDINATOR_HOST}" \
  C-m

tmux attach -t "$SESSION"