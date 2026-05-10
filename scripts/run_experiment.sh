#!/usr/bin/env bash
set -euo pipefail

# Usage: ./yourscript.sh NUM_WORKERS MODEL_NAME DISPATCHER_HOST
# Example: ./yourscript.sh 4 resnet50 10.0.0.8

NUM_WORKERS="${1:-1}"
MODEL_NAME="${2:-tiny_bert_split_size_1}"
DISPATCHER_HOST="${3:-localhost}"

SESSION="zkexp"
PROJECT_DIR="zkInfer"
# CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

# Replace any existing session with the same name
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

# Create a new tmux session
tmux new-session -d -s "$SESSION"

# Common setup command
SETUP_CMD="export PYTHONPATH=.:\"\$PYTHONPATH\""

# Dispatcher window
tmux rename-window -t "$SESSION:0" "dispatcher"
tmux send-keys -t "$SESSION:dispatcher" "$SETUP_CMD && conda activate zk && python zkInfer/dispatcher.py dispatcher.host=${DISPATCHER_HOST} dispatcher.num_prover_workers=${NUM_WORKERS}" C-m

# Worker windows
for i in $(seq 1 "$NUM_WORKERS"); do
    tmux new-window -t "$SESSION" -n "worker$i"
    tmux send-keys -t "$SESSION:worker$i" "$SETUP_CMD && conda activate zk && python zkInfer/worker.py dispatcher.host=${DISPATCHER_HOST}" C-m
done

# Submit job window
tmux new-window -t "$SESSION" -n "submit_job"
tmux send-keys -t "$SESSION:submit_job" "$SETUP_CMD && conda activate zk && python zkInfer/submit_job.py model=${MODEL_NAME} dispatcher.host=${DISPATCHER_HOST}" C-m

# Attach to tmux session
tmux attach -t "$SESSION"