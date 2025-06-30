#!/bin/bash
# Usage: ./run_zkexp.sh [NUM_WORKERS] [MODEL_NAME]
# Example: ./run_zkexp.sh 4 resnet50

NUM_WORKERS=${1:-1}          # First argument or default to 1
MODEL_NAME=${2:-mnist_classifier}  # Second argument or default

SESSION=zkexp

# Create a new tmux session (kills if exists)
tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION

# Dispatcher in window 0
tmux rename-window -t $SESSION:0 'dispatcher'
tmux send-keys -t $SESSION:dispatcher 'conda activate dzkml' C-m
tmux send-keys -t $SESSION:dispatcher 'cd ezkl-exploration' C-m
tmux send-keys -t $SESSION:dispatcher 'export PYTHONPATH=.:$PYTHONPATH' C-m
tmux send-keys -t $SESSION:dispatcher "python zkInfer/dispatcher.py dispatcher.num_prover_workers=${NUM_WORKERS}" C-m

# Optionally: Submit the job in another window, passing MODEL_NAME
tmux new-window -t $SESSION -n "submit_job"
tmux send-keys -t $SESSION:submit_job 'conda activate dzkml' C-m
tmux send-keys -t $SESSION:submit_job 'cd ezkl-exploration' C-m
tmux send-keys -t $SESSION:submit_job 'export PYTHONPATH=.:$PYTHONPATH' C-m
tmux send-keys -t $SESSION:submit_job "python zkInfer/submit_job.py model=${MODEL_NAME}" C-m

# Attach to the tmux session
tmux attach -t $SESSION
