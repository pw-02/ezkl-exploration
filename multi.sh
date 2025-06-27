#!/bin/bash
# Usage: ./batch_zkexp.sh MODEL_NAME HOST
# Example: ./batch_zkexp.sh resnet50 10.0.0.8

MODEL_NAME=${1:-mobilenetv2_050_Opset18_split_size_1}
DISPATCHER_HOST=${2:-localhost}
WORKERS_LIST=(2 4 6 8 10 1)
SESSION=zkexp

for NUM_WORKERS in "${WORKERS_LIST[@]}"; do
    echo "=========================================="
    echo "Starting experiment with $NUM_WORKERS workers..."

    # Create new tmux session
    tmux new-session -d -s $SESSION

    # Dispatcher in window 0
    tmux rename-window -t $SESSION:0 'dispatcher'
    tmux send-keys -t $SESSION:dispatcher 'conda activate dzkml' C-m
    tmux send-keys -t $SESSION:dispatcher 'cd ezkl-exploration' C-m
    tmux send-keys -t $SESSION:dispatcher 'export PYTHONPATH=.:$PYTHONPATH' C-m
    tmux send-keys -t $SESSION:dispatcher "python zkInfer/dispatcher.py dispatcher.host=${DISPATCHER_HOST} dispatcher.num_prover_workers=${NUM_WORKERS}" C-m

    # Start workers in their own windows
    for i in $(seq 1 $NUM_WORKERS); do
        tmux new-window -t $SESSION -n "worker$i"
        tmux send-keys -t $SESSION:worker$i 'conda activate dzkml' C-m
        tmux send-keys -t $SESSION:worker$i 'cd ezkl-exploration' C-m
        tmux send-keys -t $SESSION:worker$i 'export PYTHONPATH=.:$PYTHONPATH' C-m
        tmux send-keys -t $SESSION:worker$i "python zkInfer/worker.py dispatcher.host=${DISPATCHER_HOST}" C-m
    done

    # Submit the job in another window
    tmux new-window -t $SESSION -n "submit_job"
    tmux send-keys -t $SESSION:submit_job 'conda activate dzkml' C-m
    tmux send-keys -t $SESSION:submit_job 'cd ezkl-exploration' C-m
    tmux send-keys -t $SESSION:submit_job 'export PYTHONPATH=.:$PYTHONPATH' C-m
    tmux send-keys -t $SESSION:submit_job "python zkInfer/submit_job.py model=${MODEL_NAME} dispatcher.host=${DISPATCHER_HOST}" C-m

    # Wait for dispatcher to exit before moving on
    echo "Waiting for dispatcher to exit for $NUM_WORKERS workers..."
    while tmux list-windows -t $SESSION | grep -q dispatcher; do
        # Check if the dispatcher pane is still running
        tmux capture-pane -pt $SESSION:dispatcher -S -10 | grep -q "✅ Dispatcher gRPC server running" && sleep 10 || break
    done

    # Just in case: Ensure tmux session is cleaned up
    tmux kill-session -t $SESSION

    echo "Experiment with $NUM_WORKERS workers complete."
    sleep 10  # Short pause between runs (optional)
done

echo "=========================================="
echo "All experiments completed."
