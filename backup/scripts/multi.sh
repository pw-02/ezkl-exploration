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

    tmux new-session -d -s $SESSION

    tmux rename-window -t $SESSION:0 'coordinator'
    tmux send-keys -t $SESSION:coordinator 'conda activate dzkml' C-m
    tmux send-keys -t $SESSION:coordinator 'cd ezkl-exploration' C-m
    tmux send-keys -t $SESSION:coordinator 'export PYTHONPATH=.:$PYTHONPATH' C-m
    tmux send-keys -t $SESSION:coordinator "python zkInfer/coordinator.py coordinator.host=${DISPATCHER_HOST} coordinator.num_prover_workers=${NUM_WORKERS}" C-m

    for i in $(seq 1 $NUM_WORKERS); do
        tmux new-window -t $SESSION -n "worker$i"
        tmux send-keys -t $SESSION:worker$i 'conda activate dzkml' C-m
        tmux send-keys -t $SESSION:worker$i 'cd ezkl-exploration' C-m
        tmux send-keys -t $SESSION:worker$i 'export PYTHONPATH=.:$PYTHONPATH' C-m
        tmux send-keys -t $SESSION:worker$i "python zkInfer/worker.py coordinator.host=${DISPATCHER_HOST}" C-m
    done

    tmux new-window -t $SESSION -n "submit_job"
    tmux send-keys -t $SESSION:submit_job 'conda activate dzkml' C-m
    tmux send-keys -t $SESSION:submit_job 'cd ezkl-exploration' C-m
    tmux send-keys -t $SESSION:submit_job 'export PYTHONPATH=.:$PYTHONPATH' C-m
    tmux send-keys -t $SESSION:submit_job "python zkInfer/submit_job.py model=${MODEL_NAME} coordinator.host=${DISPATCHER_HOST}" C-m

    while tmux list-windows -t $SESSION 2>/dev/null | grep -q coordinator; do
    sleep 10
    done
    tmux kill-session -t $SESSION 2>/dev/null



    echo "Experiment with $NUM_WORKERS workers complete."
    sleep 10  # Short pause between runs (optional)
done

echo "=========================================="
echo "All experiments completed."
