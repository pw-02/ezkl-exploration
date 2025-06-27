#!/bin/bash
# Usage: ./yourscript.sh NUM_WORKERS MODEL_NAME
# Example: ./yourscript.sh 4 resnet50

NUM_WORKERS=${1:-1}                         # First arg = number of workers, default 1
MODEL_NAME=${2:-mobilenetv2_050_Opset18_split_size_1}  # Second arg = model name, default as before
SESSION=zkexp

# Create a new tmux session
tmux new-session -d -s $SESSION

# Dispatcher in window 0
tmux rename-window -t $SESSION:0 'dispatcher'
tmux send-keys -t $SESSION:dispatcher 'conda activate dzkml' C-m
tmux send-keys -t $SESSION:dispatcher 'cd ezkl-exploration' C-m
tmux send-keys -t $SESSION:dispatcher 'export PYTHONPATH=.:$PYTHONPATH' C-m
tmux send-keys -t $SESSION:dispatcher "python zkInfer/dispatcher.py s3_bucket=None dispatcher.host=localhost dispatcher.num_prover_workers=${NUM_WORKERS}" C-m

# Start workers in their own windows
for i in $(seq 1 $NUM_WORKERS); do
    tmux new-window -t $SESSION -n "worker$i"
    tmux send-keys -t $SESSION:worker$i 'conda activate dzkml' C-m
    tmux send-keys -t $SESSION:worker$i 'cd ezkl-exploration' C-m
    tmux send-keys -t $SESSION:worker$i 'export PYTHONPATH=.:$PYTHONPATH' C-m
    tmux send-keys -t $SESSION:worker$i 'python zkInfer/worker.py' C-m
done

# Optionally: Submit the job in another window (now using MODEL_NAME)
tmux new-window -t $SESSION -n "submit_job"
tmux send-keys -t $SESSION:submit_job 'conda activate dzkml' C-m
tmux send-keys -t $SESSION:submit_job 'cd ezkl-exploration' C-m
tmux send-keys -t $SESSION:submit_job 'export PYTHONPATH=.:$PYTHONPATH' C-m
tmux send-keys -t $SESSION:submit_job "python zkInfer/submit_job.py model=${MODEL_NAME}" C-m

# Attach to the tmux session
tmux attach -t $SESSION
