#!/bin/bash
#2,4,6,8,10
NUM_WORKERS=10    # Change this to however many workers you want

SESSION=zkexp

# Create a new tmux session
tmux new-session -d -s $SESSION

# Dispatcher in window 0
tmux rename-window -t $SESSION:0 'dispatcher'
tmux send-keys -t $SESSION:dispatcher 'conda activate dzkml' C-m
tmux send-keys -t $SESSION:dispatcher 'cd ezkl-exploration' C-m
tmux send-keys -t $SESSION:dispatcher 'export PYTHONPATH=.:$PYTHONPATH' C-m
tmux send-keys -t $SESSION:dispatcher "python zkInfer/dispatcher.py dispatcher.num_prover_workers=${NUM_WORKERS}" C-m

# Start workers in their own windows
for i in $(seq 1 $NUM_WORKERS); do
    tmux new-window -t $SESSION -n "worker$i"
    tmux send-keys -t $SESSION:worker$i 'conda activate dzkml' C-m
    tmux send-keys -t $SESSION:worker$i 'cd ezkl-exploration' C-m
    tmux send-keys -t $SESSION:worker$i 'export PYTHONPATH=.:$PYTHONPATH' C-m
    tmux send-keys -t $SESSION:worker$i 'python zkInfer/worker.py' C-m
done

# Optionally: Submit the job in another window
tmux new-window -t $SESSION -n "submit_job"
tmux send-keys -t $SESSION:submit_job 'conda activate dzkml' C-m
tmux send-keys -t $SESSION:submit_job 'cd ezkl-exploration' C-m
tmux send-keys -t $SESSION:submit_job 'export PYTHONPATH=.:$PYTHONPATH' C-m
tmux send-keys -t $SESSION:submit_job 'python zkInfer/submit_job.py' C-m

# Attach to the tmux session
tmux attach -t $SESSION
