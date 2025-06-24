#!/bin/bash

for NUM_WORKERS in 6 8 10
do
    SESSION=zkexp_${NUM_WORKERS}w

    echo "=== Launching experiment with $NUM_WORKERS workers (session: $SESSION) ==="

    tmux kill-session -t $SESSION 2>/dev/null

    tmux new-session -d -s $SESSION

    tmux rename-window -t $SESSION:0 'dispatcher'
    tmux send-keys -t $SESSION:dispatcher 'conda activate dzkml' C-m
    tmux send-keys -t $SESSION:dispatcher 'cd ezkl-exploration' C-m
    tmux send-keys -t $SESSION:dispatcher 'export PYTHONPATH=.:$PYTHONPATH' C-m
    tmux send-keys -t $SESSION:dispatcher "python zkInfer/dispatcher.py dispatcher.num_prover_workers=${NUM_WORKERS}" C-m

    for i in $(seq 1 $NUM_WORKERS); do
        tmux new-window -t $SESSION -n "worker$i"
        tmux send-keys -t $SESSION:worker$i 'conda activate dzkml' C-m
        tmux send-keys -t $SESSION:worker$i 'cd ezkl-exploration' C-m
        tmux send-keys -t $SESSION:worker$i 'export PYTHONPATH=.:$PYTHONPATH' C-m
        tmux send-keys -t $SESSION:worker$i 'python zkInfer/worker.py' C-m
    done

    tmux new-window -t $SESSION -n "submit_job"
    tmux send-keys -t $SESSION:submit_job 'conda activate dzkml' C-m
    tmux send-keys -t $SESSION:submit_job 'cd ezkl-exploration' C-m
    tmux send-keys -t $SESSION:submit_job 'export PYTHONPATH=.:$PYTHONPATH' C-m
    tmux send-keys -t $SESSION:submit_job 'python zkInfer/submit_job.py' C-m

    echo "  -> Use: tmux attach -t $SESSION"

    # Wait for manual confirmation before next experiment
    read -p "Press Enter to proceed to the next experiment with a different worker count..."
done

echo "=== All experiments launched and run sequentially. ==="
