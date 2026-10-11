#!/bin/bash
SESSION=worker
tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION 'conda activate dzkml' C-m
tmux send-keys -t $SESSION 'cd ezkl-exploration' C-m
tmux send-keys -t $SESSION 'git pull' C-m         # <-- Added line
tmux send-keys -t $SESSION 'export PYTHONPATH=.:$PYTHONPATH' C-m
tmux send-keys -t $SESSION 'python zkInfer/worker.py' C-m
# tmux attach -t $SESSION
