parallel-ssh -h hosts.txt -l ubuntu -x "-i /home/pw/ezkl-exploration/us-west-2-kp.pem" \
'tmux new-session -d -s worker "bash -c \"source ~/miniconda3/etc/profile.d/conda.sh && conda activate dzkml && cd ezkl-exploration && git pull && export PYTHONPATH=.:$PYTHONPATH && python zkInfer/worker.py\""'
