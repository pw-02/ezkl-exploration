parallel-ssh -h hosts.txt -l ubuntu -x "-i /home/pw/ezkl-exploration/us-west-2-kp.pem" \
"tmux kill-server || true"
