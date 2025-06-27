parallel-ssh -h hosts.txt -l ubuntu \
-x "-i /home/pw/ezkl-exploration/us-west-2-kp.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
"tmux kill-server || true"
