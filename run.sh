#!/bin/bash

# Run Docker containers with different memory limits
docker run --gpus all -it --rm --memory=10gb --cpus=20 pwatters991/ezkl-workloads:1.0
docker run --gpus all -it --rm --memory=8gb --cpus=20 pwatters991/ezkl-workloads:1.0
docker run --gpus all -it --rm --memory=6gb --cpus=20 pwatters991/ezkl-workloads:1.0
docker run --gpus all -it --rm --memory=4gb --cpus=20 pwatters991/ezkl-workloads:1.0
docker run --gpus all -it --rm --memory=2gb --cpus=20 pwatters991/ezkl-workloads:1.0