#!/bin/bash


docker run --gpus all -it --rm --memory=10gb --cpus=20 pwatters991/ezkl-workloads:1.0
docker run --gpus all -it --rm --memory=10gb --cpus=16 pwatters991/ezkl-workloads:1.0
docker run --gpus all -it --rm --memory=10gb --cpus=12 pwatters991/ezkl-workloads:1.0
docker run --gpus all -it --rm --memory=10gb --cpus=8 pwatters991/ezkl-workloads:1.0
docker run --gpus all -it --rm --memory=10gb --cpus=4 pwatters991/ezkl-workloads:1.0
docker run --gpus all -it --rm --memory=10gb --cpus=2 pwatters991/ezkl-workloads:1.0


# # Run Docker containers with different memory limits
# docker run --gpus all -it --rm --memory=12gb --cpus=20 pwatters991/ezkl-workloads:1.0
# docker run --gpus all -it --rm --memory=10gb --cpus=20 pwatters991/ezkl-workloads:1.0
# docker run --gpus all -it --rm --memory=8gb --cpus=20 pwatters991/ezkl-workloads:1.0
# docker run --gpus all -it --rm --memory=6gb --cpus=20 pwatters991/ezkl-workloads:1.0
# docker run --gpus all -it --rm --memory=4gb --cpus=20 pwatters991/ezkl-workloads:1.0
# docker run --gpus all -it --rm --memory=1gb --cpus=4 pwatters991/ezkl-workloads:1.0