FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

# This has been copied from https://github.com/ironbar/insightface/blob/master/recognition/arcface_torch/docker/Dockerfile

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
# Combine apt-get commands and clean up to reduce layers and image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip python3.10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Default user for calculon cluster
RUN useradd -ms /bin/bash -u 1000 python-dev && \
    echo "python-dev ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers
# Default configuration
WORKDIR /home/python-dev
USER 1000