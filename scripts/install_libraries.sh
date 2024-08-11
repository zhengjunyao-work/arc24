#!/bin/bash

# Check if vllm is installed
if ! python -c "import vllm" &> /dev/null; then
    # Uninstall torch if vllm is not installed
    pip uninstall -q -y torch
    # Install the necessary packages from the provided directory
    pip install -q --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm vllm
    pip install -q --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm peft
    pip install -q --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm trl
    pip install -q --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm bitsandbytes
else
    echo "vllm is already installed."
fi
