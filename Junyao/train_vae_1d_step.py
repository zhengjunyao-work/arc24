#!/usr/bin/env python3
"""
Temporary training script for parallel execution
Uses specific config file: Junyao/training_config_step_20250828_171051.py
"""

import sys
import os

# Add the Junyao directory to Python path
current_dir = "/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/Junyao"
sys.path.insert(0, current_dir)

# Change to the correct directory
os.chdir(current_dir)

# Import training modules
from train_vae_1d import train_vae, plot_training_results

# Override config import to use specific config file
import importlib.util
spec = importlib.util.spec_from_file_location("temp_config", "Junyao/training_config_step_20250828_171051.py")
temp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temp_config)

# Override global variables with temp config values
globals().update({k: v for k, v in temp_config.__dict__.items() 
                 if not k.startswith('_') and k.isupper()})

if __name__ == "__main__":
    # Run training with overridden config
    model, losses = train_vae()
    print(f"Training completed for config: {config_file}")
