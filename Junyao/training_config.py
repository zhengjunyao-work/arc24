#!/usr/bin/env python3
"""
Training Configuration File
Easy toggle for GPU/CPU usage and other training parameters
"""

# ===== GPU CONFIGURATION =====
# Set to True to enable GPU, False to force CPU
USE_GPU = True  # Change this to False to disable GPU

# ===== TRAINING HYPERPARAMETERS =====
BATCH_SIZE = 64  # Increased batch size for better GPU utilization
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
INPUT_LENGTH = 1124
LATENT_DIM = 64
HIDDEN_DIMS = [512, 256, 128]
NUM_HEADS = 8

# ===== MODEL CONFIGURATION =====
USE_INPUT_NORM = True      # Normalize input data
USE_BATCH_NORM = True      # Use batch normalization

# ===== OPTIMIZATION CONFIGURATION =====
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0
BETA_VAE = 1.0  # VAE loss weight

# ===== ADAPTIVE LEARNING RATE CONFIGURATION =====
# Learning rate scheduler type: 'reduce_lr_on_plateau', 'cosine', 'step', 'exponential', 'one_cycle', 'none'
LR_SCHEDULER_TYPE = 'cosine'

# Base learning rate
LEARNING_RATE = 1e-3

# Learning rate scheduler parameters
LR_SCHEDULER_PARAMS = {
    # ReduceLROnPlateau parameters
    'reduce_lr_on_plateau': {
        'mode': 'min',           # 'min' for loss, 'max' for accuracy
        'factor': 0.5,           # Factor by which to reduce LR
        'patience': 5,           # Number of epochs with no improvement
        'min_lr': 1e-6,         # Minimum learning rate
        'verbose': True,         # Print messages when LR is reduced
        'threshold': 1e-4,       # Threshold for measuring improvement
        'threshold_mode': 'rel'  # 'rel' for relative, 'abs' for absolute
    },
    
    # CosineAnnealingLR parameters
    'cosine': {
        'T_max': 50,             # Maximum number of iterations
        'eta_min': 1e-6,         # Minimum learning rate
        'last_epoch': -1         # Index of last epoch
    },
    
    # StepLR parameters
    'step': {
        'step_size': 10,         # Period of learning rate decay
        'gamma': 0.5,            # Multiplicative factor
        'last_epoch': -1         # Index of last epoch
    },
    
    # ExponentialLR parameters
    'exponential': {
        'gamma': 0.95,           # Multiplicative factor
        'last_epoch': -1         # Index of last epoch
    },
    
    # OneCycleLR parameters
    'one_cycle': {
        'max_lr': 1e-2,          # Maximum learning rate
        'total_steps': None,     # Total number of steps (auto-calculated)
        'epochs': None,          # Number of epochs (auto-calculated)
        'steps_per_epoch': None, # Steps per epoch (auto-calculated)
        'pct_start': 0.3,        # Percentage of cycle spent increasing LR
        'anneal_strategy': 'cos', # 'cos' or 'linear'
        'cycle_momentum': True,  # Whether to cycle momentum
        'base_momentum': 0.85,   # Base momentum
        'max_momentum': 0.95,    # Maximum momentum
        'div_factor': 25.0,      # Initial LR = max_lr/div_factor
        'final_div_factor': 1e4  # Final LR = max_lr/final_div_factor
    }
}

# Warmup configuration
USE_WARMUP = True
WARMUP_EPOCHS = 3
WARMUP_METHOD = 'linear'  # 'linear', 'exponential', 'cosine'

# Learning rate monitoring
MONITOR_LR = True
LOG_LR_HISTORY = True

# ===== DATA CONFIGURATION =====
TRANSFORMED_DATA_PATH = '/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json'

# ===== SAVING CONFIGURATION =====
SAVE_BEST_MODEL = True
SAVE_CHECKPOINTS = True
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N epochs
SAVE_LOSS_PLOTS = True
LOSS_PLOT_INTERVAL = 10   # Save loss plot every N epochs

# ===== MONITORING CONFIGURATION =====
PRINT_INTERVAL = 10  # Print progress every N batches
SHOW_TIMING = True   # Show timing information

def print_config():
    """Print current configuration"""
    print("=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"GPU Usage: {'✅ ENABLED' if USE_GPU else '🖥️ DISABLED'}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Input Length: {INPUT_LENGTH}")
    print(f"Latent Dimension: {LATENT_DIM}")
    print(f"Hidden Dimensions: {HIDDEN_DIMS}")
    print(f"Number of Heads: {NUM_HEADS}")
    print(f"Input Normalization: {'✅' if USE_INPUT_NORM else '❌'}")
    print(f"Batch Normalization: {'✅' if USE_BATCH_NORM else '❌'}")
    
    # Learning rate scheduler information
    print(f"\n📈 ADAPTIVE LEARNING RATE:")
    print(f"  Scheduler Type: {LR_SCHEDULER_TYPE}")
    print(f"  Warmup: {'✅' if USE_WARMUP else '❌'} ({WARMUP_EPOCHS} epochs, {WARMUP_METHOD})")
    print(f"  LR Monitoring: {'✅' if MONITOR_LR else '❌'}")
    
    # Show scheduler-specific parameters
    if LR_SCHEDULER_TYPE in LR_SCHEDULER_PARAMS:
        params = LR_SCHEDULER_PARAMS[LR_SCHEDULER_TYPE]
        print(f"  Parameters: {params}")
    
    print("=" * 50)

if __name__ == "__main__":
    print_config()
