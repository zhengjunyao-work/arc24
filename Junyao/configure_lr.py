#!/usr/bin/env python3
"""
Learning Rate Configuration Script
Easy configuration of different adaptive learning rate strategies
"""

import json
import os

def configure_lr_strategy(strategy_name: str):
    """Configure learning rate strategy in training_config.py"""
    config_file = "training_config.py"
    
    if not os.path.exists(config_file):
        print(f"❌ Configuration file {config_file} not found!")
        return
    
    # Read current configuration
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Define different LR strategies
    strategies = {
        'reduce_lr_on_plateau': {
            'scheduler_type': 'reduce_lr_on_plateau',
            'params': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-6,
                'verbose': True,
                'threshold': 1e-4,
                'threshold_mode': 'rel'
            },
            'description': 'Reduce LR when loss plateaus'
        },
        
        'cosine': {
            'scheduler_type': 'cosine',
            'params': {
                'T_max': 50,
                'eta_min': 1e-6,
                'last_epoch': -1
            },
            'description': 'Cosine annealing schedule'
        },
        
        'step': {
            'scheduler_type': 'step',
            'params': {
                'step_size': 10,
                'gamma': 0.5,
                'last_epoch': -1
            },
            'description': 'Step decay every N epochs'
        },
        
        'exponential': {
            'scheduler_type': 'exponential',
            'params': {
                'gamma': 0.95,
                'last_epoch': -1
            },
            'description': 'Exponential decay'
        },
        
        'one_cycle': {
            'scheduler_type': 'one_cycle',
            'params': {
                'max_lr': 1e-2,
                'total_steps': None,
                'epochs': None,
                'steps_per_epoch': None,
                'pct_start': 0.3,
                'anneal_strategy': 'cos',
                'cycle_momentum': True,
                'base_momentum': 0.85,
                'max_momentum': 0.95,
                'div_factor': 25.0,
                'final_div_factor': 1e4
            },
            'description': 'One-cycle policy (fast training)'
        },
        
        'none': {
            'scheduler_type': 'none',
            'params': {},
            'description': 'No learning rate scheduling'
        }
    }
    
    if strategy_name not in strategies:
        print(f"❌ Unknown strategy: {strategy_name}")
        print(f"Available strategies: {list(strategies.keys())}")
        return
    
    strategy = strategies[strategy_name]
    
    # Update the configuration using regex for more reliable replacement
    import re
    
    # Replace scheduler type
    content = re.sub(
        r"LR_SCHEDULER_TYPE = '[^']*'",
        f"LR_SCHEDULER_TYPE = '{strategy['scheduler_type']}'",
        content
    )
    
    # Replace scheduler parameters
    # Find the scheduler parameters section
    start_marker = "LR_SCHEDULER_PARAMS = {"
    end_marker = "}"
    
    start_idx = content.find(start_marker)
    if start_idx != -1:
        # Find the matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(content[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        # Create new parameters string
        params_str = "LR_SCHEDULER_PARAMS = {\n"
        for key, value in strategy['params'].items():
            if isinstance(value, str):
                params_str += f"        '{key}': '{value}',\n"
            elif isinstance(value, bool):
                params_str += f"        '{key}': {str(value)},\n"
            elif value is None:
                params_str += f"        '{key}': None,\n"
            else:
                params_str += f"        '{key}': {value},\n"
        params_str += "    }"
        
        # Replace the parameters section
        content = content[:start_idx] + params_str + content[end_idx+1:]
    
    # Write updated configuration
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Configured learning rate strategy: {strategy_name}")
    print(f"   Description: {strategy['description']}")
    print(f"   Scheduler: {strategy['scheduler_type']}")

def configure_warmup(use_warmup: bool, warmup_epochs: int = 3, method: str = 'linear'):
    """Configure warmup settings"""
    config_file = "training_config.py"
    
    if not os.path.exists(config_file):
        print(f"❌ Configuration file {config_file} not found!")
        return
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update warmup settings using regex for reliable replacement
    import re
    
    # Replace USE_WARMUP
    content = re.sub(
        r"USE_WARMUP = \w+",
        f"USE_WARMUP = {str(use_warmup)}",
        content
    )
    
    # Replace WARMUP_EPOCHS
    content = re.sub(
        r"WARMUP_EPOCHS = \d+",
        f"WARMUP_EPOCHS = {warmup_epochs}",
        content
    )
    
    # Replace WARMUP_METHOD
    content = re.sub(
        r"WARMUP_METHOD = '[^']*'",
        f"WARMUP_METHOD = '{method}'",
        content
    )
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Configured warmup:")
    print(f"   Use warmup: {use_warmup}")
    print(f"   Warmup epochs: {warmup_epochs}")
    print(f"   Warmup method: {method}")

def show_available_strategies():
    """Show available learning rate strategies"""
    strategies = {
        'reduce_lr_on_plateau': 'Reduce LR when loss plateaus (good for most cases)',
        'cosine': 'Cosine annealing schedule (smooth decay)',
        'step': 'Step decay every N epochs (simple)',
        'exponential': 'Exponential decay (continuous)',
        'one_cycle': 'One-cycle policy (fast training, requires careful tuning)',
        'none': 'No learning rate scheduling'
    }
    
    print("📈 Available Learning Rate Strategies:")
    print("=" * 50)
    for name, description in strategies.items():
        print(f"  {name:20} - {description}")
    print("=" * 50)

def main():
    """Main function"""
    print("Learning Rate Configuration Tool")
    print("=" * 40)
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'list':
            show_available_strategies()
        elif command == 'strategy':
            if len(sys.argv) > 2:
                strategy_name = sys.argv[2]
                configure_lr_strategy(strategy_name)
            else:
                print("❌ Please specify a strategy name")
                show_available_strategies()
        elif command == 'warmup':
            if len(sys.argv) > 2:
                use_warmup = sys.argv[2].lower() == 'true'
                warmup_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
                method = sys.argv[4] if len(sys.argv) > 4 else 'linear'
                configure_warmup(use_warmup, warmup_epochs, method)
            else:
                print("❌ Please specify warmup settings")
                print("Usage: python3 configure_lr.py warmup <true/false> [epochs] [method]")
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: list, strategy, warmup")
    else:
        # Interactive mode
        show_available_strategies()
        print("\nOptions:")
        print("1. Configure learning rate strategy")
        print("2. Configure warmup settings")
        print("3. Show available strategies")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            strategy = input("Enter strategy name: ").strip()
            configure_lr_strategy(strategy)
        elif choice == '2':
            use_warmup = input("Use warmup? (true/false): ").strip().lower() == 'true'
            if use_warmup:
                epochs = int(input("Warmup epochs (default 3): ").strip() or "3")
                method = input("Warmup method (linear/exponential/cosine, default linear): ").strip() or "linear"
                configure_warmup(use_warmup, epochs, method)
            else:
                configure_warmup(False)
        elif choice == '3':
            show_available_strategies()
        elif choice == '4':
            print("Goodbye!")
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()