#!/usr/bin/env python3
"""
Quick Learning Rate Comparison Tool
Customizable comparison with different parameters
"""

import sys
import os
from lr_comparison import LRComparisonRunner

def main():
    """Main function with customizable parameters"""
    print("🔬 Quick Learning Rate Strategy Comparison")
    print("=" * 50)
    
    # Get user input for comparison parameters
    print("Available strategies:")
    strategies = [
        'reduce_lr_on_plateau',
        'cosine', 
        'step',
        'exponential',
        'one_cycle',
        'none'
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy}")
    
    print("\nSelect strategies to compare (comma-separated numbers, or 'all'):")
    choice = input("Choice: ").strip()
    
    if choice.lower() == 'all':
        selected_strategies = strategies
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_strategies = [strategies[i] for i in indices if 0 <= i < len(strategies)]
        except (ValueError, IndexError):
            print("❌ Invalid choice. Using all strategies.")
            selected_strategies = strategies
    
    print(f"\nSelected strategies: {selected_strategies}")
    
    # Get number of epochs
    try:
        max_epochs = int(input("Number of epochs per strategy (default 10): ").strip() or "10")
    except ValueError:
        max_epochs = 10
    
    print(f"Epochs per strategy: {max_epochs}")
    
    # Confirm before running
    print(f"\nThis will run {len(selected_strategies)} strategies for {max_epochs} epochs each.")
    print("Estimated time: ~{:.1f} minutes".format(len(selected_strategies) * max_epochs * 2))
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Comparison cancelled.")
        return
    
    # Create and run comparison
    runner = LRComparisonRunner()
    results = runner.run_comparison(strategies=selected_strategies, max_epochs=max_epochs)
    
    print(f"\n✅ Comparison completed!")
    print(f"Results saved to: {runner.results_dir}/")
    
    # Show quick summary
    if runner.comparison_data:
        print(f"\n📊 Quick Summary:")
        print("-" * 30)
        
        best_final_loss = min(runner.comparison_data, key=lambda x: x['Final_Loss'])
        fastest_training = min(runner.comparison_data, key=lambda x: x['Training_Time'])
        
        print(f"🏆 Best Final Loss: {best_final_loss['Strategy']} ({best_final_loss['Final_Loss']:.4f})")
        print(f"⚡ Fastest Training: {fastest_training['Strategy']} ({fastest_training['Training_Time']:.1f}s)")
        
        print(f"\n📁 Check detailed results in: {runner.results_dir}/")

if __name__ == "__main__":
    main()
