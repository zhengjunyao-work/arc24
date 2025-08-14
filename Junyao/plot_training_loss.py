#!/usr/bin/env python3
"""
Script to load and plot training loss data from saved files
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_loss_data(file_path='vae_1d_training_loss_data.json'):
    """Load loss data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Could not find loss data file at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading loss data: {e}")
        return None

def plot_loss_from_data(loss_data, save_path='vae_1d_loss_analysis.png'):
    """Create comprehensive loss analysis plot from loaded data"""
    if loss_data is None:
        print("No loss data available for plotting")
        return
    
    losses = loss_data['losses']
    epochs = loss_data['epochs']
    best_loss = loss_data['best_loss']
    best_epoch = loss_data['best_epoch']
    final_loss = loss_data['final_loss']
    loss_reduction = loss_data['loss_reduction']
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VAE Training Loss Analysis', fontsize=16, fontweight='bold')
    
    # 1. Main loss curve
    axes[0, 0].plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=3)
    axes[0, 0].set_title('Training Loss Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mark best loss
    axes[0, 0].plot(best_epoch, best_loss, 'ro', markersize=8, label=f'Best: {best_loss:.4f}')
    axes[0, 0].legend()
    
    # 2. Loss reduction analysis
    axes[0, 1].plot(epochs, losses, 'g-', linewidth=2)
    axes[0, 1].axhline(y=losses[0], color='r', linestyle='--', alpha=0.7, label=f'Starting Loss: {losses[0]:.4f}')
    axes[0, 1].axhline(y=final_loss, color='b', linestyle='--', alpha=0.7, label=f'Final Loss: {final_loss:.4f}')
    axes[0, 1].set_title('Loss Reduction Analysis', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Loss statistics
    loss_stats = {
        'Min Loss': min(losses),
        'Max Loss': max(losses),
        'Final Loss': final_loss,
        'Best Loss': best_loss,
        'Loss Reduction': loss_reduction
    }
    
    colors = ['green', 'red', 'blue', 'orange', 'purple']
    bars = axes[0, 2].bar(loss_stats.keys(), loss_stats.values(), color=colors)
    axes[0, 2].set_title('Loss Statistics', fontweight='bold')
    axes[0, 2].set_ylabel('Loss Value')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, (key, value) in zip(bars, loss_stats.items()):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + max(loss_stats.values()) * 0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Loss rate of change
    if len(losses) > 1:
        loss_changes = [losses[i] - losses[i-1] for i in range(1, len(losses))]
        axes[1, 0].plot(epochs[1:], loss_changes, 'purple', linewidth=2, marker='s', markersize=3)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_title('Loss Rate of Change', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Change')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor rate analysis', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Loss Rate of Change', fontweight='bold')
    
    # 5. Smoothed loss trend
    if len(losses) > 1:
        window_size = min(5, len(losses) // 2)
        if window_size > 1:
            smoothed_losses = []
            for i in range(len(losses)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(losses), i + window_size // 2 + 1)
                smoothed_losses.append(np.mean(losses[start_idx:end_idx]))
            axes[1, 1].plot(epochs, smoothed_losses, 'orange', linewidth=3, 
                           label=f'Smoothed (window={window_size})')
        else:
            axes[1, 1].plot(epochs, losses, 'orange', linewidth=3, label='Training Loss')
        
        axes[1, 1].plot(epochs, losses, 'b-', alpha=0.3, label='Original')
        axes[1, 1].set_title('Smoothed Loss Trend', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor smoothing', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Smoothed Loss Trend', fontweight='bold')
    
    # 6. Training summary
    summary_text = f"""
Training Summary:
• Total Epochs: {len(epochs)}
• Starting Loss: {losses[0]:.4f}
• Final Loss: {final_loss:.4f}
• Best Loss: {best_loss:.4f} (Epoch {best_epoch})
• Loss Reduction: {loss_reduction:.4f}
• Improvement: {(loss_reduction/losses[0]*100):.2f}%
    """
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    axes[1, 2].set_title('Training Summary', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Loss analysis plot saved to: {save_path}")

def main():
    """Main function to load and plot loss data"""
    print("VAE Training Loss Analysis")
    print("=" * 40)
    
    # Try to load loss data
    loss_data = load_loss_data()
    
    if loss_data:
        print("Loss data loaded successfully!")
        print(f"Training epochs: {len(loss_data['epochs'])}")
        print(f"Best loss: {loss_data['best_loss']:.4f}")
        print(f"Final loss: {loss_data['final_loss']:.4f}")
        
        # Create comprehensive plot
        plot_loss_from_data(loss_data)
    else:
        print("No loss data found. Please run training first to generate loss data.")

if __name__ == "__main__":
    main()
