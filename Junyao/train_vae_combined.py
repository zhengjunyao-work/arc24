#!/usr/bin/env python3
"""
Train VAE on Combined ARC Data (Input + Output Sequences)
This script trains the VAE on both input and output sequences from ARC training data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from combined_arc_dataset import CombinedARCDataset, collate_fn
from VAEModel import VAE1D
from training_config import *

def train_vae_combined(data_path: str = "/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
                      use_both_sequences: bool = True,
                      batch_size: int = None,
                      learning_rate: float = None,
                      num_epochs: int = None,
                      device: str = None,
                      save_model: bool = True):
    """
    Train VAE on combined ARC data (input + output sequences)
    
    Args:
        data_path: Path to transformed ARC data
        use_both_sequences: Whether to use both input and output sequences
        batch_size: Batch size (uses config if None)
        learning_rate: Learning rate (uses config if None)
        num_epochs: Number of epochs (uses config if None)
        device: Device to use ('auto', 'cpu', 'mps', 'cuda')
        save_model: Whether to save the trained model
    
    Returns:
        Trained model and training losses
    """
    
    # Use config values if not specified
    batch_size = batch_size or BATCH_SIZE
    learning_rate = learning_rate or LEARNING_RATE
    num_epochs = num_epochs or NUM_EPOCHS
    
    # Device setup
    if device is None or device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"🔧 Using device: {device}")
    
    # Load combined dataset
    print(f"📥 Loading combined ARC data from {data_path}...")
    dataset = CombinedARCDataset(data_path, use_both_sequences=use_both_sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    print(f"📊 Dataset info:")
    print(f"  - Total sequences: {len(dataset)}")
    print(f"  - Use both sequences: {use_both_sequences}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of batches: {len(dataloader)}")
    
    # Create VAE model
    print(f"🏗️  Creating VAE model...")
    model = VAE1D(
        input_length=INPUT_LENGTH,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_heads=NUM_HEADS,
        use_input_norm=USE_INPUT_NORM,
        use_batch_norm=USE_BATCH_NORM
    )
    
    model.to(device)
    print(f"Model info: {model.get_model_info()}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print(f"🚀 Starting training for {num_epochs} epochs...")
    losses = []
    best_loss = float('inf')
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            recon_data, mu, logvar = model(batch_data)
            
            # Calculate losses
            recon_loss = criterion(recon_data, batch_data)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / batch_data.size(0)  # Normalize by batch size
            
            total_loss = recon_loss + BETA_VAE * kl_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1
            
            if batch_idx % PRINT_INTERVAL == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}")
                print(f"    Total Loss: {total_loss.item():.6f}")
                print(f"    Recon Loss: {recon_loss.item():.6f}")
                print(f"    KL Loss: {kl_loss.item():.6f}")
        
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        
        losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f} "
              f"(Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}), LR: {current_lr:.2e}")
        
        # Early stopping check
        if epoch > 20 and avg_loss < 1e-6:
            print(f"Early stopping at epoch {epoch+1} (loss < 1e-6)")
            break
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    print(f"Best loss: {best_loss:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("✅ Loaded best model state")
    
    # Save model
    if save_model:
        model_path = "vae_combined_trained_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_length': INPUT_LENGTH,
                'latent_dim': LATENT_DIM,
                'hidden_dims': HIDDEN_DIMS,
                'num_heads': NUM_HEADS,
                'use_input_norm': USE_INPUT_NORM,
                'use_batch_norm': USE_BATCH_NORM
            },
            'training_config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'beta_vae': BETA_VAE,
                'weight_decay': WEIGHT_DECAY,
                'use_both_sequences': use_both_sequences
            },
            'dataset_info': {
                'data_path': data_path,
                'total_sequences': len(dataset),
                'use_both_sequences': use_both_sequences
            }
        }, model_path)
        print(f"✅ Model saved to {model_path}")
    
    # Plot training results
    plot_training_results(losses, "combined")
    
    # Test reconstruction
    test_reconstruction(model, dataloader, device)
    
    return model, losses

def plot_training_results(losses, model_type):
    """Plot training results"""
    plt.figure(figsize=(15, 5))
    
    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title(f'VAE Combined Training - Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Loss reduction
    plt.subplot(1, 3, 2)
    loss_reduction = [losses[0] - loss for loss in losses]
    plt.plot(loss_reduction)
    plt.title('Loss Reduction Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Reduction')
    plt.grid(True, alpha=0.3)
    
    # Training speed
    plt.subplot(1, 3, 3)
    epochs = list(range(1, len(losses) + 1))
    plt.plot(epochs, losses)
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'vae_combined_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Training plots saved to vae_combined_training_results.png")

def test_reconstruction(model, dataloader, device, num_examples=5):
    """Test reconstruction quality"""
    print(f"\n🔄 Testing reconstruction quality...")
    
    model.eval()
    with torch.no_grad():
        # Get a few test samples
        test_samples = []
        for i, batch_data in enumerate(dataloader):
            if i >= 1:  # Just get first batch
                break
            test_samples.append(batch_data)
        
        if test_samples:
            batch_data = test_samples[0]
            original_data = batch_data[:num_examples].to(device)
            
            # Get reconstructions
            recon_data, mu, logvar = model(original_data)
            
            # Calculate metrics
            mse = torch.mean((recon_data - original_data) ** 2).item()
            mae = torch.mean(torch.abs(recon_data - original_data)).item()
            
            print(f"Test Metrics:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            
            # Visualize results
            fig, axes = plt.subplots(num_examples, 2, figsize=(15, 3*num_examples))
            if num_examples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_examples):
                # Original data
                axes[i, 0].plot(original_data[i].cpu().numpy())
                axes[i, 0].set_title(f'Original ARC Data {i+1}')
                axes[i, 0].set_ylabel('Type ID Value')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Reconstructed data
                recon_plot = recon_data[i].squeeze().cpu().numpy()
                axes[i, 1].plot(recon_plot)
                axes[i, 1].set_title(f'Reconstructed {i+1}')
                axes[i, 1].set_ylabel('Type ID Value')
                axes[i, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('vae_combined_reconstruction_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Reconstruction comparison saved to vae_combined_reconstruction_comparison.png")

def save_model(model, filepath: str = "vae_combined_trained_model.pth"):
    """Save the trained VAE model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_length': INPUT_LENGTH,
            'latent_dim': LATENT_DIM,
            'hidden_dims': HIDDEN_DIMS,
            'num_heads': NUM_HEADS,
            'use_input_norm': USE_INPUT_NORM,
            'use_batch_norm': USE_BATCH_NORM
        },
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'beta_vae': BETA_VAE,
            'weight_decay': WEIGHT_DECAY
        }
    }, filepath)
    print(f"✅ Model saved to {filepath}")

def load_model(filepath: str = "vae_combined_trained_model.pth"):
    """Load a trained VAE model"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Recreate model with saved config
    model_config = checkpoint['model_config']
    model = VAE1D(
        input_length=model_config['input_length'],
        latent_dim=model_config['latent_dim'],
        hidden_dims=model_config['hidden_dims'],
        num_heads=model_config['num_heads'],
        use_input_norm=model_config['use_input_norm'],
        use_batch_norm=model_config['use_batch_norm']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded from {filepath}")
    print(f"Model config: {model_config}")
    
    return model, checkpoint['training_config']

if __name__ == "__main__":
    # Train VAE on combined data
    model, losses = train_vae_combined(
        data_path="/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
        use_both_sequences=True,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100
    )
    
    print("\n🎉 VAE training on combined data completed!")
    print("Next step: Create encoding script to generate latent vectors")
