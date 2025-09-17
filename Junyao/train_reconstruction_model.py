#!/usr/bin/env python3
"""
Training script for the reconstruction MLP model
This model learns to reconstruct original training data from VAE-generated outputs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from reconstruction_model import create_reconstruction_model

class VAEOutputDataset(Dataset):
    """Dataset for VAE-generated outputs and original targets"""
    
    def __init__(self, data_path: str):
        """
        Initialize dataset from VAE-generated outputs
        
        Args:
            data_path: Path to JSON file with VAE outputs
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.generated_outputs = np.array(self.data['generated_outputs'])
        self.original_inputs = np.array(self.data['original_inputs'])
        
        print(f"📊 Loaded dataset:")
        print(f"  Generated outputs shape: {self.generated_outputs.shape}")
        print(f"  Original inputs shape: {self.original_inputs.shape}")
        print(f"  Number of samples: {len(self.generated_outputs)}")
    
    def __len__(self):
        return len(self.generated_outputs)
    
    def __getitem__(self, idx):
        return {
            'vae_output': torch.FloatTensor(self.generated_outputs[idx]),
            'original_target': torch.FloatTensor(self.original_inputs[idx])
        }

def collate_fn(batch):
    """Custom collate function for the dataset"""
    vae_outputs = torch.stack([item['vae_output'] for item in batch])
    original_targets = torch.stack([item['original_target'] for item in batch])
    
    return {
        'vae_output': vae_outputs,
        'original_target': original_targets
    }

def train_reconstruction_model(model_type: str = 'simple', 
                             data_path: str = "vae_generated_outputs.json",
                             batch_size: int = 32,
                             learning_rate: float = 0.001,
                             num_epochs: int = 100,
                             hidden_dims: list = [512, 256, 128],
                             dropout_rate: float = 0.2,
                             device: str = None,
                             save_model: bool = True):
    """
    Train the reconstruction MLP model
    
    Args:
        model_type: Type of model ('simple' or 'advanced')
        data_path: Path to VAE-generated outputs
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
        device: Device to use ('auto', 'cpu', 'mps', 'cuda')
        save_model: Whether to save the trained model
    
    Returns:
        Trained model and training losses
    """
    
    # Device setup
    if device is None or device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"🔧 Using device: {device}")
    
    # Load dataset
    print(f"📥 Loading VAE-generated data from {data_path}...")
    dataset = VAEOutputDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Create model
    print(f"🏗️  Creating {model_type} reconstruction model...")
    model = create_reconstruction_model(
        model_type=model_type,
        input_length=1124,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )
    
    model.to(device)
    print(f"Model info: {model.get_model_info()}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    print(f"🚀 Starting training for {num_epochs} epochs...")
    losses = []
    best_loss = float('inf')
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            vae_outputs = batch_data['vae_output'].to(device)
            original_targets = batch_data['original_target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(vae_outputs)
            loss = criterion(reconstructed, original_targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
        
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
        model_path = f"reconstruction_model_{model_type}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'model_type': model_type,
                'input_length': 1124,
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate
            },
            'training_config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'best_loss': best_loss
            }
        }, model_path)
        print(f"✅ Model saved to {model_path}")
    
    # Plot training results
    plot_training_results(losses, model_type)
    
    # Test reconstruction
    test_reconstruction(model, dataloader, device, model_type)
    
    return model, losses

def plot_training_results(losses, model_type):
    """Plot training results"""
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title(f'{model_type.title()} Reconstruction Model - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Loss reduction
    plt.subplot(1, 2, 2)
    loss_reduction = [losses[0] - loss for loss in losses]
    plt.plot(loss_reduction)
    plt.title('Loss Reduction Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Reduction')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'reconstruction_model_{model_type}_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Training plots saved to reconstruction_model_{model_type}_training_results.png")

def test_reconstruction(model, dataloader, device, model_type, num_examples=5):
    """Test reconstruction quality"""
    print(f"\n🔄 Testing {model_type} reconstruction quality...")
    
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
            vae_outputs = batch_data['vae_output'][:num_examples].to(device)
            original_targets = batch_data['original_target'][:num_examples].to(device)
            
            # Get reconstructions
            reconstructions = model(vae_outputs)
            
            # Calculate metrics
            mse = torch.mean((reconstructions - original_targets) ** 2).item()
            mae = torch.mean(torch.abs(reconstructions - original_targets)).item()
            
            print(f"Test Metrics:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            
            # Visualize results
            fig, axes = plt.subplots(num_examples, 3, figsize=(18, 3*num_examples))
            if num_examples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_examples):
                # VAE output (input to reconstruction model)
                axes[i, 0].plot(vae_outputs[i].cpu().numpy())
                axes[i, 0].set_title(f'VAE Output {i+1}')
                axes[i, 0].set_ylabel('Type ID Value')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Original target
                axes[i, 1].plot(original_targets[i].cpu().numpy())
                axes[i, 1].set_title(f'Original Target {i+1}')
                axes[i, 1].set_ylabel('Type ID Value')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Reconstruction
                axes[i, 2].plot(reconstructions[i].cpu().numpy())
                axes[i, 2].set_title(f'Reconstruction {i+1}')
                axes[i, 2].set_ylabel('Type ID Value')
                axes[i, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'reconstruction_model_{model_type}_test_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Test visualization saved to reconstruction_model_{model_type}_test_results.png")

def load_reconstruction_model(model_path: str):
    """Load a trained reconstruction model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Recreate model
    model_config = checkpoint['model_config']
    model = create_reconstruction_model(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Reconstruction model loaded from {model_path}")
    print(f"Model config: {model_config}")
    
    return model, checkpoint['training_config']

if __name__ == "__main__":
    # Train simple reconstruction model
    print("🚀 Training Simple Reconstruction Model...")
    simple_model, simple_losses = train_reconstruction_model(
        model_type='simple',
        data_path="vae_generated_outputs.json",
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2
    )
    
    print("\n" + "="*60)
    
    # Train advanced reconstruction model
    print("🚀 Training Advanced Reconstruction Model...")
    advanced_model, advanced_losses = train_reconstruction_model(
        model_type='advanced',
        data_path="vae_generated_outputs.json",
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2
    )
    
    print("\n🎉 Both reconstruction models trained successfully!")
    print("Next step: Create end-to-end pipeline script")
