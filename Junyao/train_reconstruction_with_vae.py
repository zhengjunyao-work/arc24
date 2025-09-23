#!/usr/bin/env python3
"""
Train Reconstruction Model using VAE Model's Encode Function
This script trains the reconstruction model by using the VAE model's encode function directly
instead of loading pre-encoded vectors from files.
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
from combined_arc_dataset import CombinedARCDataset, collate_fn
from train_vae_combined import load_model
from reconstruction_model import create_reconstruction_model

class VAEReconstructionDataset(Dataset):
    """Dataset that uses VAE model to encode sequences on-the-fly"""
    
    def __init__(self, data_path: str, vae_model, device: str, use_both_sequences: bool = True):
        """
        Initialize dataset that uses VAE model for encoding
        
        Args:
            data_path: Path to transformed ARC data
            vae_model: Trained VAE model
            device: Device to use for encoding
            use_both_sequences: Whether to use both input and output sequences
        """
        self.data_path = data_path
        self.vae_model = vae_model
        self.device = device
        self.use_both_sequences = use_both_sequences
        
        # Load original data
        print(f"📥 Loading original data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)
        
        # Create sample pool
        self.sample_pool = []
        self.original_info = []
        
        print("🔄 Creating sample pool...")
        for task_idx, (task_id, task_data) in enumerate(self.original_data.items()):
            if 'train' in task_data:
                for example_idx, example in enumerate(task_data['train']):
                    if 'input_type_ids' in example and 'output_type_ids' in example:
                        input_data = torch.tensor(example['input_type_ids'], dtype=torch.float32)
                        output_data = torch.tensor(example['output_type_ids'], dtype=torch.float32)
                        
                        if self.use_both_sequences:
                            # Add both input and output sequences
                            self.sample_pool.append(input_data)
                            self.original_info.append({
                                'task_id': task_id,
                                'task_idx': task_idx,
                                'example_idx': example_idx,
                                'sequence_type': 'input',
                                'original_data': example['input_type_ids']
                            })
                            
                            self.sample_pool.append(output_data)
                            self.original_info.append({
                                'task_id': task_id,
                                'task_idx': task_idx,
                                'example_idx': example_idx,
                                'sequence_type': 'output',
                                'original_data': example['output_type_ids']
                            })
                        else:
                            # Only use output sequences
                            self.sample_pool.append(output_data)
                            self.original_info.append({
                                'task_id': task_id,
                                'task_idx': task_idx,
                                'example_idx': example_idx,
                                'sequence_type': 'output',
                                'original_data': example['output_type_ids']
                            })
        
        print(f"📊 Created sample pool with {len(self.sample_pool)} sequences")
        print(f"Use both sequences: {self.use_both_sequences}")
        
        # Get latent dimension from VAE model
        self.latent_dim = vae_model.latent_dim
        print(f"📊 VAE latent dimension: {self.latent_dim}")
    
    def __len__(self):
        return len(self.sample_pool)
    
    def __getitem__(self, idx):
        # Get original sequence
        original_sequence = self.sample_pool[idx]
        
        # Encode using VAE model
        with torch.no_grad():
            # Ensure sequence is the right shape for VAE
            if original_sequence.dim() == 1:
                original_sequence = original_sequence.unsqueeze(0)  # Add batch dimension
            
            # Move to device
            original_sequence = original_sequence.to(self.device)
            
            # Get the encoded vector using the VAE model
            encoded_vector = self.vae_model.get_encoded_vector(original_sequence)
            encoded_vector = encoded_vector.squeeze(0)  # Remove batch dimension
        
        return {
            'encoded_vector': encoded_vector.cpu(),
            'original_sequence': self.sample_pool[idx]
        }
    
    def get_latent_dim(self):
        """Get the latent dimension from VAE model"""
        return self.latent_dim

def collate_fn(batch):
    """Custom collate function for the dataset"""
    encoded_vectors = torch.stack([item['encoded_vector'] for item in batch])
    original_sequences = torch.stack([item['original_sequence'] for item in batch])
    
    return {
        'encoded_vector': encoded_vectors,
        'original_sequence': original_sequences
    }

def train_reconstruction_with_vae(vae_model_path: str = "vae_combined_trained_model.pth",
                                data_path: str = "/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
                                model_type: str = 'simple',
                                batch_size: int = 32,
                                learning_rate: float = 0.001,
                                num_epochs: int = 5,
                                hidden_dims: list = [512, 256, 128],
                                dropout_rate: float = 0.2,
                                device: str = None,
                                save_model: bool = True):
    """
    Train reconstruction model using VAE model's encode function
    
    Args:
        vae_model_path: Path to trained VAE model
        data_path: Path to transformed ARC data
        model_type: Type of reconstruction model ('simple' or 'advanced')
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
    
    # Load VAE model
    print(f"📥 Loading VAE model from {vae_model_path}...")
    vae_model, _ = load_model(vae_model_path)
    vae_model.to(device)
    vae_model.eval()  # Set to evaluation mode for encoding
    
    # Create dataset that uses VAE model for encoding
    print(f"📥 Creating dataset with VAE encoding...")
    dataset = VAEReconstructionDataset(
        data_path=data_path,
        vae_model=vae_model,
        device=device,
        use_both_sequences=True
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Create reconstruction model
    print(f"🏗️  Creating {model_type} reconstruction model...")
    latent_dim = dataset.get_latent_dim()
    model = create_reconstruction_model(
        model_type=model_type,
        input_length=latent_dim,  # Input: latent dimension (64)
        output_length=1124,  # Output: original sequence length (1124)
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
            encoded_vectors = batch_data['encoded_vector'].to(device)
            original_sequences = batch_data['original_sequence'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(encoded_vectors)
            loss = criterion(reconstructed, original_sequences)
            
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
        model_path = f"reconstruction_model_with_vae_{model_type}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'model_type': model_type,
                'input_length': latent_dim,
                'output_length': 1124,
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate
            },
            'training_config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'best_loss': best_loss
            },
            'vae_config': {
                'vae_model_path': vae_model_path,
                'data_path': data_path
            }
        }, model_path)
        print(f"✅ Model saved to {model_path}")
    
    # Plot training results
    plot_training_results(losses, f"with_vae_{model_type}")
    
    # Test reconstruction
    test_reconstruction_with_vae(model, vae_model, dataloader, device, model_type)
    
    return model, losses

def plot_training_results(losses, model_type):
    """Plot training results"""
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title(f'Reconstruction Model with VAE - Training Loss')
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
    plt.savefig(f'reconstruction_with_vae_{model_type}_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Training plots saved to reconstruction_with_vae_{model_type}_training_results.png")

def test_reconstruction_with_vae(model, vae_model, dataloader, device, model_type, num_examples=5):
    """Test reconstruction quality"""
    print(f"\n🔄 Testing {model_type} reconstruction quality...")
    
    model.eval()
    vae_model.eval()
    
    with torch.no_grad():
        # Get a few test samples
        test_samples = []
        for i, batch_data in enumerate(dataloader):
            if i >= 1:  # Just get first batch
                break
            test_samples.append(batch_data)
        
        if test_samples:
            batch_data = test_samples[0]
            encoded_vectors = batch_data['encoded_vector'][:num_examples].to(device)
            original_sequences = batch_data['original_sequence'][:num_examples].to(device)
            
            # Get reconstructions
            reconstructions = model(encoded_vectors)
            
            # Calculate metrics
            mse = torch.mean((reconstructions - original_sequences) ** 2).item()
            mae = torch.mean(torch.abs(reconstructions - original_sequences)).item()
            
            print(f"Test Metrics:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            
            # Visualize results
            fig, axes = plt.subplots(num_examples, 3, figsize=(18, 3*num_examples))
            if num_examples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_examples):
                # Encoded vector (input to reconstruction model)
                axes[i, 0].plot(encoded_vectors[i].cpu().numpy())
                axes[i, 0].set_title(f'Encoded Vector {i+1}')
                axes[i, 0].set_ylabel('Latent Value')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Original sequence
                axes[i, 1].plot(original_sequences[i].cpu().numpy())
                axes[i, 1].set_title(f'Original Sequence {i+1}')
                axes[i, 1].set_ylabel('Type ID Value')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Reconstruction
                axes[i, 2].plot(reconstructions[i].cpu().numpy())
                axes[i, 2].set_title(f'Reconstruction {i+1}')
                axes[i, 2].set_ylabel('Type ID Value')
                axes[i, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'reconstruction_with_vae_{model_type}_test_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Test visualization saved to reconstruction_with_vae_{model_type}_test_results.png")

def load_reconstruction_model_with_vae(model_path: str):
    """Load a reconstruction model trained with VAE"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    model = create_reconstruction_model(
        model_type=model_config['model_type'],
        input_length=model_config['input_length'],
        output_length=model_config['output_length'],
        hidden_dims=model_config['hidden_dims'],
        dropout_rate=model_config['dropout_rate']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Reconstruction model with VAE loaded from {model_path}")
    return model, checkpoint['training_config']

if __name__ == "__main__":
    # Train simple reconstruction model using VAE
    print("🚀 Training Simple Reconstruction Model with VAE...")
    simple_model, simple_losses = train_reconstruction_with_vae(
        vae_model_path="vae_combined_trained_model.pth",
        data_path="/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
        model_type='simple',
        batch_size=32,
        learning_rate=0.001,
        num_epochs=5,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2
    )
    
    print("\n" + "="*60)
    
    # Train advanced reconstruction model using VAE
    print("🚀 Training Advanced Reconstruction Model with VAE...")
    advanced_model, advanced_losses = train_reconstruction_with_vae(
        vae_model_path="vae_combined_trained_model.pth",
        data_path="/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
        model_type='advanced',
        batch_size=32,
        learning_rate=0.001,
        num_epochs=5,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2
    )
    
    print("\n🎉 Both reconstruction models trained with VAE successfully!")
    print("This approach uses the VAE model's encode function directly instead of pre-encoded vectors")
