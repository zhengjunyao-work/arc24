#!/usr/bin/env python3
"""
Train Reconstruction Model from Encoded Vectors
This script trains the reconstruction model using pre-encoded latent vectors
instead of generating them on-the-fly from VAE outputs.
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
from encode_training_data import load_encoded_vectors, get_vector_by_original_index

class EncodedVectorDataset(Dataset):
    """Dataset for encoded vectors and their original targets"""
    
    def __init__(self, vectors_path: str, info_path: str, 
                 original_data_path: str, use_both_sequences: bool = True):
        """
        Initialize dataset from encoded vectors
        
        Args:
            vectors_path: Path to encoded vectors .npy file
            info_path: Path to original info .json file
            original_data_path: Path to original transformed ARC data
            use_both_sequences: Whether to use both input and output sequences
        """
        # Load encoded vectors and info
        self.encoded_vectors, self.original_info = load_encoded_vectors(vectors_path, info_path)
        
        print(f"📊 Loaded encoded vectors:")
        print(f"  - Shape: {self.encoded_vectors.shape}")
        print(f"  - Latent dimension: {self.encoded_vectors.shape[1]}")
        print(f"  - Number of vectors: {self.encoded_vectors.shape[0]}")
        
        # Load original data for targets
        print(f"📥 Loading original data from {original_data_path}...")
        with open(original_data_path, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)
        
        self.use_both_sequences = use_both_sequences
        
        # Create target sequences
        self.target_sequences = []
        self.vector_indices = []
        
        print("🔄 Creating target sequences...")
        for i, info in enumerate(self.original_info):
            task_id = info['task_id']
            example_idx = info['example_idx']
            sequence_type = info['sequence_type']
            
            # Get original sequence as target
            if (task_id in self.original_data and 
                'train' in self.original_data[task_id] and
                example_idx < len(self.original_data[task_id]['train'])):
                
                example = self.original_data[task_id]['train'][example_idx]
                
                if sequence_type == 'input' and 'input_type_ids' in example:
                    target_sequence = example['input_type_ids']
                elif sequence_type == 'output' and 'output_type_ids' in example:
                    target_sequence = example['output_type_ids']
                else:
                    continue
                
                self.target_sequences.append(torch.FloatTensor(target_sequence))
                self.vector_indices.append(i)
        
        print(f"📊 Dataset created:")
        print(f"  - Encoded vectors: {self.encoded_vectors.shape}")
        print(f"  - Target sequences: {len(self.target_sequences)}")
        print(f"  - Use both sequences: {use_both_sequences}")
    
    def __len__(self):
        return len(self.target_sequences)
    
    def __getitem__(self, idx):
        vector_idx = self.vector_indices[idx]
        return {
            'encoded_vector': torch.FloatTensor(self.encoded_vectors[vector_idx]),
            'target_sequence': self.target_sequences[idx]
        }
    
    def get_latent_dim(self):
        """Get the latent dimension of the encoded vectors"""
        return self.encoded_vectors.shape[1]
    
    def get_num_vectors(self):
        """Get the number of encoded vectors"""
        return self.encoded_vectors.shape[0]

def collate_fn(batch):
    """Custom collate function for the dataset"""
    encoded_vectors = torch.stack([item['encoded_vector'] for item in batch])
    target_sequences = torch.stack([item['target_sequence'] for item in batch])
    
    return {
        'encoded_vector': encoded_vectors,
        'target_sequence': target_sequences
    }

def train_reconstruction_from_vectors(model_type: str = 'simple',
                                    vectors_path: str = "encoded_vectors/training_encoded_vectors.npy",
                                    info_path: str = "encoded_vectors/training_encoded_vectors_info.json",
                                    original_data_path: str = "/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
                                    batch_size: int = 32,
                                    learning_rate: float = 0.001,
                                    num_epochs: int = 100,
                                    hidden_dims: list = [512, 256, 128],
                                    dropout_rate: float = 0.2,
                                    device: str = None,
                                    save_model: bool = True):
    """
    Train reconstruction model from encoded vectors
    
    Args:
        model_type: Type of model ('simple' or 'advanced')
        vectors_path: Path to encoded vectors
        info_path: Path to original info
        original_data_path: Path to original transformed data
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
    print(f"📥 Loading encoded vector dataset...")
    dataset = EncodedVectorDataset(
        vectors_path=vectors_path,
        info_path=info_path,
        original_data_path=original_data_path,
        use_both_sequences=True
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Create model
    print(f"🏗️  Creating {model_type} reconstruction model...")
    latent_dim = dataset.get_latent_dim()
    model = create_reconstruction_model(
        model_type=model_type,
        input_length=latent_dim,  # Input: latent dimension (64)
        output_length=1124,  # Output: original sequence length (1124)
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )
    
    # The model should output 1124 dimensions (original sequence length)
    # Let's verify the target sequence length
    if dataset.target_sequences:
        target_length = len(dataset.target_sequences[0])
        print(f"📊 Model configuration:")
        print(f"  - Input length (latent): {latent_dim}")
        print(f"  - Output length (target): {target_length}")
        print(f"  - Expected output: 1124")
        
        if target_length != 1124:
            print(f"⚠️  Warning: Target sequence length ({target_length}) doesn't match expected (1124)")
    
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
            target_sequences = batch_data['target_sequence'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(encoded_vectors)
            loss = criterion(reconstructed, target_sequences)
            
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
        model_path = f"reconstruction_model_from_vectors_{model_type}.pth"
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
            'data_config': {
                'vectors_path': vectors_path,
                'info_path': info_path,
                'original_data_path': original_data_path
            }
        }, model_path)
        print(f"✅ Model saved to {model_path}")
    
    # Plot training results
    plot_training_results(losses, f"from_vectors_{model_type}")
    
    # Test reconstruction
    test_reconstruction_from_vectors(model, dataloader, device, model_type)
    
    return model, losses

def plot_training_results(losses, model_type):
    """Plot training results"""
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title(f'Reconstruction Model from Vectors - Training Loss')
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
    plt.savefig(f'reconstruction_from_vectors_{model_type}_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Training plots saved to reconstruction_from_vectors_{model_type}_training_results.png")

def test_reconstruction_from_vectors(model, dataloader, device, model_type, num_examples=5):
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
            encoded_vectors = batch_data['encoded_vector'][:num_examples].to(device)
            target_sequences = batch_data['target_sequence'][:num_examples].to(device)
            
            # Get reconstructions
            reconstructions = model(encoded_vectors)
            
            # Calculate metrics
            mse = torch.mean((reconstructions - target_sequences) ** 2).item()
            mae = torch.mean(torch.abs(reconstructions - target_sequences)).item()
            
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
                
                # Target sequence
                axes[i, 1].plot(target_sequences[i].cpu().numpy())
                axes[i, 1].set_title(f'Target Sequence {i+1}')
                axes[i, 1].set_ylabel('Type ID Value')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Reconstruction
                axes[i, 2].plot(reconstructions[i].cpu().numpy())
                axes[i, 2].set_title(f'Reconstruction {i+1}')
                axes[i, 2].set_ylabel('Type ID Value')
                axes[i, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'reconstruction_from_vectors_{model_type}_test_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Test visualization saved to reconstruction_from_vectors_{model_type}_test_results.png")

def load_reconstruction_model_from_vectors(model_path: str):
    """Load a trained reconstruction model from vectors"""
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
    # Train simple reconstruction model from vectors
    print("🚀 Training Simple Reconstruction Model from Vectors...")
    simple_model, simple_losses = train_reconstruction_from_vectors(
        model_type='simple',
        vectors_path="encoded_vectors/training_encoded_vectors.npy",
        info_path="encoded_vectors/training_encoded_vectors_info.json",
        original_data_path="/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
        batch_size=32,
        learning_rate=0.001,
        num_epochs=5,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2
    )
    
    print("\n" + "="*60)
    
    # Train advanced reconstruction model from vectors
    print("🚀 Training Advanced Reconstruction Model from Vectors...")
    advanced_model, advanced_losses = train_reconstruction_from_vectors(
        model_type='advanced',
        vectors_path="encoded_vectors/training_encoded_vectors.npy",
        info_path="encoded_vectors/training_encoded_vectors_info.json",
        original_data_path="/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
        batch_size=32,
        learning_rate=0.001,
        num_epochs=5,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2
    )
    
    print("\n🎉 Both reconstruction models trained from vectors successfully!")
    print("Next step: Create complete pipeline script")
