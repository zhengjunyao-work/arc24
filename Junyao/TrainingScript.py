import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from VAEModel import VAE1D, loss_function_vae  # Import the new VAE1D model

class TransformedARCDataset(Dataset):
    """
    Dataset for loading transformed ARC data with type IDs
    """
    def __init__(self, data_dict, use_input_type_ids=True, use_output_type_ids=True):
        self.data_dict = data_dict
        self.use_input_type_ids = use_input_type_ids
        self.use_output_type_ids = use_output_type_ids
        self.sample_pool = []
        self.create_sample_pool()
    
    def create_sample_pool(self):
        """Create a pool of samples from the transformed dataset"""
        for task_id, task_data in self.data_dict.items():
            # Process train examples
            if 'train' in task_data:
                for example in task_data['train']:
                    # Use input_type_ids as primary data
                    if self.use_input_type_ids and 'input_type_ids' in example:
                        input_data = torch.tensor(example['input_type_ids'], dtype=torch.float32)
                        self.sample_pool.append(input_data)
                    
                    # Use output_type_ids as additional data
                    if self.use_output_type_ids and 'output_type_ids' in example:
                        output_data = torch.tensor(example['output_type_ids'], dtype=torch.float32)
                        self.sample_pool.append(output_data)
            
            # Process test examples
            if 'test' in task_data:
                for example in task_data['test']:
                    if self.use_input_type_ids and 'input_type_ids' in example:
                        input_data = torch.tensor(example['input_type_ids'], dtype=torch.float32)
                        self.sample_pool.append(input_data)
                    
                    if self.use_output_type_ids and 'output_type_ids' in example:
                        output_data = torch.tensor(example['output_type_ids'], dtype=torch.float32)
                        self.sample_pool.append(output_data)
        
        print(f"Created sample pool with {len(self.sample_pool)} samples")
        
        # Print sample statistics
        if self.sample_pool:
            sample_lengths = [len(sample) for sample in self.sample_pool]
            print(f"Sample length statistics:")
            print(f"  - Min length: {min(sample_lengths)}")
            print(f"  - Max length: {max(sample_lengths)}")
            print(f"  - Mean length: {np.mean(sample_lengths):.2f}")
            print(f"  - Std length: {np.std(sample_lengths):.2f}")
    
    def __len__(self):
        return len(self.sample_pool)
    
    def __getitem__(self, idx):
        return self.sample_pool[idx]

def load_transformed_data(file_path):
    """Load transformed ARC data from JSON file"""
    print(f"Loading transformed data from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} tasks")
    return data

def pad_sequences(sequences, target_length=1124):
    """Pad or truncate sequences to target length"""
    padded_sequences = []
    for seq in sequences:
        if len(seq) < target_length:
            # Pad with zeros
            padded_seq = torch.cat([seq, torch.zeros(target_length - len(seq))])
        elif len(seq) > target_length:
            # Truncate
            padded_seq = seq[:target_length]
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    return torch.stack(padded_sequences)

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    # Pad all sequences to the same length
    padded_batch = pad_sequences(batch, target_length=1124)
    return padded_batch
    
def main():
    """Main training function"""
    print("VAE Training with Transformed ARC Data")
    print("=" * 50)
    
    # File path for transformed data
    transformed_data_path = '/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json'
    
    # Load transformed data
    try:
        transformed_data = load_transformed_data(transformed_data_path)
    except FileNotFoundError:
        print(f"Error: Could not find transformed data file at {transformed_data_path}")
        print("Please run the transformation script first to generate the transformed data.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Hyperparameters
    batch_size = 32
    epochs = 50
    learning_rate = 1e-3
    input_length = 1124
    latent_dim = 64
    hidden_dims = [512, 256, 128]
    num_heads = 8
    
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    print("\nCreating dataset...")
    train_dataset = TransformedARCDataset(
        transformed_data, 
        use_input_type_ids=True, 
        use_output_type_ids=True
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging, increase for production
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of batches: {len(train_loader)}")
    
    # Create model
    print("\nInitializing VAE model...")
    model = VAE1D(
        input_length=input_length,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        num_heads=num_heads
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    model.train()
    
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, data in enumerate(train_loader):
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # Calculate loss
            loss = loss_function_vae(recon_batch, data, mu, logvar, beta=1.0)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'model_config': {
                    'input_length': input_length,
                    'latent_dim': latent_dim,
                    'hidden_dims': hidden_dims,
                    'num_heads': num_heads
                }
            }, "vae_attention_best.pth")
            print(f"  -> New best model saved! Loss: {best_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'model_config': {
                    'input_length': input_length,
                    'latent_dim': latent_dim,
                    'hidden_dims': hidden_dims,
                    'num_heads': num_heads
                }
            }, f"vae_attention_checkpoint_epoch_{epoch}.pth")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'model_config': {
            'input_length': input_length,
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'num_heads': num_heads
        }
    }, "vae_attention_final.pth")
    
    print(f"\nTraining completed!")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Models saved:")
    print(f"  - Best model: vae_attention_best.pth")
    print(f"  - Final model: vae_attention_final.pth")

if __name__ == "__main__":
    main()