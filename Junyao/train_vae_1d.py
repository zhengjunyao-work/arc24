import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from VAEModel import VAE1D, loss_function_vae
import matplotlib.pyplot as plt

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

def save_loss_plot(losses, epoch, save_path='vae_1d_training_progress.png'):
    """Save intermediate loss plot during training"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title(f'VAE Training Progress (Epoch {epoch})', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add current loss annotation
    if losses:
        plt.annotate(f'Current Loss: {losses[-1]:.4f}', 
                    xy=(len(losses), losses[-1]), 
                    xytext=(len(losses)-2, losses[-1] + max(losses) * 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory

def train_vae():
    # Hyperparameters optimized for Mac Mini GPU
    batch_size = 64  # Increased batch size for better GPU utilization
    num_epochs = 5
    learning_rate = 1e-3
    input_length = 1124
    latent_dim = 64
    hidden_dims = [512, 256, 128]
    num_heads = 8
    
    # Enhanced device setup for Mac Mini GPU optimization
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using Apple Silicon GPU (MPS): {device}")
        # Enable memory efficient attention if available
        if hasattr(torch.backends.mps, 'enable_memory_efficient_attention'):
            torch.backends.mps.enable_memory_efficient_attention(True)
            print("✅ Enabled memory efficient attention")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using NVIDIA GPU (CUDA): {device}")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using CPU: {device}")
    
    # Set memory fraction for MPS to avoid memory issues
    if device.type == 'mps':
        try:
            # Set memory fraction to 0.8 to leave some memory for system
            torch.mps.set_per_process_memory_fraction(0.8)
            print("✅ Set MPS memory fraction to 0.8")
        except:
            print("⚠️  Could not set MPS memory fraction")
    
    # Load transformed data
    transformed_data_path = '/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json'
    
    try:
        print("Loading transformed ARC data...")
        transformed_data = load_transformed_data(transformed_data_path)
    except FileNotFoundError:
        print(f"Error: Could not find transformed data file at {transformed_data_path}")
        print("Please run the transformation script first to generate the transformed data.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = TransformedARCDataset(
        transformed_data, 
        use_input_type_ids=True, 
        use_output_type_ids=True
    )
    
    # Create data loader optimized for GPU
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,  # Increased for better data loading
        pin_memory=True if device.type != 'cpu' else False,  # Pin memory for GPU
        persistent_workers=True if device.type != 'cpu' else False  # Keep workers alive
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {batch_size}")
    print(f"Estimated memory per batch: {batch_size * input_length * 4 / 1024 / 1024:.2f} MB")
    
    # Enable mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    if scaler:
        print("✅ Enabled mixed precision training (CUDA)")
    elif device.type == 'mps':
        print("✅ Using MPS device (mixed precision handled automatically)")
    else:
        print("⚠️  Using CPU (no mixed precision)")
    
    # Create model with attention and normalization
    print("Initializing VAE model with attention and normalization...")
    model = VAE1D(
        input_length=input_length,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        num_heads=num_heads,
        use_input_norm=True,      # Normalize input data
        use_batch_norm=True       # Use batch normalization
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    model.train()
    losses = []
    best_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    
    # Performance monitoring
    import time
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, data in enumerate(dataloader):
            # Move data to device
            data = data.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            if scaler:  # CUDA mixed precision
                with torch.cuda.amp.autocast():
                    recon_batch, mu, logvar = model(data)
                    loss = loss_function_vae(recon_batch, data, mu, logvar, beta=1.0)
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  # MPS or CPU
                recon_batch, mu, logvar = model(data)
                loss = loss_function_vae(recon_batch, data, mu, logvar, beta=1.0)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress every 10 batches with timing
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (epoch * len(dataloader) + batch_idx + 1) / elapsed
                print(f'Epoch {epoch+1}, Batch {batch_idx + 1}/{len(dataloader)}, '
                      f'Loss: {loss.item():.4f}, Speed: {batches_per_sec:.2f} batches/sec')
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, '
              f'Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s')
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'model_config': {
                    'input_length': input_length,
                    'latent_dim': latent_dim,
                    'hidden_dims': hidden_dims,
                    'num_heads': num_heads,
                    'use_input_norm': True,
                    'use_batch_norm': True
                }
            }, "vae_1d_attention_best.pth")
            print(f"  -> New best model saved! Loss: {best_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'model_config': {
                    'input_length': input_length,
                    'latent_dim': latent_dim,
                    'hidden_dims': hidden_dims,
                    'num_heads': num_heads,
                    'use_input_norm': True,
                    'use_batch_norm': True
                }
            }, f"vae_1d_attention_checkpoint_epoch_{epoch+1}.pth")
            
            # Save intermediate loss plot
            save_loss_plot(losses, epoch + 1, f'vae_1d_training_progress_epoch_{epoch+1}.png')
            print(f"  -> Training progress plot saved")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'model_config': {
            'input_length': input_length,
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'num_heads': num_heads,
            'use_input_norm': True,
            'use_batch_norm': True
        }
    }, "vae_1d_attention_final.pth")
    
    print(f"Training completed!")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Models saved:")
    print(f"  - Best model: vae_1d_attention_best.pth")
    print(f"  - Final model: vae_1d_attention_final.pth")
    
    # Enhanced training loss plotting
    print("\nGenerating training loss plots...")
    
    # Create detailed loss plot
    plt.figure(figsize=(12, 8))
    
    # Main loss plot
    plt.subplot(2, 2, 1)
    plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
    plt.title('VAE Training Loss (ARC Data)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Loss trend (smoothed)
    plt.subplot(2, 2, 2)
    if len(losses) > 1:
        # Calculate moving average for smoothing
        window_size = min(5, len(losses) // 2)
        if window_size > 1:
            smoothed_losses = []
            for i in range(len(losses)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(losses), i + window_size // 2 + 1)
                smoothed_losses.append(np.mean(losses[start_idx:end_idx]))
            plt.plot(smoothed_losses, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
        else:
            smoothed_losses = losses
            plt.plot(smoothed_losses, 'r-', linewidth=2, label='Training Loss')
    else:
        plt.plot(losses, 'r-', linewidth=2, label='Training Loss')
    
    plt.title('Smoothed Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Loss statistics
    plt.subplot(2, 2, 3)
    loss_stats = {
        'Min Loss': min(losses),
        'Max Loss': max(losses),
        'Final Loss': losses[-1],
        'Best Loss': best_loss,
        'Loss Reduction': losses[0] - losses[-1] if len(losses) > 1 else 0
    }
    
    plt.bar(loss_stats.keys(), loss_stats.values(), color=['green', 'red', 'blue', 'orange', 'purple'])
    plt.title('Loss Statistics', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (key, value) in enumerate(loss_stats.items()):
        plt.text(i, value + max(loss_stats.values()) * 0.01, f'{value:.4f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Training progress
    plt.subplot(2, 2, 4)
    epochs = list(range(1, len(losses) + 1))
    plt.plot(epochs, losses, 'g-', linewidth=2, marker='o', markersize=4)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add best loss marker
    best_epoch = losses.index(best_loss) + 1
    plt.plot(best_epoch, best_loss, 'ro', markersize=8, label=f'Best Loss: {best_loss:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('vae_1d_training_loss_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save simple loss plot as well
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('VAE Training Loss (ARC Data)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('vae_1d_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save loss data to file
    loss_data = {
        'epochs': list(range(1, len(losses) + 1)),
        'losses': losses,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'final_loss': losses[-1],
        'loss_reduction': losses[0] - losses[-1] if len(losses) > 1 else 0
    }
    
    import json
    with open('vae_1d_training_loss_data.json', 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    print("Training loss plots saved:")
    print("  - vae_1d_training_loss_detailed.png (4-panel detailed view)")
    print("  - vae_1d_training_loss.png (simple view)")
    print("  - vae_1d_training_loss_data.json (loss data)")
    
    # Test reconstruction on a few samples
    model.eval()
    with torch.no_grad():
        # Get a few samples from the dataset
        test_samples = []
        for i in range(min(5, len(train_dataset))):
            test_samples.append(train_dataset[i])
        
        test_data = torch.stack(test_samples).to(device)
        recon_data, _, _ = model(test_data)
        
        # Plot original vs reconstructed
        fig, axes = plt.subplots(5, 2, figsize=(15, 12))
        for i in range(5):
            # Original data
            axes[i, 0].plot(test_data[i].cpu().numpy())
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
        plt.savefig('vae_1d_reconstruction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return model

def load_and_sample(model_path='vae_1d_attention_best.pth'):
    """
    Load trained model and generate samples
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with the same configuration
    model_config = checkpoint['model_config']
    model = VAE1D(
        input_length=model_config['input_length'],
        latent_dim=model_config['latent_dim'],
        hidden_dims=model_config['hidden_dims'],
        num_heads=model_config['num_heads'],
        use_input_norm=model_config.get('use_input_norm', True),
        use_batch_norm=model_config.get('use_batch_norm', True)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    print(f"Epoch: {checkpoint['epoch']}")
    
    # Generate samples
    with torch.no_grad():
        samples = model.sample(num_samples=5, device=device)
        
        # Plot generated samples
        plt.figure(figsize=(15, 10))
        for i in range(5):
            plt.subplot(5, 1, i+1)
            sample_data = samples[i].squeeze().cpu().numpy()
            plt.plot(sample_data)
            plt.title(f'Generated ARC Sample {i+1}')
            plt.ylabel('Type ID Value')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('vae_1d_generated_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Generated samples saved to vae_1d_generated_samples.png")

if __name__ == "__main__":
    # Train the model
    model = train_vae()
    
    # Generate samples from trained model
    load_and_sample() 