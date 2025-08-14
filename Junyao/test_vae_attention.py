#!/usr/bin/env python3
"""
Test script for VAE1D with attention mechanisms
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from VAEModel import VAE1D, loss_function_vae

def create_sample_data(batch_size=32, seq_length=1124):
    """Create sample data for testing"""
    # Generate random sequences
    data = torch.rand(batch_size, seq_length)
    return data

def test_vae_attention():
    """Test the VAE with attention"""
    print("Testing VAE1D with Attention Mechanisms")
    print("=" * 50)
    
    # Model parameters
    input_length = 1124
    latent_dim = 64
    hidden_dims = [512, 256, 128]
    num_heads = 8
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE1D(
        input_length=input_length,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        num_heads=num_heads
    ).to(device)
    
    print(f"Model created on device: {device}")
    print(f"Input length: {input_length}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Number of attention heads: {num_heads}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create sample data
    batch_size = 4
    x = create_sample_data(batch_size, input_length).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        recon_x, mu, logvar = model(x)
        print(f"Reconstructed output shape: {recon_x.shape}")
        print(f"Latent mean shape: {mu.shape}")
        print(f"Latent logvar shape: {logvar.shape}")
        
        # Check if output length matches input
        assert recon_x.shape[-1] == input_length, f"Output length {recon_x.shape[-1]} != input length {input_length}"
        print("✅ Output length matches input length")
    
    # Test encoding
    print("\nTesting encoding...")
    mu, logvar = model.encode(x)
    print(f"Encoded mean shape: {mu.shape}")
    print(f"Encoded logvar shape: {logvar.shape}")
    
    # Test decoding
    print("\nTesting decoding...")
    z = model.reparameterize(mu, logvar)
    recon_x = model.decode(z)
    print(f"Decoded output shape: {recon_x.shape}")
    
    # Test sampling
    print("\nTesting sampling...")
    samples = model.sample(num_samples=2, device=device)
    print(f"Sampled output shape: {samples.shape}")
    
    # Test loss function
    print("\nTesting loss function...")
    loss = loss_function_vae(recon_x, x, mu, logvar)
    print(f"VAE loss: {loss.item():.4f}")
    
    print("\n✅ All tests passed!")

def train_vae_attention(epochs=10):
    """Train the VAE with attention"""
    print("\nTraining VAE1D with Attention")
    print("=" * 40)
    
    # Model parameters
    input_length = 1124
    latent_dim = 64
    hidden_dims = [512, 256, 128]
    num_heads = 8
    
    # Create model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE1D(
        input_length=input_length,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        num_heads=num_heads
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        # Create batch
        batch_size = 16
        x = create_sample_data(batch_size, input_length).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        
        # Calculate loss
        loss = loss_function_vae(recon_x, x, mu, logvar)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    # Test the model
    test_vae_attention()
    
    # Train the model
    train_vae_attention(epochs=5)
