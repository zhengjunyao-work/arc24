#!/usr/bin/env python3
"""
Test script to verify VAE normalization features
"""

import torch
import torch.nn as nn
from VAEModel import VAE1D

def test_normalization():
    """Test different normalization configurations"""
    print("Testing VAE Normalization Features")
    print("=" * 40)
    
    # Test parameters
    input_length = 1124
    latent_dim = 64
    hidden_dims = [512, 256, 128]
    num_heads = 8
    batch_size = 4
    
    # Create test data
    test_data = torch.randn(batch_size, input_length)
    print(f"Test data shape: {test_data.shape}")
    
    # Test different normalization configurations
    configs = [
        {"name": "All Normalization", "use_input_norm": True, "use_batch_norm": True},
        {"name": "No Normalization", "use_input_norm": False, "use_batch_norm": False},
        {"name": "Input Norm Only", "use_input_norm": True, "use_batch_norm": False},
        {"name": "Batch Norm Only", "use_input_norm": False, "use_batch_norm": True},
    ]
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 30)
        
        # Create model
        model = VAE1D(
            input_length=input_length,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            use_input_norm=config['use_input_norm'],
            use_batch_norm=config['use_batch_norm']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            try:
                recon_x, mu, logvar = model(test_data)
                print(f"  ✅ Forward pass successful")
                print(f"  Output shape: {recon_x.shape}")
                print(f"  Mu shape: {mu.shape}")
                print(f"  Logvar shape: {logvar.shape}")
                
                # Check output range
                output_min = recon_x.min().item()
                output_max = recon_x.max().item()
                print(f"  Output range: [{output_min:.4f}, {output_max:.4f}]")
                
            except Exception as e:
                print(f"  ❌ Forward pass failed: {e}")
    
    print(f"\n✅ All normalization tests completed!")

def test_data_normalization():
    """Test input data normalization specifically"""
    print(f"\nTesting Input Data Normalization")
    print("=" * 40)
    
    # Create model with input normalization
    model = VAE1D(
        input_length=1124,
        latent_dim=64,
        hidden_dims=[512, 256, 128],
        num_heads=8,
        use_input_norm=True,
        use_batch_norm=True
    )
    
    # Create test data with different scales
    test_cases = [
        ("Normal scale", torch.randn(2, 1124)),
        ("Large scale", torch.randn(2, 1124) * 10),
        ("Small scale", torch.randn(2, 1124) * 0.1),
        ("Mixed scale", torch.cat([torch.randn(1, 1124) * 10, torch.randn(1, 1124) * 0.1], dim=0))
    ]
    
    for name, data in test_cases:
        print(f"\n{name}:")
        print(f"  Input range: [{data.min().item():.4f}, {data.max().item():.4f}]")
        print(f"  Input std: {data.std().item():.4f}")
        
        # Test encoding
        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(data)
            print(f"  Mu range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
            print(f"  Mu std: {mu.std().item():.4f}")

if __name__ == "__main__":
    test_normalization()
    test_data_normalization()
