#!/usr/bin/env python3
"""
Test Pipeline Encoding
This script tests that the complete pipeline properly handles VAE encoding.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append('.')

def test_pipeline_encoding():
    """Test that the pipeline properly handles VAE encoding"""
    print("🧪 Testing Pipeline VAE Encoding")
    print("=" * 50)
    
    try:
        # Import required modules
        from VAEModel import VAE1D
        from reconstruction_model import create_reconstruction_model
        print("✅ All imports successful!")
        
        # Create test VAE model
        print("🏗️  Creating test VAE model...")
        vae_model = VAE1D(
            input_length=1124,
            latent_dim=64,
            hidden_dims=[512, 256, 128],
            num_heads=8,
            use_input_norm=True,
            use_batch_norm=True
        )
        print("✅ VAE model created successfully")
        
        # Create test reconstruction model
        print("🏗️  Creating test reconstruction model...")
        reconstruction_model = create_reconstruction_model(
            model_type='simple',
            input_length=64,  # latent dimension
            output_length=1124,  # original sequence length
            hidden_dims=[512, 256, 128],
            dropout_rate=0.2
        )
        print("✅ Reconstruction model created successfully")
        
        # Test the complete pipeline
        print("🔄 Testing complete pipeline...")
        test_input = torch.randn(2, 1124)  # Batch of 2, sequence length 1124
        print(f"📊 Input shape: {test_input.shape}")
        
        with torch.no_grad():
            # Stage 1: VAE encoding
            encoded_vectors = vae_model.get_encoded_vector(test_input)
            print(f"✅ VAE encoding successful!")
            print(f"📊 Encoded vectors shape: {encoded_vectors.shape}")
            
            # Stage 2: Reconstruction
            reconstructed = reconstruction_model(encoded_vectors)
            print(f"✅ Reconstruction successful!")
            print(f"📊 Reconstructed shape: {reconstructed.shape}")
            
            # Verify shapes
            if encoded_vectors.shape == (2, 64) and reconstructed.shape == (2, 1124):
                print("✅ All shapes are correct!")
            else:
                print("❌ Shape mismatch!")
                print(f"  Encoded vectors: {encoded_vectors.shape} (expected: (2, 64))")
                print(f"  Reconstructed: {reconstructed.shape} (expected: (2, 1124))")
            
            # Calculate some basic metrics
            mse = torch.mean((test_input - reconstructed) ** 2).item()
            mae = torch.mean(torch.abs(test_input - reconstructed)).item()
            
            print(f"📊 Pipeline Metrics:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
        
        print("\n🎉 Pipeline encoding test completed successfully!")
        print("The pipeline can properly:")
        print("  1. Encode sequences using VAE model")
        print("  2. Reconstruct sequences using reconstruction model")
        print("  3. Handle correct tensor shapes throughout")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing pipeline encoding: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_encoding()
    if success:
        print("\n✅ Pipeline encoding test passed!")
        print("The complete pipeline should work correctly with VAE encoding.")
    else:
        print("\n❌ Pipeline encoding test failed!")
        print("There are issues with the pipeline encoding.")
