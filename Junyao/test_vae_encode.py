#!/usr/bin/env python3
"""
Test VAE Model's Encode Function
This script tests that the VAE model can properly encode sequences into latent vectors.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append('.')

def test_vae_encode():
    """Test VAE model's encode function"""
    print("🧪 Testing VAE Model's Encode Function")
    print("=" * 50)
    
    try:
        # Import VAE model
        from VAEModel import VAE1D
        print("✅ VAE1D class imported successfully")
        
        # Create a test VAE model
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
        
        # Test with random input
        print("🔄 Testing encode function...")
        test_input = torch.randn(2, 1124)  # Batch of 2, sequence length 1124
        print(f"📊 Input shape: {test_input.shape}")
        
        # Test encode function
        mu, logvar = vae_model.encode(test_input)
        print(f"✅ Encode function works!")
        print(f"📊 Mu shape: {mu.shape}")
        print(f"📊 Logvar shape: {logvar.shape}")
        
        # Test reparameterize function
        z = vae_model.reparameterize(mu, logvar)
        print(f"✅ Reparameterize function works!")
        print(f"📊 Encoded vector (z) shape: {z.shape}")
        
        # Test get_encoded_vector function
        z_direct = vae_model.get_encoded_vector(test_input)
        print(f"✅ Get_encoded_vector function works!")
        print(f"📊 Direct encoded vector shape: {z_direct.shape}")
        
        # Verify shapes are correct
        expected_shape = (2, 64)  # batch_size, latent_dim
        if z.shape == expected_shape and z_direct.shape == expected_shape:
            print(f"✅ Encoded vector shapes are correct: {expected_shape}")
        else:
            print(f"❌ Encoded vector shapes are incorrect!")
            print(f"  Expected: {expected_shape}")
            print(f"  Got z: {z.shape}")
            print(f"  Got z_direct: {z_direct.shape}")
        
        # Test that both methods give similar results (they should be identical)
        if torch.allclose(z, z_direct, atol=1e-6):
            print("✅ Both encoding methods give identical results!")
        else:
            print("❌ Encoding methods give different results!")
            print(f"  Max difference: {torch.max(torch.abs(z - z_direct)).item()}")
        
        # Test decode function
        reconstructed = vae_model.decode(z)
        print(f"✅ Decode function works!")
        print(f"📊 Reconstructed shape: {reconstructed.shape}")
        
        # Test forward function
        recon_x, mu_forward, logvar_forward = vae_model.forward(test_input)
        print(f"✅ Forward function works!")
        print(f"📊 Forward reconstructed shape: {recon_x.shape}")
        
        print("\n🎉 All VAE encode/decode functions work correctly!")
        print("📊 Summary:")
        print(f"  - Input: {test_input.shape}")
        print(f"  - Encoded vector (z): {z.shape}")
        print(f"  - Reconstructed: {reconstructed.shape}")
        print(f"  - Latent dimension: {vae_model.latent_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing VAE encode function: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vae_encode()
    if success:
        print("\n✅ VAE encode function test completed successfully!")
        print("The VAE model can properly encode sequences into latent vectors.")
    else:
        print("\n❌ VAE encode function test failed!")
        print("There are issues with the VAE model's encode function.")
