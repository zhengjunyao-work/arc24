#!/usr/bin/env python3
"""
Complete Vector-Based Pipeline
This script implements the complete pipeline:
1. Train VAE on combined ARC data (input + output sequences)
2. Encode all training data into latent vectors with original indices
3. Train reconstruction model using the encoded vectors
4. Test the complete pipeline
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from train_vae_combined import train_vae_combined, save_model, load_model
from encode_training_data import LatentVectorEncoder
from train_reconstruction_from_vectors import train_reconstruction_from_vectors, load_reconstruction_model_from_vectors
from reconstruction_model import create_reconstruction_model

class CompleteVectorPipeline:
    """Complete pipeline using encoded vectors"""
    
    def __init__(self, device: str = None):
        """
        Initialize the pipeline
        
        Args:
            device: Device to use ('auto', 'cpu', 'mps', 'cuda')
        """
        # Device setup
        if device is None or device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"🔧 Pipeline using device: {self.device}")
        
        # Model storage
        self.vae_model = None
        self.reconstruction_model = None
        self.encoded_vectors = None
        self.original_info = None
        
        # File paths
        self.vae_model_path = "vae_combined_trained_model.pth"
        self.vectors_path = "encoded_vectors/training_encoded_vectors.npy"
        self.info_path = "encoded_vectors/training_encoded_vectors_info.json"
        self.reconstruction_model_path = "reconstruction_model_from_vectors_simple.pth"
        self.data_path = "/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json"
    
    def stage1_train_vae(self, 
                        num_epochs: int = 50,
                        batch_size: int = 32,
                        learning_rate: float = 0.001,
                        use_both_sequences: bool = True):
        """
        Stage 1: Train VAE on combined ARC data
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_both_sequences: Whether to use both input and output sequences
        """
        print("🎯 STAGE 1: Training VAE on Combined Data")
        print("=" * 50)
        
        # Train VAE
        self.vae_model, losses = train_vae_combined(
            data_path=self.data_path,
            use_both_sequences=use_both_sequences,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=self.device,
            save_model=True
        )
        
        print(f"✅ Stage 1 completed: VAE trained and saved to {self.vae_model_path}")
    
    def stage2_encode_data(self, 
                           batch_size: int = 32,
                           use_both_sequences: bool = True):
        """
        Stage 2: Encode all training data into latent vectors
        
        Args:
            batch_size: Batch size for encoding
            use_both_sequences: Whether to encode both input and output sequences
        """
        print("🎯 STAGE 2: Encoding Training Data into Vectors")
        print("=" * 50)
        
        # Create encoder
        encoder = LatentVectorEncoder(
            model_path=self.vae_model_path,
            device=self.device
        )
        
        # Encode and save
        self.encoded_vectors, self.original_info, index_mapping = encoder.encode_and_save(
            data_path=self.data_path,
            use_both_sequences=use_both_sequences,
            batch_size=batch_size,
            filename_prefix="training_encoded_vectors"
        )
        
        print(f"✅ Stage 2 completed: Encoded vectors saved")
        print(f"📊 Encoded vectors shape: {self.encoded_vectors.shape}")
        print(f"📊 Original info entries: {len(self.original_info)}")
    
    def stage3_train_reconstruction(self,
                                   model_type: str = 'simple',
                                   batch_size: int = 32,
                                   learning_rate: float = 0.001,
                                   num_epochs: int = 50):
        """
        Stage 3: Train reconstruction model using encoded vectors
        
        Args:
            model_type: Type of reconstruction model ('simple' or 'advanced')
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
        """
        print("🎯 STAGE 3: Training Reconstruction Model from Vectors")
        print("=" * 50)
        
        # Train reconstruction model
        self.reconstruction_model, losses = train_reconstruction_from_vectors(
            model_type=model_type,
            vectors_path=self.vectors_path,
            info_path=self.info_path,
            original_data_path=self.data_path,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=self.device,
            save_model=True
        )
        
        # Update model path
        self.reconstruction_model_path = f"reconstruction_model_from_vectors_{model_type}.pth"
        
        print(f"✅ Stage 3 completed: Reconstruction model trained and saved")
    
    def test_complete_pipeline(self, num_test_samples: int = 5):
        """
        Test the complete pipeline
        
        Args:
            num_test_samples: Number of test samples to evaluate
        """
        print("🧪 TESTING COMPLETE VECTOR-BASED PIPELINE")
        print("=" * 50)
        
        if self.vae_model is None:
            print("Loading VAE model...")
            self.vae_model, _ = load_model(self.vae_model_path)
            self.vae_model.to(self.device)
        
        if self.reconstruction_model is None:
            print("Loading reconstruction model...")
            self.reconstruction_model, _ = load_reconstruction_model_from_vectors(self.reconstruction_model_path)
            self.reconstruction_model.to(self.device)
        
        # Load test data
        from combined_arc_dataset import CombinedARCDataset, collate_fn
        from torch.utils.data import DataLoader
        
        dataset = CombinedARCDataset(self.data_path, use_both_sequences=True)
        dataloader = DataLoader(dataset, batch_size=num_test_samples, shuffle=True, collate_fn=collate_fn)
        
        # Get test batch
        test_batch = next(iter(dataloader))
        original_data = test_batch.to(self.device)
        
        print(f"Testing with {num_test_samples} samples...")
        
        # Run through pipeline
        with torch.no_grad():
            # Stage 1: VAE encoding
            encoded_vectors, mu, logvar = self.vae_model.encode(original_data)
            
            # Stage 2: Reconstruction from encoded vectors
            final_reconstruction = self.reconstruction_model(encoded_vectors)
            
            # Calculate metrics
            encoding_mse = torch.mean((original_data - self.vae_model.decode(encoded_vectors)) ** 2).item()
            encoding_mae = torch.mean(torch.abs(original_data - self.vae_model.decode(encoded_vectors))).item()
            
            final_mse = torch.mean((original_data - final_reconstruction) ** 2).item()
            final_mae = torch.mean(torch.abs(original_data - final_reconstruction)).item()
            
            print(f"\n📊 Pipeline Performance:")
            print(f"VAE Encoding/Decoding:")
            print(f"  MSE: {encoding_mse:.6f}")
            print(f"  MAE: {encoding_mae:.6f}")
            print(f"Final Reconstruction:")
            print(f"  MSE: {final_mse:.6f}")
            print(f"  MAE: {final_mae:.6f}")
            
            # Visualize results
            self.visualize_complete_pipeline(
                original_data, encoded_vectors, final_reconstruction, num_test_samples
            )
    
    def visualize_complete_pipeline(self, original_data, encoded_vectors, final_reconstruction, num_samples):
        """Visualize complete pipeline results"""
        print("📊 Creating complete pipeline visualization...")
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(24, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original data
            axes[i, 0].plot(original_data[i].cpu().numpy())
            axes[i, 0].set_title(f'Original ARC Data {i+1}')
            axes[i, 0].set_ylabel('Type ID Value')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Encoded vector
            axes[i, 1].plot(encoded_vectors[i].cpu().numpy())
            axes[i, 1].set_title(f'Encoded Vector {i+1}')
            axes[i, 1].set_ylabel('Latent Value')
            axes[i, 1].grid(True, alpha=0.3)
            
            # VAE reconstruction
            vae_recon = self.vae_model.decode(encoded_vectors[i:i+1])[0]
            axes[i, 2].plot(vae_recon.cpu().numpy())
            axes[i, 2].set_title(f'VAE Reconstruction {i+1}')
            axes[i, 2].set_ylabel('Type ID Value')
            axes[i, 2].grid(True, alpha=0.3)
            
            # Final reconstruction
            axes[i, 3].plot(final_reconstruction[i].cpu().numpy())
            axes[i, 3].set_title(f'Final Reconstruction {i+1}')
            axes[i, 3].set_ylabel('Type ID Value')
            axes[i, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('complete_vector_pipeline_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Complete pipeline visualization saved to complete_vector_pipeline_results.png")
    
    def run_complete_pipeline(self, 
                             vae_epochs: int = 50,
                             reconstruction_epochs: int = 50,
                             use_both_sequences: bool = True,
                             reconstruction_model_type: str = 'simple'):
        """
        Run the complete vector-based pipeline
        
        Args:
            vae_epochs: Number of epochs for VAE training
            reconstruction_epochs: Number of epochs for reconstruction training
            use_both_sequences: Whether to use both input and output sequences
            reconstruction_model_type: Type of reconstruction model
        """
        print("🚀 RUNNING COMPLETE VECTOR-BASED PIPELINE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Stage 1: Train VAE on combined data
        self.stage1_train_vae(
            num_epochs=vae_epochs,
            use_both_sequences=use_both_sequences
        )
        
        # Stage 2: Encode all training data
        self.stage2_encode_data(use_both_sequences=use_both_sequences)
        
        # Stage 3: Train reconstruction model
        self.stage3_train_reconstruction(
            model_type=reconstruction_model_type,
            num_epochs=reconstruction_epochs
        )
        
        # Test complete pipeline
        self.test_complete_pipeline()
        
        total_time = time.time() - start_time
        print(f"\n🎉 COMPLETE VECTOR-BASED PIPELINE FINISHED!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📁 Files created:")
        print(f"  - {self.vae_model_path}")
        print(f"  - {self.vectors_path}")
        print(f"  - {self.info_path}")
        print(f"  - {self.reconstruction_model_path}")
        print(f"  - complete_vector_pipeline_results.png")
        print(f"  - encoded_vectors/ directory with all vector files")

def quick_demo():
    """Quick demo with minimal training"""
    print("🚀 QUICK DEMO: Complete Vector-Based Pipeline")
    print("=" * 50)
    
    pipeline = CompleteVectorPipeline()
    
    # Run with minimal epochs for demo
    pipeline.run_complete_pipeline(
        vae_epochs=10,
        reconstruction_epochs=10,
        use_both_sequences=True,
        reconstruction_model_type='simple'
    )

def full_training():
    """Full training with more epochs"""
    print("🚀 FULL TRAINING: Complete Vector-Based Pipeline")
    print("=" * 50)
    
    pipeline = CompleteVectorPipeline()
    
    # Run with full epochs
    pipeline.run_complete_pipeline(
        vae_epochs=100,
        reconstruction_epochs=100,
        use_both_sequences=True,
        reconstruction_model_type='simple'
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        quick_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == 'full':
        full_training()
    else:
        print("Usage:")
        print("  python complete_vector_pipeline.py demo  # Quick demo (10 epochs each)")
        print("  python complete_vector_pipeline.py full  # Full training (100 epochs each)")
        print("\nOr run interactively:")
        print("  pipeline = CompleteVectorPipeline()")
        print("  pipeline.run_complete_pipeline()")
