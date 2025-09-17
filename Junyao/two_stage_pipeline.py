#!/usr/bin/env python3
"""
Two-Stage Pipeline: VAE + Reconstruction Model
This script demonstrates the complete pipeline:
1. Train VAE model on ARC data
2. Generate outputs from VAE
3. Train reconstruction model on VAE outputs
4. Test the complete pipeline
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from train_vae_1d import train_vae, save_model, load_model
from generate_vae_outputs import generate_vae_outputs
from train_reconstruction_model import train_reconstruction_model, load_reconstruction_model
from reconstruction_model import create_reconstruction_model

class TwoStagePipeline:
    """Complete two-stage pipeline: VAE + Reconstruction"""
    
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
        self.vae_outputs = None
        
        # File paths
        self.vae_model_path = "vae_1d_trained_model.pth"
        self.vae_outputs_path = "vae_generated_outputs.json"
        self.reconstruction_model_path = "reconstruction_model_simple.pth"
    
    def stage1_train_vae(self, 
                        data_path: str = "../data/transformed_data/arc-agi_training_challenges_transformed.json",
                        num_epochs: int = 50,
                        batch_size: int = 32,
                        learning_rate: float = 0.001):
        """
        Stage 1: Train VAE model
        
        Args:
            data_path: Path to transformed ARC data
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        print("🎯 STAGE 1: Training VAE Model")
        print("=" * 50)
        
        # Update training config if needed
        from training_config import *
        import training_config
        
        # Temporarily update config
        original_epochs = training_config.NUM_EPOCHS
        original_batch_size = training_config.BATCH_SIZE
        original_lr = training_config.LEARNING_RATE
        
        training_config.NUM_EPOCHS = num_epochs
        training_config.BATCH_SIZE = batch_size
        training_config.LEARNING_RATE = learning_rate
        
        try:
            # Train VAE
            self.vae_model, losses = train_vae()
            
            # Save model
            save_model(self.vae_model, self.vae_model_path)
            
            print(f"✅ Stage 1 completed: VAE trained and saved to {self.vae_model_path}")
            
        finally:
            # Restore original config
            training_config.NUM_EPOCHS = original_epochs
            training_config.BATCH_SIZE = original_batch_size
            training_config.LEARNING_RATE = original_lr
    
    def stage1_generate_outputs(self, 
                               data_path: str = "../data/transformed_data/arc-agi_training_challenges_transformed.json",
                               num_samples: int = 1000):
        """
        Stage 1: Generate outputs from trained VAE
        
        Args:
            data_path: Path to transformed ARC data
            num_samples: Number of samples to generate
        """
        print("🎯 STAGE 1: Generating VAE Outputs")
        print("=" * 50)
        
        # Generate outputs
        self.vae_outputs = generate_vae_outputs(
            model_path=self.vae_model_path,
            data_path=data_path,
            output_path=self.vae_outputs_path,
            num_samples=num_samples,
            device=self.device
        )
        
        print(f"✅ Stage 1 completed: VAE outputs saved to {self.vae_outputs_path}")
    
    def stage2_train_reconstruction(self,
                                   model_type: str = 'simple',
                                   batch_size: int = 32,
                                   learning_rate: float = 0.001,
                                   num_epochs: int = 50):
        """
        Stage 2: Train reconstruction model
        
        Args:
            model_type: Type of reconstruction model ('simple' or 'advanced')
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
        """
        print("🎯 STAGE 2: Training Reconstruction Model")
        print("=" * 50)
        
        # Train reconstruction model
        self.reconstruction_model, losses = train_reconstruction_model(
            model_type=model_type,
            data_path=self.vae_outputs_path,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=self.device,
            save_model=True
        )
        
        # Update model path
        self.reconstruction_model_path = f"reconstruction_model_{model_type}.pth"
        
        print(f"✅ Stage 2 completed: Reconstruction model trained and saved")
    
    def test_pipeline(self, num_test_samples: int = 5):
        """
        Test the complete pipeline
        
        Args:
            num_test_samples: Number of test samples to evaluate
        """
        print("🧪 TESTING COMPLETE PIPELINE")
        print("=" * 50)
        
        if self.vae_model is None:
            print("Loading VAE model...")
            self.vae_model, _ = load_model(self.vae_model_path)
            self.vae_model.to(self.device)
        
        if self.reconstruction_model is None:
            print("Loading reconstruction model...")
            self.reconstruction_model, _ = load_reconstruction_model(self.reconstruction_model_path)
            self.reconstruction_model.to(self.device)
        
        # Load test data
        from train_vae_1d import TransformedARCDataset, collate_fn
        from torch.utils.data import DataLoader
        
        dataset = TransformedARCDataset("../data/transformed_data/arc-agi_training_challenges_transformed.json")
        dataloader = DataLoader(dataset, batch_size=num_test_samples, shuffle=True, collate_fn=collate_fn)
        
        # Get test batch
        test_batch = next(iter(dataloader))
        original_data = test_batch['input_type_ids'].to(self.device)
        
        print(f"Testing with {num_test_samples} samples...")
        
        # Run through pipeline
        with torch.no_grad():
            # Stage 1: VAE encoding and decoding
            vae_output, mu, logvar = self.vae_model(original_data)
            
            # Stage 2: Reconstruction from VAE output
            final_reconstruction = self.reconstruction_model(vae_output)
            
            # Calculate metrics
            vae_mse = torch.mean((original_data - vae_output) ** 2).item()
            vae_mae = torch.mean(torch.abs(original_data - vae_output)).item()
            
            final_mse = torch.mean((original_data - final_reconstruction) ** 2).item()
            final_mae = torch.mean(torch.abs(original_data - final_reconstruction)).item()
            
            print(f"\n📊 Pipeline Performance:")
            print(f"VAE Stage:")
            print(f"  MSE: {vae_mse:.6f}")
            print(f"  MAE: {vae_mae:.6f}")
            print(f"Final Reconstruction:")
            print(f"  MSE: {final_mse:.6f}")
            print(f"  MAE: {final_mae:.6f}")
            
            # Visualize results
            self.visualize_pipeline_results(
                original_data, vae_output, final_reconstruction, num_test_samples
            )
    
    def visualize_pipeline_results(self, original_data, vae_output, final_reconstruction, num_samples):
        """Visualize pipeline results"""
        print("📊 Creating pipeline visualization...")
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(18, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original data
            axes[i, 0].plot(original_data[i].cpu().numpy())
            axes[i, 0].set_title(f'Original ARC Data {i+1}')
            axes[i, 0].set_ylabel('Type ID Value')
            axes[i, 0].grid(True, alpha=0.3)
            
            # VAE output
            axes[i, 1].plot(vae_output[i].cpu().numpy())
            axes[i, 1].set_title(f'VAE Output {i+1}')
            axes[i, 1].set_ylabel('Type ID Value')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Final reconstruction
            axes[i, 2].plot(final_reconstruction[i].cpu().numpy())
            axes[i, 2].set_title(f'Final Reconstruction {i+1}')
            axes[i, 2].set_ylabel('Type ID Value')
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('two_stage_pipeline_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Pipeline visualization saved to two_stage_pipeline_results.png")
    
    def run_complete_pipeline(self, 
                             vae_epochs: int = 50,
                             reconstruction_epochs: int = 50,
                             num_samples: int = 1000,
                             reconstruction_model_type: str = 'simple'):
        """
        Run the complete two-stage pipeline
        
        Args:
            vae_epochs: Number of epochs for VAE training
            reconstruction_epochs: Number of epochs for reconstruction training
            num_samples: Number of samples to generate
            reconstruction_model_type: Type of reconstruction model
        """
        print("🚀 RUNNING COMPLETE TWO-STAGE PIPELINE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Stage 1: Train VAE and generate outputs
        self.stage1_train_vae(num_epochs=vae_epochs)
        self.stage1_generate_outputs(num_samples=num_samples)
        
        # Stage 2: Train reconstruction model
        self.stage2_train_reconstruction(
            model_type=reconstruction_model_type,
            num_epochs=reconstruction_epochs
        )
        
        # Test pipeline
        self.test_pipeline()
        
        total_time = time.time() - start_time
        print(f"\n🎉 COMPLETE PIPELINE FINISHED!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📁 Files created:")
        print(f"  - {self.vae_model_path}")
        print(f"  - {self.vae_outputs_path}")
        print(f"  - {self.reconstruction_model_path}")
        print(f"  - two_stage_pipeline_results.png")

def quick_demo():
    """Quick demo with minimal training"""
    print("🚀 QUICK DEMO: Two-Stage Pipeline")
    print("=" * 50)
    
    pipeline = TwoStagePipeline()
    
    # Run with minimal epochs for demo
    pipeline.run_complete_pipeline(
        vae_epochs=10,
        reconstruction_epochs=10,
        num_samples=100,
        reconstruction_model_type='simple'
    )

def full_training():
    """Full training with more epochs"""
    print("🚀 FULL TRAINING: Two-Stage Pipeline")
    print("=" * 50)
    
    pipeline = TwoStagePipeline()
    
    # Run with full epochs
    pipeline.run_complete_pipeline(
        vae_epochs=100,
        reconstruction_epochs=100,
        num_samples=2000,
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
        print("  python two_stage_pipeline.py demo  # Quick demo (10 epochs each)")
        print("  python two_stage_pipeline.py full  # Full training (100 epochs each)")
        print("\nOr run interactively:")
        print("  pipeline = TwoStagePipeline()")
        print("  pipeline.run_complete_pipeline()")
