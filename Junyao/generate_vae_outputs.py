#!/usr/bin/env python3
"""
Generate outputs from trained VAE model
This script loads a trained VAE model and generates outputs from training data
"""

import torch
import torch.nn as nn
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from train_vae_1d import load_model, TransformedARCDataset, collate_fn
from VAEModel import VAE1D

def generate_vae_outputs(model_path: str = "vae_1d_trained_model.pth", 
                        data_path: str = "../data/transformed_data/arc-agi_training_challenges_transformed.json",
                        output_path: str = "vae_generated_outputs.json",
                        num_samples: int = None,
                        device: str = None):
    """
    Generate outputs from trained VAE model
    
    Args:
        model_path: Path to trained VAE model
        data_path: Path to transformed ARC data
        output_path: Path to save generated outputs
        num_samples: Number of samples to generate (None for all)
        device: Device to use ('auto', 'cpu', 'mps', 'cuda')
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
    
    # Load trained model
    print(f"📥 Loading trained VAE model from {model_path}...")
    model, training_config = load_model(model_path)
    model.to(device)
    model.eval()
    
    # Load training data
    print(f"📥 Loading training data from {data_path}...")
    dataset = TransformedARCDataset(data_path)
    
    if num_samples is not None:
        # Limit dataset size
        dataset.data = dataset.data[:num_samples]
        print(f"📊 Limited to {num_samples} samples")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    print(f"📊 Dataset size: {len(dataset)} samples")
    
    # Generate outputs
    print("🔄 Generating VAE outputs...")
    generated_outputs = []
    original_inputs = []
    latent_codes = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            input_data = batch_data['input_type_ids'].to(device)
            
            # Get VAE outputs
            recon_data, mu, logvar = model(input_data)
            
            # Store results
            for i in range(input_data.size(0)):
                generated_outputs.append(recon_data[i].cpu().numpy().tolist())
                original_inputs.append(input_data[i].cpu().numpy().tolist())
                latent_codes.append(mu[i].cpu().numpy().tolist())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Save generated outputs
    output_data = {
        'original_inputs': original_inputs,
        'generated_outputs': generated_outputs,
        'latent_codes': latent_codes,
        'num_samples': len(generated_outputs),
        'model_config': {
            'input_length': training_config.get('input_length', 1124),
            'latent_dim': training_config.get('latent_dim', 64)
        },
        'generation_info': {
            'model_path': model_path,
            'data_path': data_path,
            'device_used': device
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Generated outputs saved to {output_path}")
    print(f"📊 Generated {len(generated_outputs)} samples")
    
    # Visualize some examples
    visualize_generated_outputs(original_inputs, generated_outputs, num_examples=5)
    
    return output_data

def visualize_generated_outputs(original_inputs, generated_outputs, num_examples=5):
    """Visualize original vs generated outputs"""
    print("📊 Creating visualization...")
    
    fig, axes = plt.subplots(num_examples, 2, figsize=(15, 3*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_examples, len(original_inputs))):
        # Original input
        axes[i, 0].plot(original_inputs[i])
        axes[i, 0].set_title(f'Original Input {i+1}')
        axes[i, 0].set_ylabel('Type ID Value')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Generated output
        axes[i, 1].plot(generated_outputs[i])
        axes[i, 1].set_title(f'VAE Generated Output {i+1}')
        axes[i, 1].set_ylabel('Type ID Value')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vae_generated_outputs_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved to vae_generated_outputs_comparison.png")

def analyze_generated_outputs(output_data):
    """Analyze the generated outputs"""
    print("\n📊 Analyzing generated outputs...")
    
    original_inputs = np.array(output_data['original_inputs'])
    generated_outputs = np.array(output_data['generated_outputs'])
    
    # Calculate statistics
    print(f"Original inputs shape: {original_inputs.shape}")
    print(f"Generated outputs shape: {generated_outputs.shape}")
    
    # Mean and std
    orig_mean = np.mean(original_inputs)
    orig_std = np.std(original_inputs)
    gen_mean = np.mean(generated_outputs)
    gen_std = np.std(generated_outputs)
    
    print(f"\nOriginal inputs - Mean: {orig_mean:.4f}, Std: {orig_std:.4f}")
    print(f"Generated outputs - Mean: {gen_mean:.4f}, Std: {gen_std:.4f}")
    
    # Reconstruction error
    mse = np.mean((original_inputs - generated_outputs) ** 2)
    mae = np.mean(np.abs(original_inputs - generated_outputs))
    
    print(f"\nReconstruction Error:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # Value range analysis
    orig_min, orig_max = np.min(original_inputs), np.max(original_inputs)
    gen_min, gen_max = np.min(generated_outputs), np.max(generated_outputs)
    
    print(f"\nValue Ranges:")
    print(f"  Original: [{orig_min:.2f}, {orig_max:.2f}]")
    print(f"  Generated: [{gen_min:.2f}, {gen_max:.2f}]")

if __name__ == "__main__":
    # Generate outputs from trained VAE
    output_data = generate_vae_outputs(
        model_path="vae_1d_trained_model.pth",
        data_path="../data/transformed_data/arc-agi_training_challenges_transformed.json",
        output_path="vae_generated_outputs.json",
        num_samples=1000  # Generate outputs for first 1000 samples
    )
    
    # Analyze the generated outputs
    analyze_generated_outputs(output_data)
    
    print("\n🎉 VAE output generation completed!")
    print("Next step: Use these outputs to train the second reconstruction model")
