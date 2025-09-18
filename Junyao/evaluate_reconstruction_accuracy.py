#!/usr/bin/env python3
"""
Evaluate Reconstruction Accuracy
This script calculates exact match scores between reconstructed images and original outputs
"""

import torch
import torch.nn as nn
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from train_vae_combined import load_model
from train_reconstruction_from_vectors import load_reconstruction_model_from_vectors
from encode_training_data import load_encoded_vectors, get_vector_by_original_index
from reconstruction_model import create_reconstruction_model

def sequence_to_grid(sequence, grid_size=(30, 30)):
    """
    Convert a sequence back to a 2D grid
    This is a simplified conversion - you may need to adjust based on your tokenization method
    
    Args:
        sequence: 1D sequence of type IDs
        grid_size: Target grid size (height, width)
    
    Returns:
        2D numpy array representing the grid
    """
    # Reshape sequence to grid
    if len(sequence) >= grid_size[0] * grid_size[1]:
        # Take first grid_size[0] * grid_size[1] elements
        grid_flat = sequence[:grid_size[0] * grid_size[1]]
    else:
        # Pad with zeros if sequence is too short
        grid_flat = np.zeros(grid_size[0] * grid_size[1])
        grid_flat[:len(sequence)] = sequence
    
    # Reshape to 2D grid
    grid = grid_flat.reshape(grid_size)
    
    # Convert to integer type IDs (assuming they represent colors)
    grid = grid.astype(int)
    
    return grid

def calculate_exact_match_score(original_grid, reconstructed_grid):
    """
    Calculate exact match score between two grids
    
    Args:
        original_grid: Original 2D grid
        reconstructed_grid: Reconstructed 2D grid
    
    Returns:
        Exact match score (0.0 to 1.0)
    """
    # Ensure both grids have the same shape
    if original_grid.shape != reconstructed_grid.shape:
        # Resize to match the smaller grid
        min_height = min(original_grid.shape[0], reconstructed_grid.shape[0])
        min_width = min(original_grid.shape[1], reconstructed_grid.shape[1])
        
        original_grid = original_grid[:min_height, :min_width]
        reconstructed_grid = reconstructed_grid[:min_height, :min_width]
    
    # Calculate exact matches
    matches = (original_grid == reconstructed_grid).sum()
    total_pixels = original_grid.size
    
    exact_match_score = matches / total_pixels
    
    return exact_match_score

def evaluate_reconstruction_accuracy(vae_model_path: str = "vae_combined_trained_model.pth",
                                   reconstruction_model_path: str = "reconstruction_model_from_vectors_simple.pth",
                                   vectors_path: str = "encoded_vectors/training_encoded_vectors.npy",
                                   info_path: str = "encoded_vectors/training_encoded_vectors_info.json",
                                   original_data_path: str = "/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
                                   num_samples: int = 100,
                                   device: str = None):
    """
    Evaluate reconstruction accuracy by comparing reconstructed images with original outputs
    
    Args:
        vae_model_path: Path to trained VAE model
        reconstruction_model_path: Path to traine....
        .3
        
        
        
        
        3d reconstruction model
        vectors_path: Path to encoded vectors
        info_path: Path to original info
        original_data_path: Path to original transformed data
        num_samples: Number of samples to evaluate
        device: Device to use
    
    Returns:
        Dictionary with evaluation results
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
    
    # Load models
    print("📥 Loading models...")
    vae_model, _ = load_model(vae_model_path)
    vae_model.to(device)
    vae_model.eval()
    
    reconstruction_model, _ = load_reconstruction_model_from_vectors(reconstruction_model_path)
    reconstruction_model.to(device)
    reconstruction_model.eval()
    
    # Load encoded vectors and original info
    print("📥 Loading encoded vectors...")
    encoded_vectors, original_info = load_encoded_vectors(vectors_path, info_path)
    
    # Load original data
    print("📥 Loading original data...")
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Evaluation results
    exact_match_scores = []
    sample_results = []
    
    print(f"🔄 Evaluating {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(encoded_vectors))):
            # Get encoded vector
            encoded_vector = torch.FloatTensor(encoded_vectors[i]).unsqueeze(0).to(device)
            
            # Get original info
            info = original_info[i]
            task_id = info['task_id']
            example_idx = info['example_idx']
            sequence_type = info['sequence_type']
            
            # Get original sequence from data
            if (task_id in original_data and 
                'train' in original_data[task_id] and
                example_idx < len(original_data[task_id]['train'])):
                
                example = original_data[task_id]['train'][example_idx]
                
                if sequence_type == 'input' and 'input_type_ids' in example:
                    original_sequence = example['input_type_ids']
                elif sequence_type == 'output' and 'output_type_ids' in example:
                    original_sequence = example['output_type_ids']
                else:
                    continue
                
                # Reconstruct using reconstruction model
                reconstructed_sequence = reconstruction_model(encoded_vector).cpu().numpy()[0]
                
                # Convert sequences to grids
                original_grid = sequence_to_grid(original_sequence)
                reconstructed_grid = sequence_to_grid(reconstructed_sequence)
                
                # Calculate exact match score
                exact_match_score = calculate_exact_match_score(original_grid, reconstructed_grid)
                exact_match_scores.append(exact_match_score)
                
                # Store sample result
                sample_results.append({
                    'sample_idx': i,
                    'task_id': task_id,
                    'example_idx': example_idx,
                    'sequence_type': sequence_type,
                    'exact_match_score': exact_match_score,
                    'original_grid_shape': original_grid.shape,
                    'reconstructed_grid_shape': reconstructed_grid.shape
                })
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{num_samples} samples")
    
    # Calculate statistics
    if exact_match_scores:
        mean_score = np.mean(exact_match_scores)
        std_score = np.std(exact_match_scores)
        min_score = np.min(exact_match_scores)
        max_score = np.max(exact_match_scores)
        
        # Count perfect matches (score = 1.0)
        perfect_matches = sum(1 for score in exact_match_scores if score == 1.0)
        perfect_match_rate = perfect_matches / len(exact_match_scores)
        
        print(f"\n📊 Reconstruction Accuracy Results:")
        print(f"  Mean Exact Match Score: {mean_score:.4f}")
        print(f"  Std Exact Match Score: {std_score:.4f}")
        print(f"  Min Exact Match Score: {min_score:.4f}")
        print(f"  Max Exact Match Score: {max_score:.4f}")
        print(f"  Perfect Matches: {perfect_matches}/{len(exact_match_scores)} ({perfect_match_rate:.2%})")
        
        # Create visualization
        visualize_accuracy_results(exact_match_scores, sample_results[:10])  # Show first 10 samples
        
        # Save results
        results = {
            'evaluation_summary': {
                'num_samples_evaluated': len(exact_match_scores),
                'mean_exact_match_score': mean_score,
                'std_exact_match_score': std_score,
                'min_exact_match_score': min_score,
                'max_exact_match_score': max_score,
                'perfect_matches': perfect_matches,
                'perfect_match_rate': perfect_match_rate
            },
            'sample_results': sample_results,
            'all_scores': exact_match_scores
        }
        
        # Save to file
        results_path = "reconstruction_accuracy_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {results_path}")
        
        return results
    else:
        print("❌ No samples could be evaluated")
        return None

def visualize_accuracy_results(exact_match_scores, sample_results):
    """Visualize accuracy results"""
    print("📊 Creating accuracy visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Score distribution histogram
    axes[0, 0].hist(exact_match_scores, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Exact Match Scores')
    axes[0, 0].set_xlabel('Exact Match Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Score over samples
    axes[0, 1].plot(exact_match_scores)
    axes[0, 1].set_title('Exact Match Score by Sample')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Exact Match Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Score statistics
    mean_score = np.mean(exact_match_scores)
    std_score = np.std(exact_match_scores)
    
    axes[1, 0].bar(['Mean', 'Std', 'Min', 'Max'], 
                   [mean_score, std_score, np.min(exact_match_scores), np.max(exact_match_scores)])
    axes[1, 0].set_title('Score Statistics')
    axes[1, 0].set_ylabel('Score Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Perfect match rate
    perfect_matches = sum(1 for score in exact_match_scores if score == 1.0)
    perfect_match_rate = perfect_matches / len(exact_match_scores)
    
    axes[1, 1].pie([perfect_matches, len(exact_match_scores) - perfect_matches], 
                   labels=['Perfect Matches', 'Non-Perfect Matches'],
                   autopct='%1.1f%%')
    axes[1, 1].set_title(f'Perfect Match Rate: {perfect_match_rate:.2%}')
    
    plt.tight_layout()
    plt.savefig('reconstruction_accuracy_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Accuracy visualization saved to reconstruction_accuracy_visualization.png")

def compare_reconstruction_samples(vae_model_path: str,
                                 reconstruction_model_path: str,
                                 vectors_path: str,
                                 info_path: str,
                                 original_data_path: str,
                                 num_samples: int = 5,
                                 device: str = None):
    """
    Compare original and reconstructed samples side by side
    
    Args:
        Same as evaluate_reconstruction_accuracy
        num_samples: Number of samples to compare visually
    """
    
    # Device setup
    if device is None or device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    # Load models and data (same as above)
    vae_model, _ = load_model(vae_model_path)
    vae_model.to(device)
    vae_model.eval()
    
    reconstruction_model, _ = load_reconstruction_model_from_vectors(reconstruction_model_path)
    reconstruction_model.to(device)
    reconstruction_model.eval()
    
    encoded_vectors, original_info = load_encoded_vectors(vectors_path, info_path)
    
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"🔄 Comparing {num_samples} samples visually...")
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(min(num_samples, len(encoded_vectors))):
            # Get data
            encoded_vector = torch.FloatTensor(encoded_vectors[i]).unsqueeze(0).to(device)
            info = original_info[i]
            task_id = info['task_id']
            example_idx = info['example_idx']
            sequence_type = info['sequence_type']
            
            # Get original sequence
            if (task_id in original_data and 
                'train' in original_data[task_id] and
                example_idx < len(original_data[task_id]['train'])):
                
                example = original_data[task_id]['train'][example_idx]
                
                if sequence_type == 'input' and 'input_type_ids' in example:
                    original_sequence = example['input_type_ids']
                elif sequence_type == 'output' and 'output_type_ids' in example:
                    original_sequence = example['output_type_ids']
                else:
                    continue
                
                # Reconstruct
                reconstructed_sequence = reconstruction_model(encoded_vector).cpu().numpy()[0]
                
                # Convert to grids
                original_grid = sequence_to_grid(original_sequence)
                reconstructed_grid = sequence_to_grid(reconstructed_sequence)
                
                # Calculate score
                exact_match_score = calculate_exact_match_score(original_grid, reconstructed_grid)
                
                # Plot
                axes[i, 0].imshow(original_grid, cmap='viridis')
                axes[i, 0].set_title(f'Original {sequence_type.title()} {i+1}')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(reconstructed_grid, cmap='viridis')
                axes[i, 1].set_title(f'Reconstructed {i+1}')
                axes[i, 1].axis('off')
                
                # Difference
                diff_grid = np.abs(original_grid - reconstructed_grid)
                axes[i, 2].imshow(diff_grid, cmap='Reds')
                axes[i, 2].set_title(f'Difference (Score: {exact_match_score:.3f})')
                axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_sample_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Sample comparison saved to reconstruction_sample_comparison.png")

if __name__ == "__main__":
    # Evaluate reconstruction accuracy
    results = evaluate_reconstruction_accuracy(
        vae_model_path="vae_combined_trained_model.pth",
        reconstruction_model_path="reconstruction_model_from_vectors_simple.pth",
        vectors_path="encoded_vectors/training_encoded_vectors.npy",
        info_path="encoded_vectors/training_encoded_vectors_info.json",
        original_data_path="/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
        num_samples=100
    )
    
    if results:
        print("\n🎉 Reconstruction accuracy evaluation completed!")
        print("Next step: Analyze results and improve model if needed")
