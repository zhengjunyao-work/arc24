#!/usr/bin/env python3
"""
Encode Training Data into Latent Vectors
This script loads a trained VAE model and encodes all training data into latent vectors,
storing them with the same indices as the original data for easy retrieval.
"""

import torch
import torch.nn as nn
import json
import numpy as np
import os
from combined_arc_dataset import CombinedARCDataset, collate_fn
from train_vae_combined import load_model
from VAEModel import VAE1D

class LatentVectorEncoder:
    """Encode training data into latent vectors using trained VAE"""
    
    def __init__(self, model_path: str = "vae_combined_trained_model.pth", device: str = None):
        """
        Initialize the encoder
        
        Args:
            model_path: Path to trained VAE model
            device: Device to use ('auto', 'cpu', 'mps', 'cuda')
        """
        # Device setup
        if device:
            self.device = device

        elif device is None or device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        
        print(f"🔧 Using device: {self.device}")
        
        # Load trained model
        print(f"📥 Loading trained VAE model from {model_path}...")
        self.model, self.training_config = load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Create output directory
        self.output_dir = "encoded_vectors"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")
    
    def encode_dataset(self, data_path: str, use_both_sequences: bool = True, batch_size: int = 32):
        """
        Encode entire dataset into latent vectors
        
        Args:
            data_path: Path to transformed ARC data
            use_both_sequences: Whether to encode both input and output sequences
            batch_size: Batch size for encoding
        
        Returns:
            Dictionary with encoded vectors and metadata
        """
        print(f"📥 Loading dataset from {data_path}...")
        dataset = CombinedARCDataset(data_path, use_both_sequences=use_both_sequences)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        print(f"📊 Dataset info:")
        print(f"  - Total sequences: {len(dataset)}")
        print(f"  - Use both sequences: {use_both_sequences}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Number of batches: {len(dataloader)}")
        
        # Storage for encoded vectors
        encoded_vectors = []
        original_info = []
        
        print("🔄 Encoding sequences into latent vectors...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                batch_data = batch_data.to(self.device)
                
                # Get encoded vectors using the VAE model
                encoded_batch = self.model.get_encoded_vector(batch_data)
                
                # Store encoded vectors
                encoded_vectors.append(encoded_batch.cpu().numpy())
                
                # Store original information for this batch
                batch_start_idx = batch_idx * batch_size
                batch_end_idx = min(batch_start_idx + batch_size, len(dataset))
                
                for i in range(batch_end_idx - batch_start_idx):
                    original_info.append(dataset.get_original_info(batch_start_idx + i))
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Concatenate all encoded vectors
        encoded_vectors = np.concatenate(encoded_vectors, axis=0)
        
        print(f"✅ Encoding completed!")
        print(f"📊 Encoded vectors shape: {encoded_vectors.shape}")
        print(f"📊 Original info entries: {len(original_info)}")
        
        return encoded_vectors, original_info
    
    def save_encoded_vectors(self, encoded_vectors, original_info, 
                           filename_prefix: str = "encoded_vectors"):
        """
        Save encoded vectors with original indices
        
        Args:
            encoded_vectors: Numpy array of encoded vectors
            original_info: List of original information
            filename_prefix: Prefix for output files
        """
        print(f"💾 Saving encoded vectors...")
        
        # Save encoded vectors as numpy array
        vectors_path = os.path.join(self.output_dir, f"{filename_prefix}.npy")
        np.save(vectors_path, encoded_vectors)
        print(f"✅ Encoded vectors saved to {vectors_path}")
        
        # Save original information as JSON
        info_path = os.path.join(self.output_dir, f"{filename_prefix}_info.json")
        with open(info_path, 'w') as f:
            json.dump(original_info, f, indent=2)
        print(f"✅ Original info saved to {info_path}")
        
        # Save metadata
        metadata = {
            'encoded_vectors_shape': encoded_vectors.shape,
            'num_sequences': len(original_info),
            'latent_dim': encoded_vectors.shape[1],
            'model_info': {
                'input_length': 1124,
                'latent_dim': encoded_vectors.shape[1],
                'use_both_sequences': len(set(info['sequence_type'] for info in original_info)) > 1
            },
            'file_paths': {
                'vectors': vectors_path,
                'info': info_path
            }
        }
        
        metadata_path = os.path.join(self.output_dir, f"{filename_prefix}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved to {metadata_path}")
        
        return vectors_path, info_path, metadata_path
    
    def create_index_mapping(self, original_info):
        """
        Create mapping from original indices to encoded vector indices
        
        Args:
            original_info: List of original information
        
        Returns:
            Dictionary mapping original indices to encoded indices
        """
        print("🗂️  Creating index mapping...")
        
        index_mapping = {}
        
        for encoded_idx, info in enumerate(original_info):
            task_idx = info['task_idx']
            example_idx = info['example_idx']
            sequence_type = info['sequence_type']
            
            # Create key for original data
            key = f"task_{task_idx}_example_{example_idx}_{sequence_type}"
            index_mapping[key] = encoded_idx
        
        # Save index mapping
        mapping_path = os.path.join(self.output_dir, "index_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(index_mapping, f, indent=2)
        print(f"✅ Index mapping saved to {mapping_path}")
        
        return index_mapping
    
    def encode_and_save(self, data_path: str, use_both_sequences: bool = True, 
                       batch_size: int = 32, filename_prefix: str = "encoded_vectors"):
        """
        Complete pipeline: encode dataset and save all files
        
        Args:
            data_path: Path to transformed ARC data
            use_both_sequences: Whether to encode both input and output sequences
            batch_size: Batch size for encoding
            filename_prefix: Prefix for output files
        """
        print("🚀 Starting complete encoding pipeline...")
        
        # Encode dataset
        encoded_vectors, original_info = self.encode_dataset(
            data_path, use_both_sequences, batch_size
        )
        
        # Save encoded vectors
        vectors_path, info_path, metadata_path = self.save_encoded_vectors(
            encoded_vectors, original_info, filename_prefix
        )
        
        # Create index mapping
        index_mapping = self.create_index_mapping(original_info)
        
        print(f"\n🎉 Encoding pipeline completed!")
        print(f"📁 Files created in {self.output_dir}:")
        print(f"  - {os.path.basename(vectors_path)} (encoded vectors)")
        print(f"  - {os.path.basename(info_path)} (original info)")
        print(f"  - {os.path.basename(metadata_path)} (metadata)")
        print(f"  - index_mapping.json (index mapping)")
        
        return encoded_vectors, original_info, index_mapping

def load_encoded_vectors(vectors_path: str, info_path: str = None):
    """
    Load encoded vectors and original information
    
    Args:
        vectors_path: Path to encoded vectors .npy file
        info_path: Path to original info .json file (optional)
    
    Returns:
        Tuple of (encoded_vectors, original_info)
    """
    print(f"📥 Loading encoded vectors from {vectors_path}...")
    
    # Load encoded vectors
    encoded_vectors = np.load(vectors_path)
    print(f"📊 Loaded vectors shape: {encoded_vectors.shape}")
    
    # Load original info if provided
    original_info = None
    if info_path and os.path.exists(info_path):
        with open(info_path, 'r') as f:
            original_info = json.load(f)
        print(f"📊 Loaded {len(original_info)} original info entries")
    
    return encoded_vectors, original_info

def get_vector_by_original_index(encoded_vectors, original_info, 
                                task_idx: int, example_idx: int, sequence_type: str):
    """
    Get encoded vector by original task/example/sequence type
    
    Args:
        encoded_vectors: Numpy array of encoded vectors
        original_info: List of original information
        task_idx: Original task index
        example_idx: Original example index
        sequence_type: 'input' or 'output'
    
    Returns:
        Encoded vector or None if not found
    """
    for i, info in enumerate(original_info):
        if (info['task_idx'] == task_idx and 
            info['example_idx'] == example_idx and 
            info['sequence_type'] == sequence_type):
            return encoded_vectors[i]
    
    return None

if __name__ == "__main__":
    # Encode training data
    encoder = LatentVectorEncoder(
        model_path="vae_combined_trained_model.pth",
        device='auto'
    )
    
    # Encode and save
    encoded_vectors, original_info, index_mapping = encoder.encode_and_save(
        data_path="/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
        use_both_sequences=True,
        batch_size=32,
        filename_prefix="training_encoded_vectors"
    )
    
    print("\n🧪 Testing vector retrieval...")
    
    # Test retrieving vectors by original index
    test_task_idx = 0
    test_example_idx = 0
    
    # Get input vector
    input_vector = get_vector_by_original_index(
        encoded_vectors, original_info, test_task_idx, test_example_idx, 'input'
    )
    if input_vector is not None:
        print(f"✅ Retrieved input vector for task {test_task_idx}, example {test_example_idx}")
        print(f"   Vector shape: {input_vector.shape}")
    else:
        print(f"❌ Could not find input vector for task {test_task_idx}, example {test_example_idx}")
    
    # Get output vector
    output_vector = get_vector_by_original_index(
        encoded_vectors, original_info, test_task_idx, test_example_idx, 'output'
    )
    if output_vector is not None:
        print(f"✅ Retrieved output vector for task {test_task_idx}, example {test_example_idx}")
        print(f"   Vector shape: {output_vector.shape}")
    else:
        print(f"❌ Could not find output vector for task {test_task_idx}, example {test_example_idx}")
    
    print("\n🎉 Encoding and testing completed!")
    print("Next step: Use these encoded vectors to train the reconstruction model")
