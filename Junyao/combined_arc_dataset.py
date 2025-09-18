#!/usr/bin/env python3
"""
Combined ARC Dataset for VAE Training
This dataset combines both input and output sequences from ARC training data
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import numpy as np

class CombinedARCDataset(Dataset):
    """
    Dataset that combines both input and output sequences from ARC data
    This allows the VAE to learn from both input and output patterns
    """
    
    def __init__(self, data_path: str, use_both_sequences: bool = True):
        """
        Initialize the combined dataset
        
        Args:
            data_path: Path to transformed ARC data JSON file
            use_both_sequences: If True, use both input and output sequences
                               If False, use only output sequences (original behavior)
        """
        self.data_path = data_path
        self.use_both_sequences = use_both_sequences
        self.sample_pool = []
        self.original_indices = []  # Store original indices for easy retrieval
        
        self._load_data()
    
    def _load_data(self):
        """Load and process the ARC data"""
        print(f"Loading combined ARC data from: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} tasks")
        
        for task_idx, (task_id, task_data) in enumerate(data.items()):
            if 'train' in task_data:
                for example_idx, example in enumerate(task_data['train']):
                    if 'input_type_ids' in example and 'output_type_ids' in example:
                        input_data = torch.tensor(example['input_type_ids'], dtype=torch.float32)
                        output_data = torch.tensor(example['output_type_ids'], dtype=torch.float32)
                        
                        if self.use_both_sequences:
                            # Add both input and output sequences
                            self.sample_pool.append(input_data)
                            self.original_indices.append({
                                'task_id': task_id,
                                'task_idx': task_idx,
                                'example_idx': example_idx,
                                'sequence_type': 'input',
                                'original_data': example['input_type_ids']
                            })
                            
                            self.sample_pool.append(output_data)
                            self.original_indices.append({
                                'task_id': task_id,
                                'task_idx': task_idx,
                                'example_idx': example_idx,
                                'sequence_type': 'output',
                                'original_data': example['output_type_ids']
                            })
                        else:
                            # Only use output sequences (original behavior)
                            self.sample_pool.append(output_data)
                            self.original_indices.append({
                                'task_id': task_id,
                                'task_idx': task_idx,
                                'example_idx': example_idx,
                                'sequence_type': 'output',
                                'original_data': example['output_type_ids']
                            })
        
        print(f"Created sample pool with {len(self.sample_pool)} sequences")
        print(f"Use both sequences: {self.use_both_sequences}")
        
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
    
    def get_original_info(self, idx):
        """Get original information for a given index"""
        return self.original_indices[idx]
    
    def get_all_original_info(self):
        """Get all original information"""
        return self.original_indices

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

if __name__ == "__main__":
    # Test the dataset
    print("🧪 Testing CombinedARC Dataset...")
    
    dataset = CombinedARCDataset(
        data_path="/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json",
        use_both_sequences=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        info = dataset.get_original_info(i)
        print(f"Sample {i}: shape={sample.shape}, task={info['task_idx']}, "
              f"example={info['example_idx']}, type={info['sequence_type']}")
    
    print("✅ Dataset test completed!")
