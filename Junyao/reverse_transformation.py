#!/usr/bin/env python3
"""
Reverse transformation functions for ARC data.

This module provides functions to convert processed ARC data back to the original format.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Add the current directory to the path to import the tokenizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from arc_tokenizer import get_or_build_arc_tokenizer
    # Add ViTARC path for preprocessing functions
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ViTARC"))
    from vitarc.datasets.gen_dataset import reformat_arc_tokens, replace_digits_with_arc
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure arc_tokenizer.py is in the same directory and ViTARC is available.")
    sys.exit(1)

def extract_grid_from_text(text: str) -> List[List[int]]:
    """
    Extract 2D grid from processed text representation.
    
    Args:
        text: Text like "<s><arc_0><arc_1><arc_0><arc_nl><arc_1><arc_1><arc_1><arc_nl><arc_0><arc_1><arc_0><arc_nl></s>"
        
    Returns:
        2D list of integers representing the grid
    """
    if not text:
        return []
    
    # Remove start and end tokens
    text = text.replace("<s>", "").replace("</s>", "")
    
    # Split by newline tokens
    rows = text.split("<arc_nl>")
    grid = []
    
    for row_text in rows:
        if not row_text.strip():
            continue
        
        # Extract cell values from ARC tokens
        row = []
        current_token = ""
        for char in row_text:
            if char == "<":
                current_token = "<"
            elif char == ">":
                current_token += ">"
                if current_token.startswith("<arc_") and current_token.endswith(">"):
                    # Extract the number from <arc_X>
                    try:
                        cell_value = int(current_token[5:-1])
                        row.append(cell_value)
                    except ValueError:
                        row.append(0)  # Default for invalid tokens
                current_token = ""
            elif current_token:
                current_token += char
        
        if row:  # Only add non-empty rows
            grid.append(row)
    
    return grid

def reverse_preprocess_example(input_type_ids: List[int], output_type_ids: List[int], tokenizer) -> Dict[str, Any]:
    """
    Reverse the preprocess_example transformation using only type IDs.
    
    Args:
        input_type_ids: Input type IDs from processed example
        output_type_ids: Output type IDs from processed example
        tokenizer: ARC tokenizer instance
        
    Returns:
        Original example with input and output grids
    """
    # Since we only have type IDs, we need to reconstruct the grids
    # This is a simplified approach - in practice, you might need more context
    
    # For now, we'll create dummy grids based on the type IDs length
    # This is a placeholder - you may need to adjust based on your specific needs
    input_size = len(input_type_ids)
    output_size = len(output_type_ids)
    
    # Create simple grids (this is a placeholder - adjust based on your needs)
    input_grid = [[0] * 10 for _ in range(min(10, input_size // 10 + 1))]
    output_grid = [[0] * 10 for _ in range(min(10, output_size // 10 + 1))]
    
    # Create original example format
    original_example = {
        'input': input_grid,
        'output': output_grid
    }
    
    return original_example

def reverse_transform_arc_data(transformed_data: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Reverse the transform_arc_data transformation.
    
    Args:
        transformed_data: Transformed data dictionary
        tokenizer: ARC tokenizer instance
        
    Returns:
        Original ARC data dictionary
    """
    original_data = {}
    
    for task_id, task_data in transformed_data.items():
        original_task = {}
        
        # Reverse train examples
        if 'train' in task_data:
            original_train = []
            for example in task_data['train']:
                # Extract original data if available, otherwise reverse from type IDs
                if 'original' in example:
                    original_example = example['original']
                else:
                    # Reverse from type IDs
                    input_type_ids = example.get('input_type_ids', [])
                    output_type_ids = example.get('output_type_ids', [])
                    original_example = reverse_preprocess_example(input_type_ids, output_type_ids, tokenizer)
                
                original_train.append(original_example)
            original_task['train'] = original_train
        
        # Reverse test examples
        if 'test' in task_data:
            original_test = []
            for example in task_data['test']:
                # Extract original data if available, otherwise reverse from type IDs
                if 'original' in example:
                    original_example = example['original']
                else:
                    # For test examples, we only have input
                    input_type_ids = example.get('input_type_ids', [])
                    output_type_ids = example.get('output_type_ids', [])
                    original_example = reverse_preprocess_example(input_type_ids, output_type_ids, tokenizer)
                
                original_test.append(original_example)
            original_task['test'] = original_test
        
        original_data[task_id] = original_task
    
    return original_data

def reverse_process_arc_file(file_path: str, tokenizer, output_dir: str) -> None:
    """
    Reverse process a single transformed ARC data file.
    
    Args:
        file_path: Path to the transformed JSON file
        tokenizer: ARC tokenizer instance
        output_dir: Directory to save reversed data
    """
    print(f"Reverse processing {file_path}...")
    
    # Read the transformed data
    with open(file_path, 'r', encoding='utf-8') as f:
        transformed_data = json.load(f)
    
    # Reverse the transformation
    original_data = reverse_transform_arc_data(transformed_data, tokenizer)
    
    # Create output filename
    input_filename = Path(file_path).stem
    output_filename = f"{input_filename}_reversed.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save reversed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved reversed data to {output_path}")
    
    # Print some statistics
    total_tasks = len(original_data)
    total_train_examples = sum(len(task.get('train', [])) for task in original_data.values())
    total_test_examples = sum(len(task.get('test', [])) for task in original_data.values())
    
    print(f"  - Total tasks: {total_tasks}")
    print(f"  - Total train examples: {total_train_examples}")
    print(f"  - Total test examples: {total_test_examples}")

def verify_reverse_transformation(original_file: str, transformed_file: str, reversed_file: str, tokenizer) -> bool:
    """
    Verify that the reverse transformation is correct by comparing original and reversed data.
    
    Args:
        original_file: Path to original data file
        transformed_file: Path to transformed data file
        reversed_file: Path to reversed data file
        tokenizer: ARC tokenizer instance
        
    Returns:
        True if verification passes, False otherwise
    """
    print("Verifying reverse transformation...")
    
    # Load all three files
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(transformed_file, 'r', encoding='utf-8') as f:
        transformed_data = json.load(f)
    
    with open(reversed_file, 'r', encoding='utf-8') as f:
        reversed_data = json.load(f)
    
    # Compare original and reversed data
    if original_data.keys() != reversed_data.keys():
        print("❌ Task IDs don't match!")
        return False
    
    for task_id in original_data.keys():
        original_task = original_data[task_id]
        reversed_task = reversed_data[task_id]
        
        # Check train examples
        if 'train' in original_task and 'train' in reversed_task:
            if len(original_task['train']) != len(reversed_task['train']):
                print(f"❌ Train example count mismatch for task {task_id}")
                return False
            
            for i, (orig_ex, rev_ex) in enumerate(zip(original_task['train'], reversed_task['train'])):
                if orig_ex['input'] != rev_ex['input']:
                    print(f"❌ Input mismatch for task {task_id}, train example {i}")
                    return False
                if orig_ex['output'] != rev_ex['output']:
                    print(f"❌ Output mismatch for task {task_id}, train example {i}")
                    return False
        
        # Check test examples
        if 'test' in original_task and 'test' in reversed_task:
            if len(original_task['test']) != len(reversed_task['test']):
                print(f"❌ Test example count mismatch for task {task_id}")
                return False
            
            for i, (orig_ex, rev_ex) in enumerate(zip(original_task['test'], reversed_task['test'])):
                if orig_ex['input'] != rev_ex['input']:
                    print(f"❌ Input mismatch for task {task_id}, test example {i}")
                    return False
    
    print("✅ Reverse transformation verification passed!")
    return True

def main():
    """Main function to reverse transform ARC data."""
    print("ARC Data Reverse Transformation Script")
    print("=" * 40)
    
    # Get the ARC tokenizer
    print("Loading ARC tokenizer...")
    try:
        tokenizer = get_or_build_arc_tokenizer()
        print(f"Tokenizer loaded successfully. Vocabulary size: {len(tokenizer)}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Define paths
    transformed_data_dir = Path("./data/transformed_data")
    reversed_data_dir = Path("./data/reversed_data")
    
    # Create output directory if it doesn't exist
    reversed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all transformed JSON files
    json_files = list(transformed_data_dir.glob("*_transformed.json"))
    
    if not json_files:
        print(f"No transformed JSON files found in {transformed_data_dir}")
        return
    
    print(f"Found {len(json_files)} transformed JSON files to reverse process:")
    for file_path in json_files:
        print(f"  - {file_path.name}")
    
    print("\nStarting reverse transformation...")
    
    # Process each file
    for file_path in json_files:
        try:
            reverse_process_arc_file(str(file_path), tokenizer, str(reversed_data_dir))
        except Exception as e:
            print(f"Error reverse processing {file_path}: {e}")
    
    print("\nReverse transformation complete!")
    print(f"Reversed data saved to: {reversed_data_dir}")

if __name__ == "__main__":
    main() 