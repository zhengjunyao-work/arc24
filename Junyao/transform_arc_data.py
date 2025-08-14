#!/usr/bin/env python3
"""
Transform ARC data using the ARC tokenizer and store as JSON.

This script:
1. Loads the ARC tokenizer
2. Reads raw ARC data from data/original_data/
3. Transforms the data using the tokenizer
4. Saves the transformed data as JSON files
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
    # Add ViTARC path for preprocess_example
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ViTARC"))
    from vitarc.datasets.gen_dataset import preprocess_example
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure arc_tokenizer.py is in the same directory and ViTARC is available.")
    sys.exit(1)



def transform_arc_data(data: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Transform ARC data using the preprocess_example function.
    
    Args:
        data: Original ARC data dictionary
        tokenizer: ARC tokenizer instance
        
    Returns:
        Transformed data dictionary
    """
    transformed_data = {}
    
    for task_id, task_data in data.items():
        transformed_task = {}
        
        # Transform train examples
        if 'train' in task_data:
            transformed_train = []
            for example in task_data['train']:
                # Use preprocess_example to transform the data
                processed_example = preprocess_example(example, tokenizer)
                
                # Store only input_type_ids and output_type_ids as lists with int conversion
                transformed_example = {
                    'original': example,
                    'input_type_ids': [int(x) for x in processed_example['input_type_ids']],
                    'output_type_ids': [int(x) for x in processed_example['output_type_ids']]
                }
                
                transformed_train.append(transformed_example)
            transformed_task['train'] = transformed_train
        
        # Transform test examples
        if 'test' in task_data:
            transformed_test = []
            for example in task_data['test']:
                # For test examples, we only have input, so we need to handle this differently
                # Create a dummy output for preprocessing (will be ignored)
                test_example = {
                    'input': example['input'],
                    'output': example['input']  # Use input as dummy output
                }
                
                processed_example = preprocess_example(test_example, tokenizer)
                
                # Store only input_type_ids and output_type_ids as lists with int conversion
                transformed_example = {
                    'original': example,
                    'input_type_ids': [int(x) for x in processed_example['input_type_ids']],
                    'output_type_ids': [int(x) for x in processed_example['output_type_ids']]
                }
                
                transformed_test.append(transformed_example)
            transformed_task['test'] = transformed_test
        
        transformed_data[task_id] = transformed_task
    
    return transformed_data

def process_arc_file(file_path: str, tokenizer, output_dir: str) -> None:
    """
    Process a single ARC data file.
    
    Args:
        file_path: Path to the input JSON file
        tokenizer: ARC tokenizer instance
        output_dir: Directory to save transformed data
    """
    print(f"Processing {file_path}...")
    
    # Read the original data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Transform the data
    transformed_data = transform_arc_data(data, tokenizer)
    
    # Create output filename
    input_filename = Path(file_path).stem
    output_filename = f"{input_filename}_transformed.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save transformed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved transformed data to {output_path}")
    
    # Print some statistics
    total_tasks = len(transformed_data)
    total_train_examples = sum(len(task.get('train', [])) for task in transformed_data.values())
    total_test_examples = sum(len(task.get('test', [])) for task in transformed_data.values())
    
    print(f"  - Total tasks: {total_tasks}")
    print(f"  - Total train examples: {total_train_examples}")
    print(f"  - Total test examples: {total_test_examples}")

def main():
    """Main function to transform ARC data."""
    print("ARC Data Transformation Script")
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
    data_dir = Path("./data/original_data")
    output_dir = Path("./data/transformed_data")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in the data directory
    json_files = list(data_dir.glob("*.json"))
    json_files = [json_files[-2]]
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for file_path in json_files:
        print(f"  - {file_path.name}")
    
    print("\nStarting transformation...")
    
    # Process each file
    for file_path in json_files:
        try:
            process_arc_file(str(file_path), tokenizer, str(output_dir))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("\nTransformation complete!")
    print(f"Transformed data saved to: {output_dir}")

if __name__ == "__main__":
    main() 