#!/usr/bin/env python3
"""
Check the length of transformed input_type_ids
"""

import json
import numpy as np

def check_transformed_length():
    """Check the length of transformed data"""
    file_path = '/Users/alexzheng/Library/Mobile Documents/com~apple~CloudDocs/github/arc-24/arc24/data/transformed_data/arc-agi_training_challenges_transformed.json'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} tasks from transformed data")
        
        # Collect all input_type_ids lengths
        input_lengths = []
        output_lengths = []
        
        for task_id, task_data in data.items():
            # Check train examples
            if 'train' in task_data:
                for example in task_data['train']:
                    if 'input_type_ids' in example:
                        input_lengths.append(len(example['input_type_ids']))
                    if 'output_type_ids' in example:
                        output_lengths.append(len(example['output_type_ids']))
            
            # Check test examples
            if 'test' in task_data:
                for example in task_data['test']:
                    if 'input_type_ids' in example:
                        input_lengths.append(len(example['input_type_ids']))
                    if 'output_type_ids' in example:
                        output_lengths.append(len(example['output_type_ids']))
        
        print(f"\nInput type IDs statistics:")
        print(f"  - Count: {len(input_lengths)}")
        print(f"  - Min length: {min(input_lengths)}")
        print(f"  - Max length: {max(input_lengths)}")
        print(f"  - Mean length: {np.mean(input_lengths):.2f}")
        print(f"  - Std length: {np.std(input_lengths):.2f}")
        print(f"  - Most common length: {max(set(input_lengths), key=input_lengths.count)}")
        
        print(f"\nOutput type IDs statistics:")
        print(f"  - Count: {len(output_lengths)}")
        print(f"  - Min length: {min(output_lengths)}")
        print(f"  - Max length: {max(output_lengths)}")
        print(f"  - Mean length: {np.mean(output_lengths):.2f}")
        print(f"  - Std length: {np.std(output_lengths):.2f}")
        print(f"  - Most common length: {max(set(output_lengths), key=output_lengths.count)}")
        
        # Check if all lengths are the same
        if len(set(input_lengths)) == 1:
            print(f"\n✅ All input_type_ids have the same length: {input_lengths[0]}")
        else:
            print(f"\n⚠️  Input type IDs have varying lengths")
            print(f"   Unique lengths: {sorted(set(input_lengths))}")
        
        if len(set(output_lengths)) == 1:
            print(f"✅ All output_type_ids have the same length: {output_lengths[0]}")
        else:
            print(f"⚠️  Output type IDs have varying lengths")
            print(f"   Unique lengths: {sorted(set(output_lengths))}")
        
        # Show a few examples
        print(f"\nSample input_type_ids (first 20 values):")
        for task_id in list(data.keys())[:3]:
            if 'train' in data[task_id] and data[task_id]['train']:
                example = data[task_id]['train'][0]
                if 'input_type_ids' in example:
                    print(f"  Task {task_id}: {example['input_type_ids'][:20]}...")
                    break
        
    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_transformed_length()
