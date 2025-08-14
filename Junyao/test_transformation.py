#!/usr/bin/env python3
"""
Test script to verify the ARC data transformation works correctly.
"""

import json
import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transform_arc_data import transform_arc_data
from ViTARC.vitarc.tokenizers.arc_tokenizer import get_or_build_arc_tokenizer

def test_preprocess_example():
    """Test preprocess_example function."""
    print("Testing preprocess_example...")
    
    # Test example
    test_example = {
        'input': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        'output': [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    }
    
    # Get tokenizer
    tokenizer = get_or_build_arc_tokenizer()
    
    # Process example
    processed = preprocess_example(test_example, tokenizer)
    
    print(f"Processed keys: {list(processed.keys())}")
    print(f"Input IDs length: {len(processed['input_ids'])}")
    print(f"Labels length: {len(processed['labels'])}")
    print(f"Input text: {processed['input_text']}")
    print(f"Output text: {processed['output_text']}")
    
    print("✓ Preprocess example test passed!")

def test_tokenizer_integration():
    """Test tokenizer integration with sample data."""
    print("\nTesting tokenizer integration...")
    
    # Get tokenizer
    tokenizer = get_or_build_arc_tokenizer()
    
    # Sample ARC data
    sample_data = {
        "test_task": {
            "train": [
                {
                    "input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    "output": [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
                }
            ],
            "test": [
                {
                    "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
                }
            ]
        }
    }
    
    # Transform the data
    transformed_data = transform_arc_data(sample_data, tokenizer)
    
    # Check the structure
    assert "test_task" in transformed_data
    assert "train" in transformed_data["test_task"]
    assert "test" in transformed_data["test_task"]
    
    # Check train example
    train_example = transformed_data["test_task"]["train"][0]
    assert "input" in train_example
    assert "output" in train_example
    assert "original" in train_example["input"]
    assert "text" in train_example["input"]
    assert "token_ids" in train_example["input"]
    
    # Check test example
    test_example = transformed_data["test_task"]["test"][0]
    assert "input" in test_example
    assert "original" in test_example["input"]
    assert "text" in test_example["input"]
    assert "token_ids" in test_example["input"]
    
    print("✓ Tokenizer integration test passed!")
    
    # Print some details
    print(f"  - Input type IDs length: {len(train_example['input_type_ids'])}")
    print(f"  - Output type IDs length: {len(train_example['output_type_ids'])}")
    print(f"  - Input type IDs: {train_example['input_type_ids'][:10]}...")  # Show first 10
    print(f"  - Output type IDs: {train_example['output_type_ids'][:10]}...")  # Show first 10

def test_small_file_processing():
    """Test processing a small file."""
    print("\nTesting small file processing...")
    
    # Create a small test file
    test_data = {
        "task1": {
            "train": [
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[1, 0], [0, 1]]
                }
            ],
            "test": [
                {
                    "input": [[1, 1], [0, 0]]
                }
            ]
        }
    }
    
    # Save test file
    test_file = "test_arc_data.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    try:
        # Get tokenizer
        tokenizer = get_or_build_arc_tokenizer()
        
        # Process the file
        from transform_arc_data import process_arc_file
        process_arc_file(test_file, tokenizer, ".")
        
        # Check if output file was created
        output_file = "test_arc_data_transformed.json"
        assert os.path.exists(output_file), "Output file was not created!"
        
        # Load and verify the transformed data
        with open(output_file, 'r') as f:
            transformed = json.load(f)
        
        assert "task1" in transformed
        assert len(transformed["task1"]["train"]) == 1
        assert len(transformed["task1"]["test"]) == 1
        
        print("✓ Small file processing test passed!")
        
    finally:
        # Clean up test files
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists(output_file):
            os.remove(output_file)

def main():
    """Run all tests."""
    print("ARC Data Transformation Tests")
    print("=" * 40)
    
    try:
        test_preprocess_example()
        test_tokenizer_integration()
        test_small_file_processing()
        
        print("\n" + "=" * 40)
        print("✓ All tests passed!")
        print("The transformation script is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 