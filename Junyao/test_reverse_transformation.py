#!/usr/bin/env python3
"""
Test script to verify the reverse transformation functions work correctly.
"""

import json
import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reverse_transformation import (
    extract_grid_from_text, 
    reverse_preprocess_example, 
    reverse_transform_arc_data,
    verify_reverse_transformation
)
from transform_arc_data import transform_arc_data
from arc_tokenizer import get_or_build_arc_tokenizer

def test_extract_grid_from_text():
    """Test extracting grid from text representation."""
    print("Testing extract_grid_from_text...")
    
    # Test text
    test_text = "<s><arc_0><arc_1><arc_0><arc_nl><arc_1><arc_1><arc_1><arc_nl><arc_0><arc_1><arc_0><arc_nl></s>"
    
    # Expected grid
    expected_grid = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    
    # Extract grid
    extracted_grid = extract_grid_from_text(test_text)
    
    print(f"Input text: {test_text}")
    print(f"Extracted grid: {extracted_grid}")
    print(f"Expected grid: {expected_grid}")
    
    # Check if they match
    assert extracted_grid == expected_grid, "Grid extraction failed!"
    print("✓ Grid extraction test passed!")

def test_reverse_preprocess_example():
    """Test reverse_preprocess_example function."""
    print("Testing reverse_preprocess_example...")
    
    # Get tokenizer
    tokenizer = get_or_build_arc_tokenizer()
    
    # Test type IDs
    input_type_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    output_type_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Reverse process
    reversed_example = reverse_preprocess_example(input_type_ids, output_type_ids, tokenizer)
    
    print(f"Input type IDs: {input_type_ids}")
    print(f"Output type IDs: {output_type_ids}")
    print(f"Reversed example: {reversed_example}")
    
    # Check if we get a valid structure
    assert 'input' in reversed_example, "Reversed example missing input!"
    assert 'output' in reversed_example, "Reversed example missing output!"
    assert isinstance(reversed_example['input'], list), "Input should be a list!"
    assert isinstance(reversed_example['output'], list), "Output should be a list!"
    
    print("✓ Reverse preprocessing test passed!")

def test_full_transformation_cycle():
    """Test the full transformation cycle: original -> transformed -> reversed."""
    print("Testing full transformation cycle...")
    
    # Get tokenizer
    tokenizer = get_or_build_arc_tokenizer()
    
    # Test data
    test_data = {
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
    
    # Transform
    transformed_data = transform_arc_data(test_data, tokenizer)
    
    # Reverse transform
    reversed_data = reverse_transform_arc_data(transformed_data, tokenizer)
    
    print(f"Original data: {test_data}")
    print(f"Transformed data keys: {list(transformed_data.keys())}")
    print(f"Reversed data: {reversed_data}")
    
    # Check if original and reversed match
    assert test_data == reversed_data, "Full transformation cycle failed!"
    print("✓ Full transformation cycle test passed!")

def test_verify_reverse_transformation():
    """Test the verification function."""
    print("Testing verify_reverse_transformation...")
    
    # Create test files
    test_dir = Path("./test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Test data
    test_data = {
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
    
    # Get tokenizer
    tokenizer = get_or_build_arc_tokenizer()
    
    # Transform
    transformed_data = transform_arc_data(test_data, tokenizer)
    
    # Reverse transform
    reversed_data = reverse_transform_arc_data(transformed_data, tokenizer)
    
    # Save files
    original_file = test_dir / "original.json"
    transformed_file = test_dir / "transformed.json"
    reversed_file = test_dir / "reversed.json"
    
    with open(original_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    with open(transformed_file, 'w') as f:
        json.dump(transformed_data, f, indent=2)
    
    with open(reversed_file, 'w') as f:
        json.dump(reversed_data, f, indent=2)
    
    # Verify
    result = verify_reverse_transformation(
        str(original_file), 
        str(transformed_file), 
        str(reversed_file), 
        tokenizer
    )
    
    # Clean up
    for file in [original_file, transformed_file, reversed_file]:
        if file.exists():
            file.unlink()
    test_dir.rmdir()
    
    assert result, "Verification failed!"
    print("✓ Verification test passed!")

def main():
    """Run all tests."""
    print("Reverse Transformation Tests")
    print("=" * 30)
    
    try:
        test_extract_grid_from_text()
        test_reverse_preprocess_example()
        test_full_transformation_cycle()
        test_verify_reverse_transformation()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 