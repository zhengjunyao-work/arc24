# ARC Data Transformation Script

This script transforms ARC (Abstraction and Reasoning Corpus) data using the ARC tokenizer and stores the transformed data as JSON files.

## Overview

The transformation script:
1. Loads the ARC tokenizer using `get_or_build_arc_tokenizer()`
2. Reads raw ARC data from `data/original_data/`
3. Uses `preprocess_example()` from ViTARC to transform the data
4. Applies the tokenizer to transform the data
5. Saves the transformed data as JSON files in `data/transformed_data/`

## Files

- `transform_arc_data.py` - Main transformation script
- `reverse_transformation.py` - Reverse transformation script
- `test_transformation.py` - Test script to verify transformation functionality
- `test_reverse_transformation.py` - Test script to verify reverse transformation functionality
- `arc_tokenizer.py` - ARC tokenizer module (must be in the same directory)

## Usage

### 1. Run the transformation script

```bash
cd Junyao
python transform_arc_data.py
```

This will:
- Load the ARC tokenizer
- Process all JSON files in `../data/original_data/`
- Save transformed data to `../data/transformed_data/`

### 2. Test the transformation

```bash
python test_transformation.py
```

This runs tests to verify the transformation works correctly.

### 3. Reverse the transformation

```bash
python reverse_transformation.py
```

This reverses the transformation, converting processed data back to original format.

### 4. Test the reverse transformationpip install "pyarrow<0.15.0"

```bash
python test_reverse_transformation.py
```

This runs tests to verify the reverse transformation works correctly.

## Input Data Format

The script expects ARC data in the following format:

```json
{
  "task_id": {
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
```

## Output Data Format

The transformed data includes original data and only the essential type IDs:

```json
{
  "task_id": {
    "train": [
      {
        "original": {
          "input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
          "output": [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        },
        "input_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "output_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0]
      }
    ],
    "test": [
      {
        "original": {
          "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        },
        "input_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "output_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0]
      }
    ]
  }
}
```

## Key Functions

### `preprocess_example(example, tokenizer)`
Uses the ViTARC preprocessing pipeline:
- Converts 2D grids to ARC token format
- Adds `<s>` and `</s>` wrappers
- Tokenizes and pads/truncates to fixed length
- Generates type IDs for multi-task learning
- Returns HuggingFace-compatible format

### `transform_arc_data(data, tokenizer)`
Transforms ARC data using preprocess_example:
- Processes each task and example
- Stores both original and processed data
- Handles train and test examples separately

### `extract_grid_from_text(text)`
Extracts 2D grid from processed text representation:
- Removes `<s>` and `</s>` wrappers
- Splits by `<arc_nl>` tokens to get rows
- Extracts cell values from `<arc_X>` tokens
- Returns 2D integer grid

### `reverse_preprocess_example(input_type_ids, output_type_ids, tokenizer)`
Reverses the preprocess_example transformation using only type IDs:
- Takes input and output type IDs as parameters
- Reconstructs grids based on type ID patterns
- Returns original example format (simplified reconstruction)

### `reverse_transform_arc_data(transformed_data, tokenizer)`
Reverses the transform_arc_data transformation:
- Processes each task and example
- Uses original data if available, otherwise reverses from processed
- Handles train and test examples separately
- Returns original ARC data format

### `verify_reverse_transformation(original_file, transformed_file, reversed_file, tokenizer)`
Verifies that reverse transformation is correct:
- Compares original and reversed data
- Checks task IDs, example counts, and grid values
- Returns True if verification passes, False otherwise

### `process_arc_file(file_path, tokenizer, output_dir)`
Processes a single ARC data file:
- Reads the JSON file
- Transforms the data
- Saves the transformed data to a new JSON file
- Prints statistics about the transformation

## Dependencies

- `arc_tokenizer.py` - Must be in the same directory
- `json` - For reading/writing JSON files
- `pathlib` - For file path operations
- `numpy` - For numerical operations (optional)

## Error Handling

The script includes error handling for:
- Missing tokenizer module
- File reading/writing errors
- Invalid data formats
- Tokenizer errors

## Output Files

Transformed files are saved with the naming convention:
- `original_file.json` → `original_file_transformed.json`

## Statistics

The script prints statistics for each processed file:
- Total number of tasks
- Total number of train examples
- Total number of test examples

## Example Output

```
ARC Data Transformation Script
========================================
Loading ARC tokenizer...
Tokenizer loaded successfully. Vocabulary size: 50
Found 6 JSON files to process:
  - arc-agi_training_challenges.json
  - arc-agi_training_solutions.json
  - arc-agi_test_challenges.json
  - arc-agi_evaluation_challenges.json
  - arc-agi_evaluation_solutions.json
  - sample_submission.json

Starting transformation...
Processing ../data/original_data/arc-agi_training_challenges.json...
Saved transformed data to ../data/transformed_data/arc-agi_training_challenges_transformed.json
  - Total tasks: 400
  - Total train examples: 1200
  - Total test examples: 400

Transformation complete!
Transformed data saved to: ../data/transformed_data
``` 