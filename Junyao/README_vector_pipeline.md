# Complete Vector-Based Pipeline: VAE + Encoded Vectors + Reconstruction

This pipeline implements a sophisticated two-stage approach using encoded vectors:

1. **Stage 1**: Train VAE on ALL ARC data (input + output sequences) and save the model
2. **Stage 2**: Encode all training data into latent vectors with original indices
3. **Stage 3**: Train reconstruction model using the encoded vectors
4. **Testing**: Complete pipeline evaluation and visualization

## 🏗️ Architecture Overview

```
Original ARC Data → VAE Model → Encoded Vectors → Reconstruction Model → Final Output
     (1124)           ↓            ↓ (64 dims)         ↓ (MLP)            ↓
                  Latent Space   Stored with         Simple/Advanced    Reconstructed
                  (64 dims)      Original Indices     MLP               Original Data
```

## 📁 New Files Created

### Core Components
- `combined_arc_dataset.py` - Dataset that combines input and output sequences
- `train_vae_combined.py` - Train VAE on combined ARC data
- `encode_training_data.py` - Encode all training data into vectors with indices
- `train_reconstruction_from_vectors.py` - Train reconstruction model from vectors
- `complete_vector_pipeline.py` - Complete end-to-end pipeline

### Key Features
- **Index Preservation**: Encoded vectors maintain original task/example indices
- **Easy Retrieval**: Get vectors by original task/example/sequence type
- **Efficient Training**: Pre-encoded vectors speed up reconstruction training
- **Complete Pipeline**: End-to-end training and testing

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)
```bash
# Quick demo (10 epochs each)
python complete_vector_pipeline.py demo

# Full training (100 epochs each)
python complete_vector_pipeline.py full
```

### Option 2: Step-by-Step
```bash
# Step 1: Train VAE on combined data
python train_vae_combined.py

# Step 2: Encode all training data into vectors
python encode_training_data.py

# Step 3: Train reconstruction model from vectors
python train_reconstruction_from_vectors.py
```

### Option 3: Interactive
```python
from complete_vector_pipeline import CompleteVectorPipeline

# Create pipeline
pipeline = CompleteVectorPipeline()

# Run complete pipeline
pipeline.run_complete_pipeline(
    vae_epochs=50,
    reconstruction_epochs=50,
    use_both_sequences=True,
    reconstruction_model_type='simple'
)
```

## 🔧 Key Components

### 1. Combined ARC Dataset (`combined_arc_dataset.py`)
```python
# Uses both input and output sequences
dataset = CombinedARCDataset(
    data_path="path/to/data.json",
    use_both_sequences=True  # Use both input and output
)

# Each sample has original index information
info = dataset.get_original_info(idx)
# Returns: {'task_idx': 0, 'example_idx': 0, 'sequence_type': 'input', ...}
```

### 2. VAE Training (`train_vae_combined.py`)
```python
# Train VAE on combined data
model, losses = train_vae_combined(
    data_path="path/to/data.json",
    use_both_sequences=True,
    batch_size=32,
    learning_rate=0.001,
    num_epochs=100
)
```

### 3. Vector Encoding (`encode_training_data.py`)
```python
# Encode all data into vectors
encoder = LatentVectorEncoder(model_path="vae_model.pth")
encoded_vectors, original_info, index_mapping = encoder.encode_and_save(
    data_path="path/to/data.json",
    use_both_sequences=True,
    batch_size=32
)

# Retrieve vector by original index
vector = get_vector_by_original_index(
    encoded_vectors, original_info, 
    task_idx=0, example_idx=0, sequence_type='input'
)
```

### 4. Reconstruction Training (`train_reconstruction_from_vectors.py`)
```python
# Train reconstruction model from vectors
model, losses = train_reconstruction_from_vectors(
    model_type='simple',
    vectors_path="encoded_vectors/training_encoded_vectors.npy",
    info_path="encoded_vectors/training_encoded_vectors_info.json",
    original_data_path="path/to/original_data.json",
    batch_size=32,
    learning_rate=0.001,
    num_epochs=50
)
```

## 📊 Output Files

### Model Files
- `vae_combined_trained_model.pth` - VAE trained on combined data
- `reconstruction_model_from_vectors_simple.pth` - Simple reconstruction model
- `reconstruction_model_from_vectors_advanced.pth` - Advanced reconstruction model

### Vector Files (in `encoded_vectors/` directory)
- `training_encoded_vectors.npy` - Encoded vectors (N x 64)
- `training_encoded_vectors_info.json` - Original information for each vector
- `training_encoded_vectors_metadata.json` - Metadata about the encoding
- `index_mapping.json` - Mapping from original indices to vector indices

### Visualization Files
- `vae_combined_training_results.png` - VAE training plots
- `reconstruction_from_vectors_simple_training_results.png` - Reconstruction training plots
- `reconstruction_from_vectors_simple_test_results.png` - Reconstruction test results
- `complete_vector_pipeline_results.png` - Complete pipeline visualization

## 🗂️ Vector Storage System

### Directory Structure
```
encoded_vectors/
├── training_encoded_vectors.npy          # Encoded vectors (N x 64)
├── training_encoded_vectors_info.json     # Original info for each vector
├── training_encoded_vectors_metadata.json # Encoding metadata
└── index_mapping.json                    # Index mapping
```

### Vector Retrieval
```python
# Load encoded vectors
encoded_vectors, original_info = load_encoded_vectors(
    "encoded_vectors/training_encoded_vectors.npy",
    "encoded_vectors/training_encoded_vectors_info.json"
)

# Get vector by original index
vector = get_vector_by_original_index(
    encoded_vectors, original_info,
    task_idx=0, example_idx=0, sequence_type='input'
)
```

### Index Mapping
```python
# Load index mapping
with open("encoded_vectors/index_mapping.json", 'r') as f:
    index_mapping = json.load(f)

# Key format: "task_{task_idx}_example_{example_idx}_{sequence_type}"
# Example: "task_0_example_0_input" -> vector_index
```

## 🧪 Model Types

### Simple MLP (`reconstruction_model.py`)
- Standard multi-layer perceptron
- Input: 64-dim encoded vector
- Output: 1124-dim reconstructed sequence
- Architecture: 64 → 512 → 256 → 128 → 1124

### Advanced MLP (`reconstruction_model.py`)
- Residual connections
- Self-attention mechanism
- More sophisticated architecture
- Potentially better performance

## 📈 Performance Metrics

The pipeline tracks several metrics:
- **VAE Training**: Total loss, reconstruction loss, KL divergence loss
- **Vector Encoding**: Encoding time, vector statistics
- **Reconstruction Training**: MSE loss, MAE loss
- **Pipeline Testing**: End-to-end reconstruction quality

## 🔄 Complete Pipeline Flow

1. **Data Loading**: Load ARC data with both input and output sequences
2. **VAE Training**: Train VAE on combined data (input + output)
3. **Model Saving**: Save trained VAE model
4. **Vector Encoding**: Encode all training data into latent vectors
5. **Index Preservation**: Store vectors with original indices
6. **Reconstruction Training**: Train MLP on encoded vectors
7. **Pipeline Testing**: Test complete pipeline end-to-end
8. **Visualization**: Create comprehensive comparison plots

## 🛠️ Customization

### Adding New Models
```python
# In reconstruction_model.py
class CustomReconstructionModel(nn.Module):
    def __init__(self, input_length=64, ...):
        # Your custom architecture
        pass

# In train_reconstruction_from_vectors.py
def create_reconstruction_model(model_type='custom', **kwargs):
    if model_type == 'custom':
        return CustomReconstructionModel(**kwargs)
```

### Modifying Pipeline
```python
# In complete_vector_pipeline.py
pipeline = CompleteVectorPipeline()

# Custom stage 1
pipeline.stage1_train_vae(num_epochs=100, use_both_sequences=True)

# Custom stage 2
pipeline.stage2_encode_data(batch_size=64)

# Custom stage 3
pipeline.stage3_train_reconstruction(
    model_type='advanced',
    num_epochs=100
)
```

## 🐛 Troubleshooting

### Common Issues
1. **CUDA/MPS errors**: Set `device='cpu'` in pipeline initialization
2. **Memory issues**: Reduce `batch_size` or use smaller datasets
3. **File not found**: Ensure data paths are correct
4. **Import errors**: Run from the `Junyao` directory

### Performance Tips
1. Use GPU if available (`device='auto'`)
2. Increase batch size for faster training
3. Use advanced MLP for potentially better results
4. Monitor loss curves for convergence
5. Pre-encoded vectors speed up reconstruction training

## 📝 Key Advantages

### Index Preservation
- Encoded vectors maintain original task/example indices
- Easy retrieval by original data location
- No loss of data organization

### Efficient Training
- Pre-encoded vectors eliminate VAE forward passes during reconstruction training
- Faster reconstruction model training
- Better memory efficiency

### Complete Pipeline
- End-to-end training and testing
- Comprehensive visualization
- Easy to modify and extend

## 🎯 Use Cases

1. **Research**: Study VAE latent space and reconstruction quality
2. **Data Compression**: Compress ARC data into 64-dim vectors
3. **Feature Learning**: Learn meaningful representations of ARC patterns
4. **Model Comparison**: Compare different reconstruction architectures
5. **Pipeline Development**: Build more complex multi-stage systems

## 📚 Next Steps

1. **Experiment with different VAE architectures**
2. **Try different reconstruction models**
3. **Analyze latent space properties**
4. **Build more complex pipelines**
5. **Apply to other sequence data**
