# Two-Stage Pipeline: VAE + Reconstruction Model

This pipeline implements a two-stage approach for ARC data processing:

1. **Stage 1**: Train a VAE model on ARC data and generate outputs
2. **Stage 2**: Train a simple MLP model to reconstruct original data from VAE outputs

## 🏗️ Architecture Overview

```
Original ARC Data → VAE Model → VAE Outputs → Reconstruction Model → Final Reconstruction
     (1124)           ↓            ↓              ↓                    ↓
                  Latent Space   Generated     Simple MLP         Reconstructed
                  (64 dims)      Outputs       (512→256→128)      Original Data
```

## 📁 Files Created

### Core Models
- `VAEModel.py` - VAE architecture with attention and normalization
- `reconstruction_model.py` - Simple and advanced MLP models

### Training Scripts
- `train_vae_1d.py` - Train VAE model (updated with save/load functionality)
- `train_reconstruction_model.py` - Train reconstruction MLP
- `generate_vae_outputs.py` - Generate outputs from trained VAE

### Pipeline Script
- `two_stage_pipeline.py` - Complete end-to-end pipeline

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)
```bash
# Quick demo (10 epochs each)
python two_stage_pipeline.py demo

# Full training (100 epochs each)
python two_stage_pipeline.py full
```

### Option 2: Step-by-Step
```bash
# Step 1: Train VAE
python train_vae_1d.py

# Step 2: Generate VAE outputs
python generate_vae_outputs.py

# Step 3: Train reconstruction model
python train_reconstruction_model.py
```

### Option 3: Interactive
```python
from two_stage_pipeline import TwoStagePipeline

# Create pipeline
pipeline = TwoStagePipeline()

# Run complete pipeline
pipeline.run_complete_pipeline(
    vae_epochs=50,
    reconstruction_epochs=50,
    num_samples=1000,
    reconstruction_model_type='simple'
)
```

## 🔧 Configuration

### VAE Model Configuration
Edit `training_config.py` to adjust:
- `INPUT_LENGTH`: 1124 (ARC token sequence length)
- `LATENT_DIM`: 64 (latent space dimension)
- `HIDDEN_DIMS`: [256, 128] (encoder/decoder layers)
- `NUM_HEADS`: 8 (attention heads)
- `USE_INPUT_NORM`: True (input normalization)
- `USE_BATCH_NORM`: True (batch normalization)

### Reconstruction Model Configuration
Available in `train_reconstruction_model.py`:
- `model_type`: 'simple' or 'advanced'
- `hidden_dims`: [512, 256, 128] (MLP layers)
- `dropout_rate`: 0.2 (regularization)
- `batch_size`: 32
- `learning_rate`: 0.001

## 📊 Output Files

### Model Files
- `vae_1d_trained_model.pth` - Trained VAE model
- `reconstruction_model_simple.pth` - Simple reconstruction model
- `reconstruction_model_advanced.pth` - Advanced reconstruction model

### Data Files
- `vae_generated_outputs.json` - VAE-generated outputs and original targets

### Visualization Files
- `vae_1d_training_results.png` - VAE training plots
- `vae_generated_outputs_comparison.png` - VAE output comparison
- `reconstruction_model_simple_training_results.png` - Reconstruction training plots
- `reconstruction_model_simple_test_results.png` - Reconstruction test results
- `two_stage_pipeline_results.png` - Complete pipeline visualization

## 🧪 Model Types

### Simple MLP (`reconstruction_model.py`)
- Standard multi-layer perceptron
- Batch normalization and dropout
- Xavier weight initialization
- Faster training, simpler architecture

### Advanced MLP (`reconstruction_model.py`)
- Residual connections
- Self-attention mechanism
- Multiple output heads
- More sophisticated, potentially better performance

## 📈 Performance Metrics

The pipeline tracks several metrics:
- **MSE (Mean Squared Error)**: Primary reconstruction loss
- **MAE (Mean Absolute Error)**: Alternative loss metric
- **Training Time**: Time per epoch and total training time
- **Convergence**: Best loss and convergence epoch

## 🔄 Pipeline Flow

1. **Data Loading**: Load transformed ARC data (1124-length sequences)
2. **VAE Training**: Train VAE with attention and normalization
3. **VAE Generation**: Generate outputs from training data
4. **Reconstruction Training**: Train MLP on VAE outputs
5. **Pipeline Testing**: Test complete pipeline end-to-end
6. **Visualization**: Create comparison plots

## 🛠️ Customization

### Adding New Models
```python
# In reconstruction_model.py
class CustomReconstructionModel(nn.Module):
    def __init__(self, ...):
        # Your custom architecture
        pass

# In train_reconstruction_model.py
def create_reconstruction_model(model_type='custom', **kwargs):
    if model_type == 'custom':
        return CustomReconstructionModel(**kwargs)
```

### Modifying Pipeline
```python
# In two_stage_pipeline.py
pipeline = TwoStagePipeline()

# Custom stage 1
pipeline.stage1_train_vae(num_epochs=100, batch_size=64)

# Custom stage 2
pipeline.stage2_train_reconstruction(
    model_type='advanced',
    num_epochs=100
)
```

## 🐛 Troubleshooting

### Common Issues
1. **CUDA/MPS errors**: Set `device='cpu'` in pipeline initialization
2. **Memory issues**: Reduce `batch_size` or `num_samples`
3. **File not found**: Ensure data paths are correct
4. **Import errors**: Run from the `Junyao` directory

### Performance Tips
1. Use GPU if available (`device='auto'`)
2. Increase batch size for faster training
3. Use advanced MLP for potentially better results
4. Monitor loss curves for convergence

## 📝 Notes

- The VAE learns a compressed representation (64 dimensions) of ARC data
- The reconstruction model learns to map VAE outputs back to original data
- This creates a two-stage compression/decompression pipeline
- Useful for understanding VAE latent space and reconstruction quality
