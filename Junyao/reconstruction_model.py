#!/usr/bin/env python3
"""
Simple Multi-Layer Perceptron (MLP) for reconstruction task
This model takes VAE-generated outputs and reconstructs the original training data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionMLP(nn.Module):
    """
    Simple MLP for reconstructing original data from VAE outputs
    
    Architecture:
    Input: VAE generated output (length 1124)
    Output: Original training data (length 1124)
    """
    
    def __init__(self, input_length: int = 1124, hidden_dims: list = [512, 256, 128], 
                 dropout_rate: float = 0.2, activation: str = 'relu'):
        """
        Initialize the reconstruction MLP
        
        Args:
            input_length: Length of input sequence (VAE output)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'leaky_relu', 'gelu', 'swish')
        """
        super(ReconstructionMLP, self).__init__()
        
        self.input_length = input_length
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_length
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, input_length))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()  # Swish is SiLU in PyTorch
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
        
        Returns:
            Reconstructed tensor of shape (batch_size, input_length)
        """
        return self.network(x)
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_length': self.input_length,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'MLP'
        }

class AdvancedReconstructionMLP(nn.Module):
    """
    Advanced MLP with residual connections and attention
    
    This is a more sophisticated version with:
    - Residual connections
    - Self-attention mechanism
    - Multiple output heads
    """
    
    def __init__(self, input_length: int = 1124, hidden_dims: list = [512, 256, 128],
                 dropout_rate: float = 0.2, num_heads: int = 8, use_residual: bool = True):
        super(AdvancedReconstructionMLP, self).__init__()
        
        self.input_length = input_length
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(input_length, hidden_dims[0])
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.hidden_layers.append(layer)
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dims[-1], input_length)
        
        # Residual projection (if needed)
        if use_residual:
            self.residual_proj = nn.Linear(input_length, input_length)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass with residual connections and attention
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
        
        Returns:
            Reconstructed tensor of shape (batch_size, input_length)
        """
        # Store input for residual connection
        residual = x
        
        # Input projection
        x = self.input_proj(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Self-attention (reshape for attention)
        batch_size = x.size(0)
        x_reshaped = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x = attn_out.squeeze(1)  # (batch_size, hidden_dim)
        
        # Output projection
        x = self.output_proj(x)
        
        # Residual connection
        if self.use_residual:
            residual_proj = self.residual_proj(residual)
            x = x + residual_proj
        
        return x
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_length': self.input_length,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'Advanced MLP with Attention'
        }

def create_reconstruction_model(model_type: str = 'simple', **kwargs):
    """
    Factory function to create reconstruction models
    
    Args:
        model_type: Type of model ('simple' or 'advanced')
        **kwargs: Additional arguments for model creation
    
    Returns:
        Initialized model
    """
    if model_type == 'simple':
        return ReconstructionMLP(**kwargs)
    elif model_type == 'advanced':
        return AdvancedReconstructionMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test the models
    print("🧪 Testing Reconstruction Models...")
    
    # Test simple MLP
    print("\n1. Simple MLP:")
    simple_model = ReconstructionMLP(input_length=1124, hidden_dims=[512, 256, 128])
    print(f"Model info: {simple_model.get_model_info()}")
    
    # Test with sample input
    sample_input = torch.randn(4, 1124)
    sample_output = simple_model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {sample_output.shape}")
    
    # Test advanced MLP
    print("\n2. Advanced MLP:")
    advanced_model = AdvancedReconstructionMLP(input_length=1124, hidden_dims=[512, 256, 128])
    print(f"Model info: {advanced_model.get_model_info()}")
    
    # Test with sample input
    sample_output_adv = advanced_model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {sample_output_adv.shape}")
    
    print("\n✅ Model tests completed!")
