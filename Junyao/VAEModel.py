import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, img_channels=1, latent_channels=128):
        super(VAE, self).__init__()
        self.img_channels = img_channels
        self.latent_channels = latent_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1124, 32, 2, stride = 2, padding = 1),  # 32x32x32 # here does not require an input of fixed size. stride and padding will operation on input and create feature maps.
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, 2, 1),           # 64x16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 2, 2, 1),          # 128x8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 2, 2, 1),         # 256x4x4
            nn.ReLU(),
            
        )
        self.conv_mu = nn.Conv2d(256, latent_channels, 1, 1)
        self.conv_logvar = nn.Conv2d(256, latent_channels, 1, 1)
        self.decode_conv = nn.Conv2d( latent_channels,256, 1, 1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # img_channelsx64x64
            nn.Sigmoid(),
        )

    def encode(self, x):
        conv_h = self.encoder(x)
        # h = h.view(h.size(0), -1)
        mu = self.conv_mu(conv_h)
        logvar = self.conv_logvar(conv_h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decode_conv(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class VAE1D(nn.Module):
    """
    VAE model for 1D sequences with input/output length of 1124 and attention mechanisms
    """
    def __init__(self, input_length=1124, latent_dim=64, hidden_dims=[512, 256, 128], num_heads=8, 
                 use_input_norm=True, use_batch_norm=True):
        super(VAE1D, self).__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.use_input_norm = use_input_norm
        self.use_batch_norm = use_batch_norm
        
        # Input normalization
        if use_input_norm:
            self.input_norm = nn.LayerNorm(input_length)
        
        # Encoder layers with attention
        self.encoder_layers = nn.ModuleList()
        in_channels = 1  # Single channel for 1D data
        
        for hidden_dim in hidden_dims:
            # Conv layer with enhanced normalizationrtrrtr
            conv_components = [nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1)]
            
            if use_batch_norm:
                conv_components.append(nn.BatchNorm1d(hidden_dim))
            
            conv_components.extend([
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            
            conv_layer = nn.Sequential(*conv_components)
            
            # Attention layer
            attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads, 
                batch_first=True,
                dropout=0.1
            )
            
            self.encoder_layers.append(nn.ModuleDict({
                'conv': conv_layer,
                'attention': attention_layer
            }))
            
            in_channels = hidden_dim
        
        # Calculate the size after encoding
        encoded_length = input_length
        for _ in hidden_dims:
            encoded_length = (encoded_length + 2 - 1) // 2  # ceil division
        
        self.encoded_length = encoded_length
        self.encoded_features = hidden_dims[-1]
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.encoded_features * self.encoded_length, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_features * self.encoded_length, latent_dim)
        
        # Decoder input projection
        self.fc_decoder = nn.Linear(latent_dim, self.encoded_features * self.encoded_length)
        
        # Decoder layers with attention
        self.decoder_layers = nn.ModuleList()
        hidden_dims_reversed = list(reversed(hidden_dims))
        
        for i, hidden_dim in enumerate(hidden_dims_reversed[:-1]):
            next_dim = hidden_dims_reversed[i + 1] if i + 1 < len(hidden_dims_reversed) else 1
            
            # ConvTranspose layer with enhanced normalization
            conv_transpose_components = [nn.ConvTranspose1d(hidden_dim, next_dim, kernel_size=3, stride=2, padding=1, output_padding=1)]
            
            if use_batch_norm:
                conv_transpose_components.append(nn.BatchNorm1d(next_dim))
            
            conv_transpose_components.extend([
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            
            conv_transpose_layer = nn.Sequential(*conv_transpose_components)
            
            # Attention layer
            attention_layer = nn.MultiheadAttention(
                embed_dim=next_dim, 
                num_heads=num_heads, 
                batch_first=True,
                dropout=0.1
            )
            
            self.decoder_layers.append(nn.ModuleDict({
                'conv_transpose': conv_transpose_layer,
                'attention': attention_layer
            }))
        
        # Final layer to get exact output size
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims_reversed[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def encode(self, x):
        """
        Encode input sequence to latent space with attention
        Args:
            x: Input tensor of shape (batch_size, 1, input_length)
        Returns:
            mu, logvar: Mean and log variance of latent space
        """
        # Ensure input is 3D: (batch_size, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Apply input normalization if enabled
        if self.use_input_norm:
            # Apply layer norm across the sequence dimension
            x = self.input_norm(x)
        
        h = x
        # Apply encoder layers with attention
        for layer in self.encoder_layers:
            # Conv layer
            h = layer['conv'](h)
            
            # Prepare for attention: (batch, channels, length) -> (batch, length, channels)
            h_att = h.transpose(1, 2)
            
            # Apply self-attention
            h_att, _ = layer['attention'](h_att, h_att, h_att)
            
            # Convert back: (batch, length, channels) -> (batch, channels, length)
            h = h_att.transpose(1, 2)
        
        h_flat = h.reshape(h.size(0),-1)
        
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from latent space
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent representation back to sequence with attention
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
        Returns:
            Reconstructed sequence
        """
        h = self.fc_decoder(z)
        h = h.view(h.size(0), self.encoded_features, self.encoded_length)
        
        # Apply decoder layers with attention
        for layer in self.decoder_layers:
            # ConvTranspose layer
            h = layer['conv_transpose'](h)
            
            # Prepare for attention: (batch, channels, length) -> (batch, length, channels)
            h_att = h.transpose(1, 2)
            
            # Apply self-attention
            h_att, _ = layer['attention'](h_att, h_att, h_att)
            
            # Convert back: (batch, length, channels) -> (batch, channels, length)
            h = h_att.transpose(1, 2)
        
        # Final layer
        output = self.final_layer(h)
        
        # Ensure output has exact length
        if output.size(-1) != self.input_length:
            # Pad or crop to exact size
            if output.size(-1) < self.input_length:
                # Pad with zeros
                pad_size = self.input_length - output.size(-1)
                output = F.pad(output, (0, pad_size))
            else:
                # Crop to exact size
                output = output[:, :, :self.input_length]
        
        return output
    
    def forward(self, x):
        """
        Forward pass through the VAE
        Args:
            x: Input tensor of shape (batch_size, input_length) or (batch_size, 1, input_length)
        Returns:
            recon_x: Reconstructed output
            mu: Mean of latent space
            logvar: Log variance of latent space
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def sample(self, num_samples=1, device='cpu'):
        """
        Sample from the latent space
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)


def loss_function_vae(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function with reconstruction and KL divergence
    Args:
        recon_x: Reconstructed output
        x: Original input
        mu: Mean of latent space
        logvar: Log variance of latent space
        beta: Weight for KL divergence (for beta-VAE)
    Returns:
        Total loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss
