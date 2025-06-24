from torch import nn
from modular_autoencoder.autoencoder_base import AutoencoderBase

class ImageAutoencoder(AutoencoderBase, nn.Module):
    """
    Autoencoder for 64x64 grayscale images
    Compresses image to latent space and reconstructs it
    """
    def __init__(self, latent_dim=4):
        super(ImageAutoencoder, self).__init__()
        
        # Encoder: 64x64 -> latent_dim
        self.encoder = nn.Sequential(
            # First conv block: 64x64 -> 32x32
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Second conv block: 32x32 -> 16x16  
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Third conv block: 16x16 -> 8x8
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Fourth conv block: 8x8 -> 4x4
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Flatten and compress to latent space
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim)
        )
        
        # Decoder: latent_dim -> 64x64
        self.decoder = nn.Sequential(
            # Expand from latent space
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.ReLU(),
            
            # Reshape to feature maps
            nn.Unflatten(1, (64, 4, 4)),
            
            # First deconv: 4x4 -> 8x8
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Second deconv: 8x8 -> 16x16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Third deconv: 16x16 -> 32x32
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Final deconv: 32x32 -> 64x64
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
