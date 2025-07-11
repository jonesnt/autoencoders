import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class ImageAutoencoder(nn.Module):
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

def create_sample_data(num_samples=100):
    """
    Create sample 64x64 images for testing
    Replace this with your actual data loading function
    """
    # Create random patterns similar to your microscopy data
    data = []
    for i in range(num_samples):
        # Create empty 64x64 tensor
        img = torch.empty((64, 64))
        
        # Generate random values for each pixel in image
        c = 0.4
        phi = (np.pi / 4) * np.random.uniform(0,1)
        d = (2 * np.pi) * np.random.uniform()
        for u in range(img.shape[0]):  # rows
            for v in range(img.shape[1]):  # columns
                img[u][v] = (np.sin(c * ((u + v * np.tan(phi)) * np.cos(phi) + d)) + 1) / 2

        # Randomize orientation (apply once per image)
        variant = np.random.randint(0, 4)
        if variant == 1:  # Transpose
            img = torch.transpose(img, 0, 1)
        elif variant == 2:  # Flip horizontal
            img = torch.fliplr(img)
        elif variant == 3:  # Flip vertical
            img = torch.flipud(img)
        # variant == 0: no transformation
        
        # Clamp values to [0, 1] range to prevent outliers
        img = torch.clamp(img, 0, 1)
        
        data.append(img.unsqueeze(0))  # Add channel dimension
    
    return torch.stack(data)

def load_real_images(data_folder, num_samples=None):
    """
    Load real 256x256 images from a folder
    Replace create_sample_data() with this function when you have real data
    
    Args:
        data_folder: Path to folder containing .png or .jpg images
        num_samples: Number of images to load (None = load all)
    
    Returns:
        torch.Tensor: Stack of images with shape (N, 1, 256, 256)
    """
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(data_folder, ext)))
    
    if num_samples:
        image_files = image_files[:num_samples]
    
    images = []
    for img_path in image_files:
        # Load and convert to grayscale
        img = Image.open(img_path).convert('L')
        
        # Resize to 256x256 if needed
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [0, 1]
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
        images.append(img_tensor)
    
    return torch.stack(images)

def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = ImageAutoencoder(latent_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create sample data (replace with real data loading)
    print("Creating sample data...")
    data = create_sample_data(200).to(device)
    print(f"Data shape: {data.shape}")
    
    # Uncomment these lines when you have real data:
    # print("Loading real images...")
    # data = load_real_images("/path/to/your/images", num_samples=200).to(device)
    # print(f"Data shape: {data.shape}")
    
    # Split into train/test
    train_data = data[:160]
    test_data = data[160:]
    
    print("Starting training...")
    
    # Training loop
    num_epochs = 5000
    batch_size = 8
    
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Shuffle training data
        perm = torch.randperm(len(train_data))
        train_data = train_data[perm]
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Forward pass
            encoded, decoded = model(batch)
            loss = criterion(decoded, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}')
            
            # Monitor reconstruction quality every 100 epochs
            if epoch % 100 == 0 and epoch > 0:
                model.eval()
                with torch.no_grad():
                    sample_batch = train_data[:4]
                    _, sample_decoded = model(sample_batch)
                    sample_loss = criterion(sample_decoded, sample_batch)
                    print(f'  Sample reconstruction loss: {sample_loss:.6f}')
                    
                    # Check if outputs are too close to zero
                    mean_output = sample_decoded.mean().item()
                    print(f'  Mean output value: {mean_output:.4f}')
                    if mean_output < 0.01:
                        print("  WARNING: Outputs very close to zero - model may be collapsing!")
                model.train()
    
    print("Training completed!")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_encoded, test_decoded = model(test_data[:4])
        test_loss = criterion(test_decoded, test_data[:4])
        print(f'Test Loss: {test_loss:.6f}')
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_loss,
        'train_losses': train_losses,
        'latent_dim': 128,
    }, 'autoencoder_256.pth')
    
    print("Model saved as 'autoencoder_256.pth'")
    
    # Save some example reconstructions as images
    save_reconstructions(test_data[:4], test_decoded)
    
    # Plot training loss
    plot_training_loss(train_losses)

def save_reconstructions(original, reconstructed):
    """Save original and reconstructed images for comparison"""
    for i in range(len(original)):
        # Original image
        orig_img = original[i].squeeze().cpu().numpy()
        orig_pil = Image.fromarray((orig_img * 63).astype(np.uint8), mode='L')
        orig_pil.save(f'original_{i}.png')
        
        # Reconstructed image  
        recon_img = reconstructed[i].squeeze().cpu().numpy()
        recon_pil = Image.fromarray((recon_img * 63).astype(np.uint8), mode='L')
        recon_pil.save(f'reconstructed_{i}.png')
    
    print("Saved example images: original_*.png and reconstructed_*.png")

def plot_training_loss(losses):
    """Plot and save training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved training loss plot as 'training_loss.png'")

def load_trained_model(model_path='autoencoder_256.pth'):
    """
    Load a previously trained model
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        model: Loaded PyTorch model
        info: Dictionary with training information
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with saved latent dimension
    latent_dim = checkpoint.get('latent_dim', 128)
    model = ImageAutoencoder(latent_dim=latent_dim).to(device)
    
    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    info = {
        'final_loss': checkpoint.get('final_loss', None),
        'train_losses': checkpoint.get('train_losses', []),
        'latent_dim': latent_dim
    }
    
    print(f"Loaded model with latent dimension: {latent_dim}")
    print(f"Final training loss: {info['final_loss']:.6f}")
    
    return model, info

if __name__ == "__main__":
    main() 