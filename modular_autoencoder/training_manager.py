import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse

from modular_autoencoder.autoencoder_enum import AUTOENCODERS
from modular_autoencoder.image_handler import ImageHandler

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

def main(epochs, batch_size, learning_rate, model_name, num_images):
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model instantiation
    if model_name in AUTOENCODERS.__members__:
        model_class = AUTOENCODERS[model_name].value
        model = model_class().to(device)
    else:
        print(f"Model {model_name} not found in AUTOENCODERS enum.")
        return

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create training data
    print(f"Generating {num_images} training images...")
    image_handler = ImageHandler()
    data = image_handler.generate_training_data(num_images=num_images, width=64, height=64)
    print(f"Data shape: {data[0].shape}")

    # Split train/test
    train_data = data[:np.floor(3/5 * num_images)]
    test_data = data[np.floor(3/5 * num_images):]

    print("starting training...")
    
    train_losses = []
    avg_loss = 0  # Initialize avg_loss to ensure it is always defined

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        perm = torch.randperm(len(train_data))
        train_data = train_data[perm]

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

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
            print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6f}')

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
        
    print("Training complete.")

    # Test model
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
    }, 'autoencoder_out.pth')

    print("Model saved as 'autoencoder_out.pth'")

    # Save some example reconstructions as images
    save_reconstructions(test_data[:4], test_decoded)

    # Plot training loss
    plot_training_loss(train_losses)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Manager for Modular Autoencoder")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--model_name', type=str, default='BASIC', help='Name of model to use from enumeration')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to generate for training')
    args = parser.parse_args()

    main(args.epochs, args.batch_size, args.learning_rate, args.model_name, args.num_images)
