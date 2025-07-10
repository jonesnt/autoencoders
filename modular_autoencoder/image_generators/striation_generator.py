import torch
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import Image

try:
    from .image_generator import ImageGenerator
except ImportError:
    from image_generator import ImageGenerator


class StriationGenerator(ImageGenerator):
    def generate(self, width=64, height=64):
        img = torch.empty((width, height), dtype=torch.float32)

        # Generate random values for each pixel in image
        c = np.random.uniform(0.2,0.4)
        phi = (np.pi / 4) * np.random.uniform(0,1)
        d = (2 * np.pi) * np.random.uniform()
        for u in range(img.shape[0]):  # rows
            for v in range(img.shape[1]):  # columns
                img[u][v] = (np.sin(c * ((u + v * np.tan(phi)) * np.cos(phi) + d)) + 1) / 2

        # Add Poisson noise
        # A higher scaling factor means a higher signal-to-noise ratio (less noise).
        poisson_scaling_factor = 25.0  # Adjust to control Poisson noise level
        img_with_poisson_noise = torch.poisson(img * poisson_scaling_factor) / poisson_scaling_factor

        # Add Gaussian noise
        gaussian_std_dev = 0.05  # Adjust to control Gaussian noise level
        gaussian_noise = torch.randn_like(img) * gaussian_std_dev
        noisy_img = img_with_poisson_noise + gaussian_noise
        
        # Clamp the final values to be within [0, 1]
        img = torch.clamp(noisy_img, 0, 1)

        # Randomize orientation
        variant = np.random.randint(0,4)
        match variant:
            case 1:  # Transpose
                img = torch.transpose(img,0,1)
            case 2:  # Invert on x axis
                img = torch.flip(img, dims=[1])  # Equivalent to np.fliplr
            case 3:  # Invert on y axis
                img = torch.flip(img, dims=[0])  # Equivalent to np.flipud
            case _:
                pass

        return img

if __name__ == "__main__":
    # Create an instance of the generator
    generator = StriationGenerator()
    
    # Generate a test image (128x128 for better visualization)
    test_image = generator.generate(width=128, height=128)
    
    # Convert the PyTorch tensor to a numpy array and ensure it's in the right format for saving
    img_array = test_image.numpy()
    
    # Scale values to 0-255 range and convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    
    # Create a PIL Image from the numpy array
    img = Image.fromarray(img_array, mode='L')  # 'L' is for grayscale
    
    # Save the image
    output_path = "striation_test.png"
    img.save(output_path)
    
    # Also display using matplotlib for immediate feedback
    plt.figure(figsize=(5, 5))
    plt.imshow(img_array, cmap='gray')
    plt.title("Striation Generator Test Image")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("striation_test_plot.png")
    
    print(f"Test image generated and saved as '{output_path}' and 'striation_test_plot.png'")
