from image_generator import ImageGenerator
import torch
import numpy as np

class SinusoidalImageGenerator(ImageGenerator):
    """
    Generates images with sinusoidal patterns.
    """
    def generate(self, width=64, height=64):
        """
        Generates a single 64x64 image with a random sinusoidal pattern.
        """
        img = torch.empty((width, height))
        
        # Generate random values for the sinusoidal pattern
        c = np.random.uniform(0.2, 0.4)
        phi = (np.pi / 4) * np.random.uniform(0, 1)
        d = (2 * np.pi) * np.random.uniform()
        
        for u in range(img.shape[0]):  # rows
            for v in range(img.shape[1]):  # columns
                img[u][v] = (np.sin(c * ((u + v * np.tan(phi)) * np.cos(phi) + d)) + 1) / 2
        
        # Add Gaussian Noise
        mean = 0.0
        std_dev = 0.02
        gaussian_noise = torch.normal(mean, std_dev, img.shape)
        img = img + gaussian_noise
        
        # Clamp values to [0, 1] range
        img = torch.clamp(img, 0, 1)
        
        return img.unsqueeze(0) # Add channel dimension
