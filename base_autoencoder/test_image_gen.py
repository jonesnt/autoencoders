import numpy as np
import torch
from PIL import Image
import autoencoder_256 as ae

def generate_images(num_images=5):
    images = ae.create_sample_data(num_images)
    return images

def save_images(images):
    for i in range(len(images)):
        # Original image
        img = images[i].squeeze().cpu().numpy()
        fnal_image = Image.fromarray((img * 63).astype(np.uint8), mode='L')
        fnal_image.save(f'sample_image_{i}.png')
    
    print("Saved " + str(len(images)) + " sample images.")

def main():
    save_images(generate_images())

if __name__ == "__main__":
    main() 