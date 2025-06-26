import torch
import sys
from image_generators.sinusoidal_image_generator import SinusoidalImageGenerator
from autoencoder_enum import AUTOENCODERS  # or import your model class directly

def load_model(model_path, model_name="BASIC"):
    # Instantiate the model
    model_class = AUTOENCODERS[model_name].value
    model = model_class()
    # Load checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, image):
    with torch.no_grad():
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        output = model(image)
    return output

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model_eval.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    model = load_model(model_path)
    generator = SinusoidalImageGenerator()
    image = generator.generate(64, 64)
    image = torch.tensor(image, dtype=torch.float32)
    output = evaluate_model(model, image)
    print("Model output:", output)

    # Save the output as an image
    from PIL import Image
    import numpy as np
    # If output is a tuple (encoded, decoded), use decoded
    if isinstance(output, tuple):
        output_img = output[1]
    else:
        output_img = output
    # Remove batch and channel dimensions if present
    output_img = output_img.squeeze().cpu().detach().numpy()
    # Normalize to [0, 255] for saving
    output_img = (output_img * 63).astype(np.uint8)
    img = Image.fromarray(output_img, mode='L')
    img.save('test_eval.png')
    print("Saved model output as test_eval.png")