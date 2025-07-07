from PIL import Image
import torch
from simple_generator import SimpleGenerator

def test_generate_output_shape():
    gen = SimpleGenerator()
    width, height = 16, 16
    image = gen.generate(width=width, height=height)
    assert isinstance(image, torch.Tensor), "Output is not a torch.Tensor"
    assert image.shape == (width, height), f"Output shape {image.shape} != ({width}, {height})"

def test_generate_output_dtype():
    gen = SimpleGenerator()
    image = gen.generate(width=8, height=8)
    assert image.dtype == torch.float32, f"Output dtype {image.dtype} != torch.float32"

def test_generate_nonnegative():
    gen = SimpleGenerator()
    image = gen.generate(width=8, height=8)
    assert torch.all(image >= 0), "Image contains negative values"

def save_tensor_as_png(tensor, filename="generated_image.png"):
    # Normalize to [0, 255] and convert to uint8
    norm_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    arr = (norm_tensor.cpu().numpy() * 255).astype('uint8')
    img = Image.fromarray(arr)
    img.save(filename)

if __name__ == "__main__":
    test_generate_output_shape()
    test_generate_output_dtype()
    test_generate_nonnegative()
    # Save a sample image
    gen = SimpleGenerator()
    image = gen.generate(width=64, height=64)
    save_tensor_as_png(image, "generated_image.png")
    print("All tests passed and image saved.")