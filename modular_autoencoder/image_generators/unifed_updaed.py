import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ortho import ortho


class OrthoLatticeGenerator:
    def __init__(self, lattice_type="primitive"):
        """
        Generator for orthorhombic and cubic lattices.

        Parameters
        ----------
        lattice_type : str
            Type of lattice to generate. Options are:
            - "primitive" : orthorhombic primitive lattice
            - "bcc"       : body-centered orthorhombic lattice
            - "fcc"       : face-centered orthorhombic lattice
            - "cubic"     : cubic lattice (special case of orthorhombic with equal step sizes)
        """
        valid_types = ["primitive", "bcc", "fcc", "cubic"]
        if lattice_type not in valid_types:
            raise ValueError(f"Invalid lattice_type {lattice_type}, must be one of {valid_types}")
        self.lattice_type = lattice_type

    def generate(self, width=64, height=64, sigma=1.5, amplitude=1.0,
                 lattice_range=10, density=4, seed=None):
        """
        Generate an image of the selected lattice projected onto 2D.

        Parameters
        ----------
        width, height : int
            Dimensions of the output image (in pixels).
        sigma : float
            Standard deviation of the Gaussian "atom blobs" (in pixels).
        amplitude : float
            Intensity scaling for each Gaussian blob.
        lattice_range : int
            Range of integer lattice coordinates to sample in each direction.
        density : float
            Controls how many lattice spacings fit into the image.
        seed : int or None
            Random seed for orientation reproducibility.
            - If None, a random seed is chosen automatically.
            - If int, the same lattice orientation will be regenerated.

        Returns
        -------
        (torch.Tensor, int)
            A (width x height) tensor containing the generated grayscale lattice image,
            normalized to [0, 1], and the seed used.
        """
        # ----- Handle random seed -----
        if seed is None:
            seed = np.random.randint(0, 1e6)  # pick random seed if not given
        np.random.seed(seed)  # apply seed so ortho() uses this orientation
        print(f"Using random seed: {seed}")

        # Pixel grid
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y, indexing="ij")
        image = np.zeros((width, height), dtype=np.float32)

        # Get orientation + projection info
        if self.lattice_type == "cubic":
            image_dat = ortho(sig1=1, sig2=1, sig3=1)
        else:
            image_dat = ortho()
        A = image_dat["matrix_A"].numpy()

        # ---- Define basis atoms depending on lattice type ----
        if self.lattice_type in ["primitive", "cubic"]:
            basis_atoms = [(0.0, 0.0, 0.0)]
        elif self.lattice_type == "bcc":
            basis_atoms = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
        elif self.lattice_type == "fcc":
            basis_atoms = [
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.0),
                (0.5, 0.0, 0.5),
                (0.0, 0.5, 0.5),
            ]

        # ---- Loop over lattice points ----
        for i in range(-lattice_range, lattice_range):
            for j in range(-lattice_range, lattice_range):
                for k in range(-lattice_range, lattice_range):
                    for dx, dy, dz in basis_atoms:
                        atom_coords = A @ np.array([i + dx, j + dy, k + dz], dtype=np.float32)
                        x0, y0 = atom_coords

                        # Scale & center
                        scale = width / density
                        x0 = int(x0 * scale + width // 2)
                        y0 = int(y0 * scale + height // 2)

                        if not (0 <= x0 < width and 0 <= y0 < height):
                            continue

                        gauss = amplitude * np.exp(
                            -((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)
                        )
                        image += gauss

        # Normalize
        image = image / image.max()
        return torch.tensor(image, dtype=torch.float32), seed


def save_tensor_as_png(tensor, seed, lattice_type="primitive", filename=None):
    """
    Save a [0,1] float tensor as an 8-bit grayscale PNG.

    Parameters
    ----------
    tensor : torch.Tensor
        Image tensor in [0, 1].
    seed : int
        Seed number used for generation.
    lattice_type : str
        The lattice type (used in filename).
    filename : str or None
        If None, auto-generate filename as {lattice_type}_seed{seed}.png.
    """
    if filename is None:
        filename = f"{lattice_type}_seed{seed}.png"

    arr = (tensor.cpu().numpy() * 255).astype("uint8")
    img = Image.fromarray(arr, mode="L")
    img.save(filename)
    print(f"Saved image to {filename}")


if __name__ == "__main__":
    # Choose lattice type: "primitive", "bcc", "fcc", "cubic"
    gen = OrthoLatticeGenerator(lattice_type="fcc")

    # Generate with random seed
    image, seed = gen.generate(width=64, height=64, density=5, seed=None)

    # Save image with seed in filename
    save_tensor_as_png(image, seed, lattice_type=gen.lattice_type)

    # Show image with seed label
    plt.imshow(image.cpu().numpy(), cmap="gray")
    plt.title(f"{gen.lattice_type.upper()} Lattice - Seed {seed}")
    plt.show()
