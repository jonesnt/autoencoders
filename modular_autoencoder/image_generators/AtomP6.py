# AtomP6
from image_generator import ImageGenerator
import torch

import numpy as np
import matplotlib.pyplot as plt
import AtomP1
import DataSim
import SetPoints
import sets
import sphangl


class AtomP6(ImageGenerator):
    """
    Generates images with atom-like patterns.
    """
    def generate(self, width=64, height=64):
        """
        Generates a single 64x64 image with atom-like patterns.
        """
        res = width * height
        atom_type = 1
        spread = 9+atom_type**2
        a1 = sphangl()
        a2 = sphangl()

        u1, v1, d1 = AtomP1(a1, atom_type, spread)
        u2, v2, d2 = AtomP1(a2, atom_type, spread)

        n = res + spread*4
        m = n-1
        s = int(np.floor(n/2))
        k = 2
        c = 2*np.random.rand(10, 1) - 1
        
        bb = sets(c, m, k)
        bi = (2*spread + np.arange(1, res+1) - 1).astype(int)
        bbb = bb[np.ix_(bi, bi)]

        plt.figure(10)
        plt.imshow(bbb)
        plt.colorbar()
    
        plt.show()

        u, v, d = SetPoints(u1, v1, d1, spread, s, bb)

        u0, v0, d0 = SetPoints(u2, v2, d2, spread, s, -bb)

        plt.figure(2)
        plt.plot(u * spread, v * spread, 'r.', label='u,v')
        plt.plot(u0 * spread, v0 * spread, 'b.', label='u0,v0')
        plt.legend()
        plt.show()

        u9 = np.concatenate([u, u0])
        v9 = np.concatenate([v, v0])
        d9 = np.concatenate([d, d0])

        img = DataSim(s, u9, v9, d9, spread, atom_type)
        img = torch.from_numpy(img)  # Convert NumPy array to PyTorch tensor
        return img.unsqueeze(0)  # Add channel dimension
