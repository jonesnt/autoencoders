import numpy as np
import torch

def ortho(sig1, sig2, sig3):
    r1 = np.random.uniform(-1, 1.0000001)  # Adding a small epsilon to make upper bound inclusive
    r2 = np.random.uniform(0, np.pi * 2)  # np.random.uniform upper bound is already exclusive
    n1 = np.sqrt(1 - r1*r1) * np.cos(r2)
    n2 = np.sqrt(1 - r1*r1) * np.sin(r2)
    n3 = r1

    n = torch.tensor([n1, n2, n3], dtype=torch.float32)
