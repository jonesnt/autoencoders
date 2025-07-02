import numpy as np
from torch import linalg as LA
import torch
# outputs: 
# n_vec: the normal vector to the image plane
# v_space: This outputed as a 3x3 matrix (tensor) the three vectors alpha*u, beta*v, gamma*w.
# d: shift value of the imae plane
# sig1, sig2, sig3 are the step sizes of a rectangular lattice
def ortho(sig1=1, sig2=1, sig3=1, spread=1):
    # create normal vector n on unit sphere
    # Sample r1 uniformly from [-1, 1]
    r1 = 2 * np.random.rand() - 1
    # Sample r2 uniformly from [0, 2*pi]
    r2 = np.random.rand() * 2 * np.pi
    # calculate components of the unit normal vector 
    n1 = np.sqrt(1 - r1 ** 2) * np.cos(r2)
    n2 = np.sqrt(1 - r1 ** 2) * np.sin(r2)
    n3 = r1

    # uniform random direction normal to image plane
    n = torch.tensor([n1, n2, n3], dtype=torch.float32)

    # generate basis vectors of orthogonal coordinate system
    e1 = torch.tensor([1, 0, 0], dtype=torch.float32)
    e2 = torch.tensor([0, 1, 0], dtype=torch.float32)
    e3 = torch.tensor([0, 0, 1], dtype=torch.float32)

    # calculate projections of basis vectors onto normal vector n
    proj_e1 = sig1 * LA.norm(((torch.dot(n, e1)) / (torch.norm(n) ** 2)) * n)
    proj_e2 = sig2 * LA.norm(((torch.dot(n, e2)) / (torch.norm(n) ** 2)) * n)
    proj_e3 = sig3 * LA.norm(((torch.dot(n, e3)) / (torch.norm(n) ** 2)) * n)

    # create tensor containing projections
    proj_e = torch.tensor([proj_e1, proj_e2, proj_e3], dtype=torch.float32)

    # assign w direction : possible mirroring of directions
    if proj_e3 == torch.max(proj_e):
        gamma = sig3
        w = e3
        beta = sig2
        v = e2
        alpha = sig1
        u = e1
    elif proj_e2 == torch.max(proj_e):
        gamma = sig2
        w = e2
        beta = sig3
        v = e3
        alpha = sig1
        u = e1
    else:
        gamma = sig1
        w = e1
        beta = sig3
        v = e3
        alpha = sig2
        u = e2

    # choose random offset d along the normal vector
    d = np.random.rand() * n3 * gamma

    # generate image plane
    image_plane = n1 * u + n2 * v + n3 * w - d

    # turn image plane into tensor
    image_plane_tensor = torch.tensor(image_plane, dtype=torch.float32)

    # calculate scaling factor for matrix_A
    s_f = 1 / (np.sqrt(1 - (n1 ** 2)))

    # calculate matrix_A w/o scaling factor
    matrix_A_incomplete = torch.tensor([
        [1 - n1 ** 2, -n1 * n2, -n1 * n3],
        [0, n3, -n2]
    ], dtype=torch.float32)

    #full matrix A
    matrix_A = s_f * matrix_A_incomplete

    image_dat = {
        'image_plane_tensor': image_plane_tensor,
        's_f': s_f,
        'matrix_A': matrix_A,
        'd': d,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'n1': n1,
        'n2': n2,
        'n3': n3
    }

    return image_dat

