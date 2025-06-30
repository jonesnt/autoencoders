# AtomP6


import numpy as np
import matplotlib.pyplot as plt
import AtomP1
import DataSim
import SetPoints
import sets
import sphangl

def main():
    res = 256
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

    aa = DataSim(s, u9, v9, d9, spread, atom_type)

    pass