# sets


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def bb(c, m, k):
    n = m-2*k
    f = np.zeros((n, n))
    y = 1.6*(2*np.arange(1, n+1)-1)/n-1
    x = y.reshape(-1, 1)    
    o1 = np.ones((n, 1))
    f = (
    o1*(c[0]*o1.reshape(-1, 1)/3 + c[1]*y + c[2]*(y**2) + c[3]*(y**3))
    + x*(c[4]*o1.reshape(-1, 1) + c[5]*y + c[6]*(y**2))
    + (x**2)*(c[7]*o1.reshape(-1, 1) + c[8]*y)
    + (x**3)*(c[9]*o1.reshape(-1, 1))
)
    b = (np.sign(f)+1)/2
    kk = 2*k+1
    k2 = (kk/2)**2
    z = np.zeros((kk, kk))
    for j1 in range(kk):
        for j2 in range(kk):
            if (j1-k-1)**2+(j2-k-1)**2 < k2:
                    z[j1, j2] = 1


    bb1 = convolve2d(b, z, mode='same')/np.sum(z)
    bb2 = convolve2d(1-b, z, mode='same')/np.sum(z)
    bb = np.round(bb1-bb2)

    plt.figure(22)
    plt.imshow(bb, cmap='viridis')
    plt.colorbar()
    plt.axis('off')

    plt.figure(23)
    plt.imshow(b, cmap='viridis')
    plt.colorbar()

    plt.show()
    
    return bb