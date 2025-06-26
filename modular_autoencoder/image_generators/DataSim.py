# DataSim


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def DataSim(s, u1, v1, d, sp, type):
    res = 256
    n = res + sp*4
    k = n+2*sp
    dd = 2
    cc = 0.15 # Gaussian parameter (Default: 0.15)
    sz = (sp**2)/(64*type)
    brt1b = 0.5 # Threshold value of current background


    # colormap for the noise images
    cmg = np.ones((256, 3))
    cmg[:128, 0] = np.arange(128)/128.0
    cmg[:128, 1] = np.arange(128)/128.0
    cmg[128:, 1] = np.arange(127, -1, -1)/128.0
    cmg[128:, 2] = np.arange(127, -1, -1)/128.0

    # Noise parameters

    atom_vary = 0.2/5
    brt_vary = 0.9/10
    PNI = 4.5 # Poisson Noise Strength (default 3)
    GNI = 0.4 # Gaussian Noise Strength (default 1)
    StriS = 0.5 # Striation Strength (default 1)

    nn = len(u1)
    ju1 = np.floor(sp*u1).astype(int)
    jv1 = np.floor(sp*v1).astype(int)

    u = u1 + atom_vary*(np.random.randn(nn))
    v = v1 + atom_vary*(np.random.randn(nn))

    delta = brt_vary*(np.random.randn(1))
    if type == 2:
        delta = np.abs(brt_vary*(np.random.randn(1)/20))
    
    P = np.zeros((k, k))
    P1 = np.zeros((k, k))

    for l in range(nn):
        ku = ju1[l]+s
        kv = jv1[l]+s
        zx = (np.arange(-sp, sp) + sp*u1[l] - ju1[l])[:, np.newaxis]*np.ones((1, sp*2))
        zy = np.ones((sp*2, 1))*(np.arange(-sp, sp) + sp*v1[l] - jv1[l])
        R1 = 4*(((dd-d[l])/2)**1.5)*np.exp(-cc*(zx**2 + zy**2))/((((dd-d[l])**0.5))/sz)
        PP1 = P1[ku + np.arange(0, sp*2), kv + np.arange(0, sp*2)]

        zx = (np.arange(-sp, sp) + sp*u[l] - ju1[l])[:, np.newaxis]*np.ones((1, sp*2))
        zy = np.ones((sp*2, 1))*(np.arange(-sp, sp) + sp*v[l] - jv1[l])
        R = 4*(((dd-d[l])/2)**1.5)*np.exp(-cc*(zx**2 + zy**2))/((((dd-d[l])**0.5)+delta)/sz)
        PP = P[ku + np.arange(0, sp*2), kv + np.arange(0, sp*2)]

        if PP1[sp, sp]< brt1b:
            P[ku + np.arange(0, sp*2), kv + np.arange(0, sp*2)] = PP + R
            P1[ku + np.arange(0, sp*2), kv + np.arange(0, sp*2)] = PP1 + R1

    cl = 2*np.array([-1, 1])

    # Image without noise, small variations to position or brightness, non-adjusted grayscale
    P0F = P1[np.arange(k/2-res/2, k/2+res/2), np.arange(k/2-res/2, k/2+res/2)]
    aamin = np.min(P0F)
    aamax = np.max(P0F)
    aa = (P0F-aamin)/(aamax-aamin)

    plt.figure(8)
    plt.imshow(P0F, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    P7 = P0F.copy()
    plt.show()

    # Image without noise but with small variations to position or brightness
    P0F = P[np.arange(k/2-res/2, k/2+res/2), np.arange(k/2-res/2, k/2+res/2)]
    P7 = P7-P0F

    # Image showing brightness/position variations
    plt.figure(17)
    cmg_cmap = ListedColormap(cmg)
    plt.imshow(P7, cmap=cmg_cmap)
    plt.clim(cl)
    plt.colorbar()
    plt.axis('off')
    
    plt.show()

    xx = np.ones((2*k+1, 1))*np.arange(-k/2, k/2+0.5, 0.5)
    yy = np.arange(-k/2, k/2+0.5, 0.5)[:, np.newaxis]*np.ones((1, 2*k+1))
    ys = -np.sin((np.random.rand(1, 1)*np.pi)/sz)*xx + np.cos((np.random.rand(1, 1)*np.pi)/sz)*yy

    GaussN = GNI*(np.max(P)/8*np.random.rand(k, k))
    PoissonN = PNI*(P**0.5)*(0.025*np.random.rand(k, k))
    striN = (np.cos(ys[152:152+k, 152:152+k]*np.pi/3)+0.5)/2*StriS
    
    PF = GaussN.copy()
    P0F = PF[np.arange(k/2-res/2, k/2+res/2), np.arange(k/2-res/2, k/2+res/2)]

    # Image showing Gaussian noise component
    plt.figure(14)
    plt.imshow(P0F, cmap=cmg_cmap)
    plt.clim(cl)
    plt.colorbar()
    plt.axis('off')
    
    plt.show()

    PF = PoissonN.copy()
    P0F = PF[np.arange(k/2-res/2, k/2+res/2), np.arange(k/2-res/2, k/2+res/2)]

    # Image showing Gaussian noise component
    plt.figure(15)
    plt.imshow(P0F, cmap=cmg_cmap)
    plt.colorbar()
    plt.axis('off')
    plt.clim(cl)
    plt.show()

    PF = striN.copy()
    P0F = PF[np.arange(k/2-res/2, k/2+res/2), np.arange(k/2-res/2, k/2+res/2)]

    # Image showing striation noise component
    plt.figure(16)
    plt.imshow(P0F, cmap=cmg_cmap)
    plt.clim(cl)
    plt.colorbar()
    plt.axis('off')
    
    plt.show()

    # All noise components combined
    P0F = GaussN + PoissonN + striN
    P0F = PF[np.arange(k/2-res/2, k/2+res/2), np.arange(k/2-res/2, k/2+res/2)]
    plt.figure(18)
    plt.imshow(P0F+P7, cmap=cmg_cmap)
    plt.clim(cl)
    plt.colorbar()
    plt.axis('off')
   
    plt.show()

    PF = P + GaussN + PoissonN + striN
    P0F = PF[np.arange(k/2-res/2, k/2+res/2), np.arange(k/2-res/2, k/2+res/2)]
    plt.figure(13)
    aamin = np.min(P0F)
    aamax = np.max(P0F)
    plt.imshow(P0F, 'gray')
    plt.clim([0, aamax]) 
    plt.colorbar()
    plt.axis('off')
    plt.show()

    return aa