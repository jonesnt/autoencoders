# AtomP1


import numpy as np

def AtomP1(a0, atom_type, spread):
    dd = 2
    res = 256

    n = res + spread*4
    k = n +2*spread

    c = a0[0]
    s = a0[1]
    q = a0[3]
    sq = np.sqrt(1+q**2)
    rho = np.array([c, s, q]) / sq

    d = np.zeros(((2*k+1)**2))

    if atom_type==1:
        xx = np.ones((2*k+1, 1)) * np.arange(-k, k+1)
        x = xx.flatten()
        yy = np.arange(-k, k+1).reshape(-1, 1) * np.ones((1, 2*k+1))
        y = yy.flatten()
        z = q * (c * x + s * y)
        zi = np.floor(z)
        d = (z - zi) / sq
        
    elif atom_type==2:
        xx = np.ones((2*k+1, 1)) * np.arange(-k/2, k/2+0.5, 0.5)
        x = xx.flatten()
        yy = np.arange(-k/2, k/2+0.5, 0.5).reshape(-1, 1) * np.ones((1, 2*k+1))
        y = yy.flatten()
        z = q * (c * x + s * y)
        zi = np.zeros((2*k+1)**2)
        for i in range(2*k+1):
            for j in range(2*k+1):
                ij = (2*k+1)*(i) + j
                if i%2 != 0 and j%2 != 0:
                    zi[ij] = np.floor(z[ij])
                    d[ij] = z[ij] - zi[ij]
                elif i%2 == 0 and j%2 == 0:
                    if (z[ij]-np.floor(z[ij])) < 0.5:
                            zi[ij] = np.floor(z[ij])-0.5
                            d[ij] = (z[ij] - zi[ij])/sq
                    else:
                            zi[ij] = np.floor(z[ij])+0.5
                            d[ij] = (z[ij] - zi[ij])/sq
                else:
                    d[ij] = dd - 0.001
    elif atom_type==3:
        xx = np.ones((2*k+1, 1)) * np.arange(-k/2, k/2+0.5, 0.5)
        x = xx.flatten()
        yy = np.arange(-k/2, k/2+0.5, 0.5).reshape(-1, 1) * np.ones((1, 2*k+1))
        y = yy.flatten()
        z = q * (c * x + s * y)
        zi = np.zeros((2*k+1)**2)
        for i in range(2*k+1):
            for j in range(2*k+1):
                ij = (2*k+1)*(i) + j
                if (i%2 != 0 and j%2 != 0) or (i%2 == 0 and j%2 == 0):
                    zi[ij] = np.floor(z[ij])
                    d[ij] = z[ij] - zi[ij]
                else:
                    if (z[ij]-np.floor(z[ij])) < 0.5:
                        zi[ij] = np.floor(z[ij])-0.5
                    else:
                        zi[ij] = np.floor(z[ij])+0.5
                    d[ij] = (z[ij] - zi[ij])/sq

    uu = rho[0] * x + rho[1] * y + rho[2] * zi
    vv = -s*x + c*y
    c3 = np.cos(a0[2]*np.pi)
    s3 = np.sin(a0[2]*np.pi)
    u1 = c3*uu + s3*vv
    v1 = -s3*uu + c3*vv

    return u1, v1, d.flatten()