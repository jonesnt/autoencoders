# SetPoints


import numpy as np

def SetPoints(u1, v1, d1, sp, s, bb):
    nn = len(u1)
    ju1 = np.floor(sp*u1).astype(int)
    jv1 = np.floor(sp*v1).astype(int)

    u = np.zeros(nn)
    v = np.zeros(nn)
    d = np.zeros(nn)

    m=0

    for l in range(nn):
        if np.abs(ju1[l])<s and np.abs(jv1[l])<s:
            ku = ju1[l]+s
            kv = jv1[l]+s
            if bb[ku, kv] > 0:
                u[m] = u1[l]
                v[m] = v1[l]
                d[m] = d1[l]
                m += 1

    u = u[:m]
    v = v[:m]
    d = d[:m]

    return u, v, d