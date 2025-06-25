# sphangl


import numpy as np

def sphangl():
    w = 2
    a0 = np.random.rand(3,1)

    while w > 1:
        a = np.random.rand()
        a2 = np.random.rand(2, 1)
        s = np.sqrt(np.sum(a2**2))
        w = a*s

    a0 = np.array([
        [a2[0, 0]/s],
        [a2[0, 0]/s],
        [np.random.rand()],
        [s]
    ])

    return a0