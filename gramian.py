import numpy as np


def gram_matrix(v):
    n_channels = v.shape[0]
    print str(n_channels)
    nv = np.reshape(v, (n_channels, -1))
    print str(nv.shape)
    gram = np.dot(nv, nv.T)
    return gram
