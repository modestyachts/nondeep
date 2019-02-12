import ctypes
from numpy.ctypeslib import ndpointer, as_array
import numpy as np
import time
from numpy import asfortranarray as F_ARRAY
from numpy import ascontiguousarray as C_ARRAY
from load_shared_lib import load_shared_lib
import pywren

ctypes_compute_gmm  = load_shared_lib().ctypes_compute_gmm

ctypes_compute_gmm.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ndpointer(ctypes.c_float)]

def compute_gmm(descs, n_gauss):
    n_dim = descs.shape[1]
    means = np.zeros((n_dim, n_gauss), dtype='float32')
    covars = np.zeros((n_dim, n_gauss), dtype='float32')
    coefs = np.zeros((n_gauss), dtype='float32')
    descs = C_ARRAY(descs)
    descs = descs.astype('float32')

    ctypes_compute_gmm(n_gauss, n_dim, descs, descs.shape[0], 0, means, covars, coefs)
    print(coefs)
    print(np.sum(coefs))
    return means.T, covars.T, coefs


if __name__ == "__main__":
    descs = np.random.randn(int(1e4), 128).astype('float32')
    compute_gmm(descs, 16)


    descs = np.array([[3,3],[8,8], [16,16]]).astype('float32')
    print(descs)
    means, covars, coefs = compute_gmm(descs, 3)
    print("Means:")
    print(means)
    assert(np.all(np.isclose(means, descs)))








