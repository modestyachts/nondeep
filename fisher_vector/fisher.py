import ctypes
from numpy.ctypeslib import ndpointer, as_array
import numpy as np
import time
from numpy import asfortranarray as F_ARRAY
from numpy import ascontiguousarray as C_ARRAY
from load_shared_lib import load_shared_lib
import pywren

ctypes_get_fisher_features = load_shared_lib().ctypes_get_fisher_features
#ctypes_get_fisher_features = ctypes.cdll.LoadLibrary('lib/libImageFeatures.so').ctypes_get_fisher_features
ctypes_get_fisher_features.argtypes = [ndpointer(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int]
ctypes_get_fisher_features.restype = ctypes.POINTER(ctypes.c_float)

def fisher_vector_features(descriptors, gmm_means, gmm_covars, gmm_priors):
    n_gauss = gmm_means.shape[0]
    n_dim = gmm_means.shape[1]
    fvenc_length = 2 * n_dim * n_gauss

    assert(descriptors.shape[1] == n_dim)
    descriptors = C_ARRAY(descriptors)
    gmm_means = C_ARRAY(gmm_means.astype('float32'))
    gmm_covars = C_ARRAY(gmm_covars.astype('float32'))
    gmm_priors = C_ARRAY(gmm_priors.astype('float32'))


    ctypes_get_fisher_features.restype = ndpointer(dtype=ctypes.c_float, shape=(fvenc_length,))
    fisher_features = ctypes_get_fisher_features(gmm_means, gmm_means.shape[0], n_dim, n_gauss, gmm_covars, gmm_covars.shape[0], gmm_priors, gmm_priors.shape[0], descriptors, descriptors.shape[0])
    return fisher_features


if __name__ == "__main__":
    n_gauss = 16
    n_dim = 64
    np.random.seed(0)
    descs = np.genfromtxt("fisher_descs.txt",delimiter=",").reshape(64, 13165).astype('float32')
    means = np.genfromtxt("fisher_means.txt", delimiter=",").reshape(n_dim, n_gauss).astype('float32').T
    covars = np.genfromtxt("fisher_covars.txt", delimiter=",").reshape(n_dim, n_gauss).astype('float32').T
    priors = np.genfromtxt("fisher_weights.txt", delimiter=",").reshape(n_gauss).astype('float32')
    t = time.time()
    features = fisher_vector_features(descs.T, means, covars, priors)
    e = time.time()
    print("FETURIZATION RUNTIME", e - t)
    fisher_keystone = np.array([ [float(y) for y in x.strip().split(",")] for x in open("keystone_fisher.txt").read().strip().split("\n")])
    print(fisher_keystone)
    print(features)
    assert(np.mean(np.abs(fisher_keystone - features)) < 1e-3)
    print("Python Fisher matches Keystone Fisher")












