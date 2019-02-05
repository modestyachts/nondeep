import pywren
import fisher
import numpy as np
import time

if __name__ == "__main__":
    n_gauss = 16
    n_dim = 64
    np.random.seed(0)

    descs = np.genfromtxt("fisher_descs.txt",delimiter=",").reshape(64, 13165).astype('float32').T
    means = np.genfromtxt("fisher_means.txt", delimiter=",").reshape(n_dim, n_gauss).astype('float32').T
    covars = np.genfromtxt("fisher_covars.txt", delimiter=",").reshape(n_dim, n_gauss).astype('float32').T
    priors = np.genfromtxt("fisher_weights.txt", delimiter=",").reshape(n_gauss).astype('float32')



    def time_fv(_):
        t = time.time()
        features = fisher.fisher_vector_features(descs, means, covars, priors)
        e = time.time()
        return e - t, features
    pwex = pywren.default_executor()
    futures = pwex.map(time_fv, range(1))
    print("PYWREN FEATURIZATION RUNTIME ", futures[0].result()[0])

    features = futures[0].result()[1]
    fisher_keystone = np.array([ [float(y) for y in x.strip().split(",")] for x in open("keystone_fisher.txt").read().strip().split("\n")])
    assert(np.mean(np.abs(fisher_keystone - features)) < 1e-3)
    print("Python Fisher matches Keystone Fisher")



