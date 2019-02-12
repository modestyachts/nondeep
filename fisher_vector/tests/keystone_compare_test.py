import sys
sys.path.insert(0, "..")
import fisher
import argparse
import numpy as np
import boto3
import scipy.misc
import sift
import time
import pywren
from sklearn.mixture import GaussianMixture
import io
import scipy.linalg
import sklearn.metrics as metrics
from numba import jit
import concurrent.futures as fs
import numpy as np
import lcs
from numpy import ascontiguousarray as C_ARRAY

def convert_keystone_csv_to_numpy(fname):
    return np.array([[float(x) for x in y.split(",")] for y in open(fname).read().strip().split("\n")])

if __name__ == "__main__":
    im = scipy.misc.imread("./ILSVRC2012_val_00000293.JPEG")
    # convert image to BGR 
    im = im[:, :, ::-1]
    descs = lcs.lcs(im).reshape(-1, 96)
    descs_mean = descs[:, :48]
    descs_std = descs[:, 48:]
    descs  = np.vstack((descs_mean,descs_std)).reshape((-1,96),order='F').T

    descs_keystone = C_ARRAY(convert_keystone_csv_to_numpy("./lcs_imagenet.txt"))
    pca_mat = convert_keystone_csv_to_numpy("./pcaMat_lcs.csv")
    weights = convert_keystone_csv_to_numpy("./gmmCoefs_lcs.csv")
    means = convert_keystone_csv_to_numpy("./gmmMeans_lcs.csv").T
    covars = convert_keystone_csv_to_numpy("./gmmVars_lcs.csv").T
    descs = pca_mat.dot(descs)
    pca_keystone = convert_keystone_csv_to_numpy("./pca_keystone_lcs.txt")
    gmm = (means, covars, weights)
    fv_keystone = convert_keystone_csv_to_numpy("./fisher_keystone_lcs.txt")
    fv_features = fisher.fisher_vector_features(descs.astype('float32').T, *gmm)


    print("AVG LCS DIFF ", np.mean(np.abs(fv_features - fv_keystone)))
    print("MAX LCS DIFF ", np.max(np.abs(fv_features - fv_keystone)))

    im = scipy.misc.imread("./ILSVRC2012_val_00000293.JPEG", flatten=True)
    im /= 255.0

    descs = sift.sift(im).T
    descs_keystone = C_ARRAY(convert_keystone_csv_to_numpy("./sift_imagenet.txt"))

    pca_mat = convert_keystone_csv_to_numpy("./pcaMat_sift.csv")
    weights = convert_keystone_csv_to_numpy("./gmmCoefs_sift.csv")
    means = convert_keystone_csv_to_numpy("./gmmMeans_sift.csv").T
    covars = convert_keystone_csv_to_numpy("./gmmVars_sift.csv").T


    descs = pca_mat.dot(descs)
    pca_keystone = convert_keystone_csv_to_numpy("./pca_keystone_sift.txt")



    gmm = (means, covars, weights)
    fv_keystone = convert_keystone_csv_to_numpy("./fisher_keystone_sift.txt")
    fv_features = fisher.fisher_vector_features(descs.astype('float32').T, *gmm)
    print("AVG SIFT DIFF ", np.mean(np.abs(fv_features - fv_keystone)))
    print("MAX SIFT DIFF ", np.max(np.abs(fv_features - fv_keystone)))


