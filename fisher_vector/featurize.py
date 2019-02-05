import fisher
import numpy as np
from sift import sift
from lcs import lcs
import scipy.misc
import boto3
import pywren
import time

def image_fisher_featurize(im_key, gmm_sift, gmm_lcs, pca_sift, pca_lcs):
    t = time.time()
    s3 = boto3.resource('s3')
    s3.Bucket("pictureweb").download_file(im_key, "/tmp/img.jpg")
    im = scipy.misc.imread("/tmp/img.jpg", flatten=True)
    descs = sift(im).dot(pca_sift.T)
    sift_features = fisher.fisher_vector_features(descs.astype('float32'), *gmm_sift)
    im = scipy.misc.imread("/tmp/img.jpg")
    descs = lcs(im).reshape(-1, 96).dot(pca_lcs.T)
    lcs_features = fisher.fisher_vector_features(descs.astype('float32'), *gmm_lcs)
    out_features = np.hstack((sift_features, lcs_features)).T
    e = time.time()
    return out_features, e - t

if __name__ == "__main__":
    #img = scipy.misc.imread('test.png')
    im_key = 'imagenet_train_100k_pixels/0/10026.JPEG'
    n_gauss = 4096
    n_dim = 64
    np.random.seed(0)
    means = np.random.rand(n_gauss, n_dim).astype('float32')
    covars = np.random.rand(n_gauss, n_dim).astype('float32')
    priors = np.ones(n_gauss).astype('float32')/n_gauss
    gmm = (means, covars, priors)
    pca_sift = np.random.randn(n_dim,128).astype('float32')
    pca_lcs = np.random.randn(n_dim,96).astype('float32')
    pwex = pywren.default_executor()
    #image_fisher_featurize(im_key, gmm, gmm, pca_sift, pca_lcs)
    futures = pwex.map(lambda x: image_fisher_featurize(im_key, gmm, gmm, pca_sift, pca_lcs), [0])
    feats, tot_time = futures[0].result()
    print("Featurize took {0}".format(tot_time))
    print(feats.shape)






