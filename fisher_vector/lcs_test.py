import numpy as np
import lcs
import time
import io
import boto3
import scipy.misc
import fisher
import pywren

t = time.time()
im_key = "imagenet_train_100k_pixels/8/11857.JPEG"

client = boto3.client('s3')
bio = io.BytesIO(client.get_object(Bucket="pictureweb", Key=im_key)["Body"].read())

pca_mat_lcs = np.load("pca_mat_lcs.npy")
weights = np.load('gmm_lcs_weights_{0}_centers.npy'.format(16))
means = np.load('gmm_lcs_means_{0}_centers.npy'.format(16))
covars= np.load('gmm_lcs_covars_{0}_centers.npy'.format(16))
gmm_lcs = (means, covars, weights)

img = scipy.misc.imread(bio, flatten=True)
pwex = pywren.default_executor()
z = lcs.lcs(img).reshape(-1, 96)






