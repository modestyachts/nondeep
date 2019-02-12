import numpy as np
from numpywren.matrix import BigMatrix, BigSymmetricMatrix
import numpywren.matrix_utils as utils
import concurrent.futures as fs
import time
import scipy.linalg
import sklearn.metrics
from sklearn.datasets import fetch_mldata
from numpywren.matrix_init import local_numpy_init, shard_matrix
from scipy.linalg import solve
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shard dual model.')
    t = time.time()
    parser.add_argument('model_path', type=str, help="model_path")
    args = parser.parse_args()
    model = np.load(args.model_path)
    model_key = args.model_path.replace(".npy", "")
    bigm  = BigMatrix(model_key, bucket="pictureweb", shape=model.shape, shard_sizes=[4096,1000])
    print("Sharding model to {0} in bucket {1}".format(bigm.key, bigm.bucket))
    shard_matrix(bigm, model, n_jobs=64)
    e = time.time()
    print("model shard complete {0}".format(e - t))


