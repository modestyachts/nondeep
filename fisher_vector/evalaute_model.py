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
import imagenet_featurize_fv as fv
import pywren
import pywren.wrenconfig as wc
from numpywren import matrix, binops
from numpywren.binops import gemm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shard dual model.')
    t = time.time()
    parser.add_argument('model_key', type=str, help="model_path")
    parser.add_argument('train_key', type=str, help="train_key")
    parser.add_argument('test_key', type=str, help="train_key")
    parser.add_argument('--train_labels', type=str, help="train_labels", default="y_train_fishervector.npy")
    parser.add_argument('--test_labels', type=str, help="test_labels", default="y_test_fishervector.npy")
    args = parser.parse_args()

    y_train = np.load(args.train_labels)
    y_test = np.load(args.test_labels)

    K_train = matrix.BigSymmetricMatrix(args.train_key, bucket="pictureweb")
    K_test = matrix.BigMatrix(args.test_key, bucket="pictureweb")
    model = matrix.BigMatrix(args.model_key, bucket="pictureweb", shape=(K_train.shape[0], int(np.max(y_train)+1)), shard_sizes=(4096, 1000), write_header=True)

    config = wc.default()
    config['runtime']['s3_bucket'] = 'pictureweb'
    config['runtime']['s3_key'] = 'pywren.runtime/pywren_runtime-3.6-pictureweb.tar.gz'
    config['standalone']['sqs_queue_name'] = 'pictureweb'
    print("please launch some standalone instances for this script....")
    pwex = pywren.standalone_executor(config=config)
    print("Evaluating Train")
    t = time.time()
    y_train_pred  = gemm(pwex, K_train, model, overwrite=False, tasks_per_job=1, gemm_impl=2)
    e = time.time()
    print("Train Eval took {0}".format(e -t))
    print("Downloading train")
    y_train_pred_local = y_train_pred.numpy()
    train_top1 = fv.top_k_accuracy(y_train, y_train_pred_local, k=1)
    train_top5 = fv.top_k_accuracy(y_train, y_train_pred_local, k=5)
    print("Train top 5 accuracy {0}, Train top 1 accuracy {1}".format(train_top1, train_top5))



    print("Evaluating Test")
    t = time.time()
    y_test_pred  = gemm(pwex, K_test.T, model, overwrite=False, tasks_per_job=1, gemm_impl=2)
    e = time.time()
    print("Test Eval took {0}".format(e -t))
    print("Downloading test")
    y_test_pred_local = y_test_pred.numpy()
    test_top1 = fv.top_k_accuracy(y_test, y_test_pred_local, k=1)
    test_top5 = fv.top_k_accuracy(y_test, y_test_pred_local, k=5)
    print("test top 5 accuracy {0}, test top 1 accuracy {1}".format(test_top1, test_top5))



