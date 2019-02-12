from numpywren.matrix import BigMatrix, BigSymmetricMatrix
import numpywren.matrix_utils as utils 
import imagenet_featurize_fv as fv
import bcd
import argparse
import pywren
import time
import boto3
import io
import numpy as np
import os
import concurrent.futures as fs
import subprocess
from subprocess import Popen
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

'''
How to run:
python solve.py "rbf(XXT(X_block_scrambled_3_3_pool_12_12_patch_stride_1_1_blocks_no_normalize_04.22.2017, 3)" "scrambled_train_labels.npy"
'''


def solve_fn(train_key, train_labels_key, test_key, test_labels_key, bucket, lambdav, blocks_per_iter, epochs, num_classes, eval_interval, start_block, start_epoch, warm_start, prev_yhat, num_test_blocks, sheet):
    client = boto3.client('s3')
    resp = client.get_object(Key=train_labels_key, Bucket=bucket)
    bio = io.BytesIO(resp["Body"].read())
    y_train = np.load(bio).astype('int')
    y_train_enc = np.eye(num_classes)[y_train]
    N_train = y_train.shape[0]
    K_train = BigSymmetricMatrix(train_key, bucket=bucket)
    if (test_key != None and test_labels_key != None):
        resp = client.get_object(Key=test_labels_key, Bucket=bucket)
        bio = io.BytesIO(resp["Body"].read())
        y_test = np.load(bio)
        N_test = y_test.shape[0]

        K_test = BigMatrix(test_key, bucket=bucket)
        if (num_test_blocks == None):
            num_test_blocks = len(K_test._block_idxs(1))
        test_size = min(num_test_blocks*K_test.shard_sizes[1], N_test)
        print("Test Size", test_size)
    else:
        K_test = None
        y_test = None

    print(K_test)
    print(y_test)
    def eval_fn(model, y_hat, y, lambdav, block, epoch, iter_time):
        print("Evaluating and saving result to s3")
        print(model.dtype)

        row_blocks = K_test._block_idxs(0)
        if (not (K_test is None) and not (y_test is None)):
            if ((block == 0 and epoch == 0) or (not os.path.exists("/dev/shm/K_test_block"))):
                futures = utils.get_matrix_blocks_full_async(K_test, "/dev/shm/K_test_block", row_blocks, K_test._block_idxs(1)[:num_test_blocks], big_axis=0)
                print("Downloading test block")
                fs.wait(futures)
            K_test_block = utils.load_mmap("/dev/shm/K_test_block", (K_test.shape[0], test_size), "float64")
            model = model.astype('float64')
            print("K_Test BLOCK SHAPE ", K_test_block.shape)
            y_test_pred = K_test_block.T.dot(model)
            print(y_test_pred.shape)
            test_top1 = fv.top_k_accuracy(y_test[:test_size], y_test_pred, k=1)
            test_top5 = fv.top_k_accuracy(y_test[:test_size], y_test_pred, k=5)
            print("Top 1 test {0}, Top 5 test {1}".format(test_top1, test_top5))
        else:
            test_top1 = "NA"
            test_top5 = "NA"

        print("Calculate objective value")
        tr_wTKw = np.sum(model * y_hat)
        w_2 = np.linalg.norm(model)
        tr_yw = np.sum(model * y)
        obj = 0.5*tr_wTKw + 0.5*lambdav*((w_2)**2) - tr_yw
        print("Objective value {0}".format(obj))


        print("Calculating top 5 and top 1 accuracies")
        train_top1 = fv.top_k_accuracy(y_train, y_hat, k=1)
        train_top5 = fv.top_k_accuracy(y_train, y_hat, k=5)
        print("Top 1 train {0}, Top 5 train {1}".format(train_top1, train_top5))
        matrix_name = K_train.key
        client = boto3.client('s3')


        print("Updating sheet")
        # update google sheet
        scope = ['https://spreadsheets.google.com/feeds']
        credentials = ServiceAccountCredentials.from_json_keyfile_name('/tmp/gspread_onlyshallow.json', scope)
        gc = gspread.authorize(credentials)

        # Hard coded
        bcd_sheet = gc.open("OnlyShallowImagenet").worksheet(sheet)
        bcd_sheet.append_row([matrix_name, lambdav, epoch, block, blocks_per_iter, obj, train_top1, test_top1, train_top5, test_top5, iter_time])

        np.save("/tmp/kernel_yhat_test", y_test_pred)

        return 0

    if (prev_yhat != None):
        prev_yhat = np.load(prev_yhat)

    if (warm_start != None):
        warm_start = np.load(warm_start)

    bcd.block_kernel_solve(K_train, y_train_enc, epochs=epochs, lambdav=lambdav, blocks_per_iter=blocks_per_iter, eval_fn=eval_fn, eval_interval=eval_interval, start_block=start_block, start_epoch=start_epoch, y_hat=prev_yhat, warm_start=warm_start, num_blocks=len(K_train._blocks(0)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve ')
    parser.add_argument('--train_key', type=str, help="S3 Key to sharded train matrix", default="gemm(BigMatrix(imagenet_train_2m_fishervector), BigMatrix(imagenet_train_2m_fishervector).T)")
    parser.add_argument('--train_labels_key', type=str, help="S3 Key to sharded train matrix", default="y_train_fishervector.npy")
    parser.add_argument('--test_key', type=str, help="S3 Key to sharded test matrix", default="gemm(BigMatrix(imagenet_train_2m_fishervector), BigMatrix(imagenet_test_2m_fishervector).T)")
    parser.add_argument('--test_labels_key', type=str, help="S3 Key to sharded test matrix", default="y_test_fishervector.npy")
    parser.add_argument('--bucket', type=str, help="S3 bucket where sharded matrices live", default="pictureweb")
    parser.add_argument('--lambdav', type=float, help="regularization value", default=1e-5)
    parser.add_argument('--blocks_per_iter', type=int, help="regularization value", default=15)
    parser.add_argument('--epochs', type=int, help="regularization value", default=1)
    parser.add_argument('--start_epoch', type=int, help="start epoch", default=0)
    parser.add_argument('--start_block', type=int, help="start block", default=0)
    parser.add_argument('--num_classes', type=str, help="regularization value", default=1000)
    parser.add_argument('--warm_start', type=str, help="path to warm start numpy array", default=None)
    parser.add_argument('--prev_yhat', type=str, help="path to a previous y_hat", default=None)
    parser.add_argument('--eval_interval', type=int, help="how often do you want to evaluate and update s3 with results", default=1)
    parser.add_argument('--num_test_blocks', type=int, help="how many blocks to test", default=1)
    parser.add_argument('--sheet', type=str, help="which sheet to update", default="bcd")
    args = parser.parse_args()
    solve_fn(args.train_key, args.train_labels_key, args.test_key, args.test_labels_key, args.bucket, args.lambdav, args.blocks_per_iter, args.epochs, args.num_classes, args.eval_interval, args.start_block, args.start_epoch, args.warm_start, args.prev_yhat, args.num_test_blocks, args.sheet)






