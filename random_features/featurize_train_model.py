import argparse
import time
import multiprocessing as mp
import numpy as np
import scipy.linalg
import torch

import coatesng
import utils

MAX_THREADS = 16
TOT_PATCHES = int(1e5)

def build_featurizer(patch_size, pool_size, pool_stride, bias, patch_distribution, num_filters, num_channels, seed, filter_scale, X_train=None, filter_batch_size=2048):
    dtype = 'float32'
    if (patch_distribution == 'empirical'):
        assert X_train is not None, "X_train must be provided when patch distribution == empirical"
        all_patches, idxs = utils.grab_patches(
            X_train, patch_size=patch_size, max_threads=MAX_THREADS, seed=seed, tot_patches=TOT_PATCHES)
        all_patches = utils.normalize_patches(
            all_patches, zca_bias=filter_scale)
        idxs = np.random.choice(
            all_patches.shape[0], num_filters, replace=False)
        filters = all_patches[idxs].astype(dtype)
        print("filters shape", filters.shape)
    elif (patch_distribution == 'gaussian'):
        filters = np.random.randn(
            num_filters, num_channels, patch_size, patch_size).astype(dtype) * filter_scale
        print("filters shape", filters.shape)
    elif (patch_distribution == 'laplace'):
        filters = np.random.laplace(loc=0.0, scale=filter_scale, size=(
            num_filters*num_channels*patch_size*patch_size)).reshape(num_filters, num_channels, patch_size, patch_size)
        filters = filters.astype('float32')
        print("filters shape", filters.shape)
    else:
        raise Exception(
            f"Unsupported patch distribution : {patch_distribution}")
    net = coatesng.BasicCoatesNgNet(filters, pool_size=pool_size, pool_stride=pool_stride,
                                    bias=bias, patch_size=patch_size, filter_batch_size=filter_batch_size)
    return net


def featurize(dataset, patch_size, patch_distribution, num_filters, pool_size, pool_stride, bias, filter_scale, seed, data_batchsize=1024, filter_batch_size=1024, zca=False):
    data = utils.load_dataset(dataset)
    gpu = torch.cuda.is_available()
    num_channels = 3
    dtype = 'float32'
    X_train = data['X_train']
    X_test = data['X_test']
    if zca:
        X_train, X_test = utils.preprocess(X_train, X_test)
    y_train = data['y_train']
    y_test = data['y_test']
    if ("cifar" in dataset):
        dataset = "cifar-10"
    elif ("mnist" in dataset):
        dataset = "mnist"
    else:
        raise Exception("unsupported dataset")
    featurizer = build_featurizer(patch_size, pool_size, pool_stride, bias, patch_distribution,
                                  num_filters, num_channels, seed, filter_scale, X_train, filter_batch_size)
    start = time.time()
    if not zca:
        assert X_train.dtype == np.uint8
        print("Scaling input by 255..")
        X_train = X_train.astype(dtype)
        X_test = X_test.astype(dtype)
        X_train /= 255.0
        X_test /= 255.0
    X_train_lift = coatesng.coatesng_featurize(
        featurizer, X_train.astype(dtype), data_batchsize=data_batchsize, gpu=gpu)
    X_test_lift = coatesng.coatesng_featurize(
        featurizer, X_test.astype(dtype), data_batchsize=data_batchsize, gpu=gpu)
    end = time.time()
    featurizer = featurizer.cpu()
    print(
        f"featurization complete, featurized {X_train.shape[0]} training points  and {X_test.shape[0]} testing points, {X_train_lift.shape[1]} output features, took {end - start} seconds")
    return X_train_lift, X_test_lift, y_train, y_test, featurizer.cpu()

def train_ls_model(X_train, y_train, reg):
    X_train = X_train.astype('float64')
    y = np.eye(np.max(y_train) + 1)[y_train]
    XTX = X_train.T.dot(X_train)
    XTy = X_train.T.dot(y)
    idxs = np.diag_indices(X_train.shape[1])
    XTX[idxs] += reg
    model = scipy.linalg.solve(XTX, XTy)
    XTX[idxs] -= reg
    return model

def train_ls_model_unsupervised(X_train, y_train, reg, n_train=4096):
    X_train = X_train.astype('float64')
    y = np.eye(np.max(y_train) + 1)[y_train]


    XTX = X_train.T.dot(X_train)
    XTX /= X_train.shape[0]

    X_train = X_train[:n_train]
    y = y[:n_train]

    XTy = X_train.T.dot(y)
    XTy /= n_train

    idxs = np.diag_indices(X_train.shape[1])
    XTX[idxs] += reg
    model = scipy.linalg.solve(XTX, XTy)
    XTX[idxs] -= reg
    return model

def eval_ls_model(model, X, y):
    y_pred = X.dot(model)
    return np.sum(np.argmax(y_pred, axis=1) == y)/y.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'generate a convolutional random features model')
    parser.add_argument('--num_filters', default=16, type=int)
    parser.add_argument('--dataset', help="cifar-10 or mnist (default cifar-10)", default='cifar-10')
    parser.add_argument('--patch_size', type=int, default=6)
    parser.add_argument('--subset', type=int, default=50000)
    parser.add_argument('--pool_size', type=int, default=15)
    parser.add_argument('--pool_stride', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bias', type=float, default=1.0)
    parser.add_argument('--filter_scale', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patch_distribution', type=str, default='empirical')
    parser.add_argument('--regularizer', type=float, default=1e-4)
    parser.add_argument('--unsupervised', const=True, action="store_const")
    parser.add_argument('--zca', const=True, action="store_const")
    args = parser.parse_args()

    X_train_lift, X_test_lift, y_train, y_test, featurizer = featurize(args.dataset, args.patch_size, args.patch_distribution, args.num_filters, args.pool_size, args.pool_stride, args.bias, args.filter_scale, args.seed, data_batch_size=args.batch_size, zca=args.zca)

    X_train_lift = X_train_lift[:args.subset]
    y_train = y_train[:args.subset]

    for reg in [1e-8, 1e-4, 1e-2,1,10,100,1000,10000]:
        print("regularization: ", reg)
        if args.unsupervised:
            model = train_ls_model_unsupervised(X_train_lift, y_train, reg)
        else:
            model = train_ls_model(X_train_lift, y_train, reg)
        train_acc = eval_ls_model(model, X_train_lift, y_train)
        test_acc = eval_ls_model(model, X_test_lift, y_test)
        print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
