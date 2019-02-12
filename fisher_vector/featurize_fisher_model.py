import numpywren.matrix_utils as utils
import numpywren.matrix as matrix
import fisher
import argparse
import numpy as np
import boto3
import scipy.misc
import sift
import lcs
import time
import pywren
from sklearn.mixture import GaussianMixture
import io
import scipy.linalg
import sklearn.metrics as metrics
from numba import jit
import concurrent.futures as fs
from native_gmm import compute_gmm
from numpy import ascontiguousarray as C_ARRAY
import pickle

SIFT_DESC_LENGTH = 131
LCS_DESC_LENGTH = 98

def chunk_idxs(size, chunks):
    chunk_size  = int(np.ceil(size/chunks))
    idxs = list(range(0, size+1, chunk_size))
    if (idxs[-1] != size):
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))

def calculate_sifts(img_keys, out_matrix, block_idx, descs_per_img=16):
    sifts = []
    s3 = boto3.resource('s3')
    import time
    t = time.time()
    np.random.seed(block_idx)
    for im_key in img_keys:
        s3.Bucket("pictureweb").download_file(im_key, "/tmp/img.jpg")
        im = scipy.misc.imread("/tmp/img.jpg", flatten=True)
        im = im.astype('float32')
        im /= 255.0
        descs = sift.sift(im)
        assert(np.all(descs >= 0))
        idxs = np.random.choice(descs.shape[0], descs_per_img)
        descs = np.sqrt(descs)
        sifts.append(descs[idxs, :])
    out_matrix.put_block(np.vstack(sifts), block_idx, 0)
    e = time.time()
    return e - t

def calculate_lcs(img_keys, out_matrix, block_idx, descs_per_img=16):
    feats = []
    s3 = boto3.resource('s3')
    import time
    t = time.time()
    np.random.seed(block_idx)
    for im_key in img_keys:
        s3.Bucket("pictureweb").download_file(im_key, "/tmp/img.jpg")
        im = scipy.misc.imread("/tmp/img.jpg")
        im = im.astype('float32')
        descs = lcs.lcs(im).reshape(-1, LCS_DESC_LENGTH)
        idxs = np.random.choice(descs.shape[0], descs_per_img)
        feats.append(descs[idxs, :])
    out_matrix.put_block(np.vstack(feats), block_idx, 0)
    e = time.time()
    return e - t


def compute_pca_matrix(X, truncate=64):
    X = X.reshape(X.shape[0], -1)
    X = X.astype('float64')
    if (truncate == None):
        truncate = X.shape[1]
    mu = np.mean(X, axis=0)
    X = X - mu
    XTX = X.T.dot(X)
    vals, vecs = np.linalg.eigh(XTX)
    pca = np.ascontiguousarray(vecs[:, ::-1])
    vals = np.ascontiguousarray(vals[::-1])
    return pca[:, :truncate], mu

def image_fisher_featurize_sift(im_keys, out_matrix, bidx, gmm_sift, pca_sift_mat, pca_sift_mu):
    features =  []
    t = time.time()
    for im_key in im_keys:
        s3 = boto3.resource('s3')
        client = boto3.client('s3')
        bio = io.BytesIO(client.get_object(Bucket="pictureweb", Key=im_key)["Body"].read())
        im = scipy.misc.imread(bio, flatten=True)
        im = im.astype('float32')
        im /= 255.0
        descs = sift.sift(im)
        descs = (descs).dot(pca_sift_mat)
        sift_features = fisher.fisher_vector_features(descs.astype('float32'), *gmm_sift)
        features.append(sift_features)
    #sqrt normalization
    signs = np.sign(features)
    features = signs * np.sqrt(np.abs(features))

    feature_norms = np.linalg.norm(features, axis=1)[:, np.newaxis]
    features /= feature_norms

    out_matrix.put_block(features, bidx, 0)
    e = time.time()
    return e - t

def image_fisher_featurize_sift_lcs(im_keys, out_matrix, bidx, gmm_sift, pca_sift_mat,gmm_lcs, pca_lcs_mat):
    all_sift_features =  []
    all_lcs_features =  []
    t = time.time()
    for im_key in im_keys:
        s3 = boto3.resource('s3')
        client = boto3.client('s3')
        bio = io.BytesIO(client.get_object(Bucket="pictureweb", Key=im_key)["Body"].read())
        im = scipy.misc.imread(bio, flatten=True)
        im = im.astype('float32')
        im /= 255.0
        sift_descs = sift.sift(im)
        assert(np.all(sift_descs >= 0))
        sift_descs = np.sqrt(sift_descs)
        sift_descs = (sift_descs).dot(pca_sift_mat)
        sift_features = fisher.fisher_vector_features(sift_descs.astype('float32'), *gmm_sift)
        sift_features /= np.linalg.norm(sift_features)

        bio = io.BytesIO(client.get_object(Bucket="pictureweb", Key=im_key)["Body"].read())
        im = scipy.misc.imread(bio)
        lcs_descs = lcs.lcs(im).reshape(-1, LCS_DESC_LENGTH)
        try:
            assert(np.any(np.isnan(lcs_descs)) == False)
        except:
            raise Exception("RAISING LCS Error pre pca in {0}".format(im_key))

        lcs_descs = (lcs_descs).dot(pca_lcs_mat)
        try:
            assert(np.any(np.isnan(lcs_descs)) == False)
        except:
            raise Exception("RAISING LCS Error post pca in {0}".format(im_key))


        lcs_features = fisher.fisher_vector_features(lcs_descs.astype('float32'), *gmm_lcs)
        lcs_features /= np.linalg.norm(lcs_features)
        try:
            assert(np.any(np.isnan(lcs_features)) == False)
        except:
            raise Exception("RAISING LCS Fisher Vector Error in {0}".format(im_key))

        all_sift_features.append(sift_features)
        all_lcs_features.append(lcs_features)

    all_sift_features = np.array(all_sift_features)
    all_lcs_features = np.array(all_lcs_features)

    assert(np.any(np.isnan(all_sift_features)) == False)
    #sqrt normalization
    signs = np.sign(all_sift_features)
    all_sift_features = signs * np.sqrt(np.abs(all_sift_features))
    feature_norms = np.linalg.norm(all_sift_features, axis=1)[:, np.newaxis]
    assert(np.any(np.isnan(feature_norms)) == False)
    all_sift_features /= feature_norms
    assert(np.any(np.isnan(all_sift_features)) == False)


    assert(np.any(np.isnan(all_lcs_features)) == False)
    signs = np.sign(all_lcs_features)
    all_lcs_features = signs * np.sqrt(np.abs(all_lcs_features))
    feature_norms = np.linalg.norm(all_lcs_features, axis=1)[:, np.newaxis]
    assert(np.any(np.isnan(feature_norms)) == False)
    all_lcs_features /= feature_norms
    assert(np.any(np.isnan(all_lcs_features)) == False)

    features = np.hstack((all_sift_features, all_lcs_features))
    out_matrix.put_block(features, bidx, 0)
    e = time.time()
    return e - t, t, e

def top_k_accuracy(labels, y_pred, k=5):
    top_k_preds = get_top_k(y_pred, k=k)
    if (len(labels.shape) == 1):
        labels = labels[:, np.newaxis]
    correct = np.sum(np.any(top_k_preds == labels, axis=1))
    return correct/float(labels.shape[0])

def get_top_k(y_pred, k=5, threads=70):
    with fs.ThreadPoolExecutor(max_workers=threads) as executor:
        idxs = chunk_idxs(y_pred.shape[0], threads)
        futures = []
        for (sidx, eidx) in idxs:
            futures.append(executor.submit(_get_top_k, y_pred[sidx:eidx, :], k))
        fs.wait(futures)
        results = np.vstack(list(map(lambda x: x.result(), futures)))
    return results

@jit(nopython=True, nogil=True)
def _get_top_k(y_pred, k=5):
    top_k_preds = np.ones((y_pred.shape[0], k))
    top_k_pred_weights = np.ones((y_pred.shape[0], k))

    top_k_pred_weights *= -99999999
    for i in range(y_pred.shape[0]):
        top_k = top_k_preds[i, :]
        top_k_pred_weights_curr = top_k_pred_weights[i, :]
        for j in range(y_pred.shape[1]):
            in_top_k = False
            for elem in top_k_pred_weights_curr:
                in_top_k = in_top_k | (y_pred[i,j] > elem)
            if (in_top_k):
                min_idx = 0
                for z in range(top_k_pred_weights_curr.shape[0]):
                    if top_k_pred_weights_curr[min_idx] > top_k_pred_weights_curr[z]:
                        min_idx = z
                top_k[min_idx] = j
                top_k_pred_weights_curr[min_idx] = y_pred[i,j]

    return top_k_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make some imagenet features')
    parser.add_argument('--train_keys', default="./imagenet_train_keys")
    parser.add_argument('--validation_keys', default="./imagenet_validation_keys")
    parser.add_argument('--random_seed', default=0)
    parser.add_argument('--chunk_size', default=32, type=int)
    parser.add_argument('--num_centers', default=16, type=int)
    parser.add_argument('--pca_dim', default=64)
    parser.add_argument('--pca_sample_images', default=int(1e5))
    parser.add_argument('--pca_sample_descs_per_image', default=int(16))
    parser.add_argument('--use_cache_gmm_sift', const=True, action='store_const', default=False)
    parser.add_argument('--use_cache_gmm_lcs', const=True, action='store_const', default=False)
    args = parser.parse_args()
    get_cls = lambda x: int(x.split("/")[1])
    train_keys = sorted([x.strip() for x in open(args.train_keys).readlines()], key=get_cls)
    test_keys = sorted([x.strip() for x in open(args.validation_keys).readlines()], key=get_cls)

    #train_keys = np.array(train_keys)[:1300*10]
    #test_keys = np.array(test_keys)[:50*10]

    train_keys = np.array(train_keys)
    test_keys = np.array(test_keys)

    np.random.seed(args.random_seed)
    np.random.shuffle(train_keys)
    np.random.shuffle(test_keys)

    N_train = len(train_keys)
    N_test = len(test_keys)

    print("{0} train keys".format(N_train))
    print("{0} test keys".format(N_test))

    train_chunks = list(utils.chunk(train_keys, args.chunk_size))
    test_chunks = list(utils.chunk(test_keys, args.chunk_size))


    print("{0} train chunks".format(len(train_chunks)))
    print("{0} test chunks".format(len(test_chunks)))

    # calculate number of output features
    num_features = 4 * args.pca_dim * args.num_centers
    #num_features = 2 * args.pca_dim * args.num_centers
    print("Num features ", num_features)
    # sample descriptors to learn PCA mapping and GMM
    train_keys = np.array(train_keys)
    pca_sample_images = min(args.pca_sample_images, N_train)
    idxs_sample = np.random.choice(N_train, pca_sample_images, replace=False)
    pca_dpi = args.pca_sample_descs_per_image
    num_sample_descs = pca_dpi * pca_sample_images
    sifts_hash = utils.hash_string(utils.hash_args((train_keys, args.pca_dim, pca_sample_images, args.pca_sample_descs_per_image, args.random_seed, args.pca_dim)) + utils.hash_function(calculate_sifts) + utils.hash_function(sift.sift))
    lcs_hash = utils.hash_string(utils.hash_args((train_keys, args.pca_dim, pca_sample_images, args.pca_sample_descs_per_image, args.random_seed, args.pca_dim)) + utils.hash_function(calculate_lcs) + utils.hash_function(lcs.lcs))

    sift_sample_descs = matrix.BigMatrix(sifts_hash, shape=(num_sample_descs, SIFT_DESC_LENGTH), shard_sizes=(pca_dpi**2, SIFT_DESC_LENGTH), write_header=True)

    lcs_sample_descs = matrix.BigMatrix(lcs_hash, shape=(num_sample_descs, LCS_DESC_LENGTH), shard_sizes=(pca_dpi**2, LCS_DESC_LENGTH), write_header=True)

    block_idxs_not_exist = sift_sample_descs.block_idxs_not_exist
    print("Sample Descs Blocks not exist", len(block_idxs_not_exist))
    print("Sample Descs Blocks total", len(sift_sample_descs.block_idxs))
    pca_sample_train_keys = train_keys[idxs_sample]
    chunked_train_keys = list(utils.chunk(pca_sample_train_keys, pca_dpi))
    pwex = pywren.default_executor()
    t = time.time()
    print("Calculating SIFT samples...")
    # only map over the blocks that don't exist
    blocks_to_map = []
    for i,b in enumerate(chunked_train_keys):
        if ((i,0) in block_idxs_not_exist):
            blocks_to_map.append((i,b))
    print("{0} blocks to map".format(len(blocks_to_map)))
    pywren_map = lambda x: calculate_sifts(x[1], sift_sample_descs, x[0], descs_per_img=pca_dpi)
    futures = pwex.map(pywren_map, blocks_to_map)
    pywren.wait(futures)
    [f.result() for f in futures]
    e = time.time()
    print("{0} seconds to calculate sifts".format(e -t))
    t = time.time()
    sifts_sample_local = sift_sample_descs.numpy(workers=128)
    e = time.time()
    print(e - t, "seconds to download sift")
    print(sifts_sample_local.shape)

    block_idxs_not_exist = lcs_sample_descs.block_idxs_not_exist
    print("Sample Descs Blocks not exist", len(block_idxs_not_exist))
    print("Sample Descs Blocks total", len(lcs_sample_descs.block_idxs))
    pca_sample_train_keys = train_keys[idxs_sample]
    chunked_train_keys = list(utils.chunk(pca_sample_train_keys, pca_dpi))
    pwex = pywren.default_executor()
    t = time.time()
    print("Calculating lcs samples...")
    # only map over the blocks that don't exist
    blocks_to_map = []
    for i,b in enumerate(chunked_train_keys):
        if ((i,0) in block_idxs_not_exist):
            blocks_to_map.append((i,b))
    print("{0} blocks to map".format(len(blocks_to_map)))
    futures = pwex.map(lambda x: calculate_lcs(x[1], lcs_sample_descs, x[0], descs_per_img=pca_dpi), blocks_to_map)
    pywren.wait(futures)
    [f.result() for f in futures]
    e = time.time()
    print("{0} seconds to calculate lcs".format(e -t))
    t = time.time()
    lcs_sample_local = lcs_sample_descs.numpy(workers=128)
    e = time.time()
    print(e - t, "seconds to download lcs")
    print(lcs_sample_local.shape)

    print("Learning PCA matrix sift")
    if (not args.use_cache_gmm_sift):
        pca_mat_sift, mu_sift = compute_pca_matrix(sifts_sample_local, truncate=args.pca_dim)
        sifts_sample_local_pca = (sifts_sample_local).dot(pca_mat_sift)
        np.save('pca_mat_sift', pca_mat_sift)
        np.save('pca_mu_sift', mu_sift)
    else:
        pca_mat_sift = np.load('pca_mat_sift.npy')
        mu = np.load('pca_mu_sift.npy')
        sifts_sample_local_pca = (sifts_sample_local).dot(pca_mat_sift)

    if (not args.use_cache_gmm_lcs):
        pca_mat_lcs, mu_lcs = compute_pca_matrix(lcs_sample_local, truncate=args.pca_dim)
        lcs_sample_local_pca = (lcs_sample_local).dot(pca_mat_lcs)
        np.save('pca_mat_lcs', pca_mat_lcs)
        np.save('pca_mu_lcs', mu_lcs)
    else:
        pca_mat_lcs = np.load('pca_mat_lcs.npy')
        mu_lcs = np.load('pca_mu_lcs.npy')
        lcs_sample_local_pca = (lcs_sample_local).dot(pca_mat_lcs)


    idxs = np.random.choice(sifts_sample_local_pca.shape[0], int(1e5), replace=False)
    if (not args.use_cache_gmm_sift):
        gmm = GaussianMixture(n_components=args.num_centers, covariance_type='diag', verbose=3, init_params="kmeans", max_iter=400, random_state=args.random_seed)
        print("fitting GMM SIFT")
        gmm.fit(sifts_sample_local_pca[idxs, :])
        weights = gmm.weights_
        means = gmm.means_
        covars = gmm.covariances_
        gmm_sift = (means, covars, weights)
        np.save('gmm_sift_weights_{0}_centers'.format(args.num_centers), weights)
        np.save('gmm_sift_means_{0}_centers'.format(args.num_centers), means)
        np.save('gmm_sift_covars_{0}_centers'.format(args.num_centers), covars)
    else:
        pwex = pywren.default_executor()
        pca_mat_sift = np.load("pca_mat_sift.npy")
        mu = np.load("pca_mu_sift.npy")
        weights = np.load('gmm_sift_weights_{0}_centers.npy'.format(args.num_centers))
        means = np.load('gmm_sift_means_{0}_centers.npy'.format(args.num_centers))
        covars= np.load('gmm_sift_covars_{0}_centers.npy'.format(args.num_centers))
        gmm_sift = (means, covars, weights)


    if (not args.use_cache_gmm_lcs):
        gmm = GaussianMixture(n_components=args.num_centers, covariance_type='diag', verbose=3, init_params="kmeans", max_iter=400, random_state=args.random_seed)
        print("fitting GMM lcs")
        gmm.fit(lcs_sample_local_pca[idxs, :])
        weights = gmm.weights_
        means = gmm.means_
        covars = gmm.covariances_
        gmm_lcs = (means, covars, weights)
        np.save('gmm_lcs_weights_{0}_centers'.format(args.num_centers), weights)
        np.save('gmm_lcs_means_{0}_centers'.format(args.num_centers), means)
        np.save('gmm_lcs_covars_{0}_centers'.format(args.num_centers), covars)
    else:
        pca_mat_lcs = np.load("pca_mat_lcs.npy")
        mu = np.load("pca_mu_lcs.npy")
        weights = np.load('gmm_lcs_weights_{0}_centers.npy'.format(args.num_centers))
        means = np.load('gmm_lcs_means_{0}_centers.npy'.format(args.num_centers))
        covars= np.load('gmm_lcs_covars_{0}_centers.npy'.format(args.num_centers))
        gmm_lcs = (means, covars, weights)



    train_hash = utils.hash_string(utils.hash_args((train_keys, args.pca_dim, args.pca_sample_images, args.pca_sample_descs_per_image, args.random_seed, args.pca_dim, args.num_centers, args.chunk_size, sifts_hash, gmm_sift, gmm_lcs, pca_mat_sift, pca_mat_lcs)) + utils.hash_function(image_fisher_featurize_sift_lcs) + utils.hash_function(sift.sift) + utils.hash_function(lcs.lcs) + utils.hash_function(fisher.fisher_vector_features))
    test_hash = utils.hash_string(utils.hash_args((test_keys, args.pca_dim, args.pca_sample_images, args.pca_sample_descs_per_image, args.random_seed, args.pca_dim, args.num_centers, args.chunk_size, sifts_hash, gmm_sift, gmm_lcs, pca_mat_sift, pca_mat_lcs)) + utils.hash_function(image_fisher_featurize_sift_lcs) + utils.hash_function(sift.sift) + utils.hash_function(lcs.lcs) + utils.hash_function(fisher.fisher_vector_features))

    # allocate numpywren output matrix
    print("Train hash", train_hash)
    print("Test hash", test_hash)
    imagenet_train_featurized = matrix.BigMatrix(train_hash, shape=(N_train, num_features), shard_sizes=[args.chunk_size, num_features], write_header=True)
    imagenet_validation_featurized = matrix.BigMatrix(test_hash, shape=(N_test, num_features), shard_sizes=[args.chunk_size, num_features], write_header=True)


    print(imagenet_train_featurized.shape)


    #image_fisher_featurize_sift(train_chunks[0], imagenet_train_featurized, 0, (gmm.means_, gmm.covariances_, gmm.weights_), pca_mat, mu)
    # fisher vector featurize
    print("Total train blocks to featurize ", len(imagenet_train_featurized.block_idxs))
    print("Train Blocks left to featurize ", len(imagenet_train_featurized.block_idxs_not_exist))
    blocks_to_map = [x[0] for x in imagenet_train_featurized.block_idxs_not_exist]
    t = time.time()
    print("Mapping train...")
    futures_train = pwex.map(lambda x: image_fisher_featurize_sift_lcs(train_chunks[x], imagenet_train_featurized, x, gmm_sift, pca_mat_sift, gmm_lcs, pca_mat_lcs), blocks_to_map)
    print("Total test blocks to featurize ", len(imagenet_validation_featurized.block_idxs))
    print("Test Blocks left to featurize ", len(imagenet_validation_featurized.block_idxs_not_exist))
    blocks_to_map = [x[0] for x in imagenet_validation_featurized.block_idxs_not_exist]

    print("Mapping validation...")
    futures_test = pwex.map(lambda x: image_fisher_featurize_sift_lcs(test_chunks[x], imagenet_validation_featurized, x, gmm_sift, pca_mat_sift, gmm_lcs, pca_mat_lcs), blocks_to_map)


    print("Waiting for train...")
    pywren.wait(futures_train)
    print("Waiting for validation...")
    pywren.wait(futures_test)
    e = time.time()
    print("Total end to end job time", e - t)
    print("Average train job time ", np.mean([f.result()[0] for f in futures_train]))
    print("Average test job time ", np.mean([f.result()[0] for f in futures_test]))

    print("saving train times")

    np.save("train_times", np.array([f.result()[1:] for f in futures_train]))

    imagenet_train_featurized.dtype = "float32"
    imagenet_validation_featurized.dtype = "float32"

    X_train = imagenet_train_featurized.numpy(workers=256)
    X_test = imagenet_validation_featurized.numpy(workers=256)

    y_train = np.array([int(x.split("/")[1]) for x in train_keys])
    y_test = np.array([int(x.split("/")[1]) for x in test_keys])
    features = {}
    features['gmm_sift'] = gmm_sift
    features['pca_mat_sift'] = pca_mat_sift
    features['gmm_lcs'] = gmm_lcs
    features['pca_mat_lcs'] = pca_mat_lcs
    features['X_train'] = X_train
    features['X_test'] = X_test
    features['y_train'] = y_train
    features['y_test'] = y_test
    print('serializing features...')
    data = pickle.dumps(features, protocol=4)
    print('saving features...')
    with open('fishervector_features.pickle', 'wb+') as f:
        f.write(data)

    print("Num Classes train {0}".format(len(list(set(list(y_train))))))
    print("Num Classes test {0}".format(len(list(set(list(y_test))))))

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    y_train_enc = np.eye(1000)[y_train]
    XTX = X_train.T.dot(X_train)
    XTy = X_train.T.dot(y_train_enc)
    num_features = XTX.shape[0]
    #np.save("XTX_64k", XTX)
    #np.save("XTy_64k", XTX)
    for reg in [1e-8,1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        model = scipy.linalg.solve(XTX + reg*np.eye(num_features), XTy, sym_pos=True)
        print("Regularization ", reg)
        y_train_pred = X_train.dot(model)
        y_test_pred = X_test.dot(model)

        print("Top 1 Train acc ", top_k_accuracy(y_train, y_train_pred, k=1))
        print("Top 1 Test acc ", top_k_accuracy(y_test, y_test_pred, k=1))
        print("Top 5 Train acc ", top_k_accuracy(y_train, y_train_pred, k=5))
        print("Top 5 Test acc ", top_k_accuracy(y_test, y_test_pred, k=5))













