import argparse
import pickle
from numba import jit
import numpy as np
import scipy.linalg
import concurrent.futures as fs

def chunk_idxs(size, chunks):
    chunk_size  = int(np.ceil(size/chunks))
    idxs = list(range(0, size+1, chunk_size))
    if (idxs[-1] != size):
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))


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
    parser = argparse.ArgumentParser(description='train fisher model.')
    parser.add_argument('features_path', type=str, help="feature_path")
    parser.add_argument('out_name', type=str, help="fisher_out_path")
    args = parser.parse_args()
    with open(args.features_path, "rb") as f:
        data = pickle.loads(f.read())
    gmm_sift = data['gmm_sift']
    pca_mat_sift = data['pca_mat_sift']
    gmm_lcs = data['gmm_lcs']
    pca_mat_lcs = data['pca_mat_lcs']
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    models = []
    y_train_enc = np.eye(1000)[y_train]
    print("compuing XTX") 
    XTX = X_train.T.dot(X_train)
    XTy = X_train.T.dot(y_train_enc)
    num_features = XTX.shape[0]
    for reg in [1e-4]:
        model = {}
        model['gmm_sift'] = gmm_sift
        model['pca_mat_sift'] = pca_mat_sift
        model['gmm_lcs'] = gmm_lcs
        model['pca_mat_lcs'] = pca_mat_lcs
        weights = scipy.linalg.solve(XTX + reg*np.eye(num_features), XTy, sym_pos=True)
        print("Regularization ", reg)
        y_train_pred = X_train.dot(weights)
        y_test_pred = X_test.dot(weights)
        train_acc = top_k_accuracy(y_train, y_train_pred, k=1)
        test_acc =  top_k_accuracy(y_test, y_test_pred, k=1)
        train_acc_top_5  = top_k_accuracy(y_train, y_train_pred, k=5)
        test_acc_top_5  = top_k_accuracy(y_test, y_test_pred, k=5)
        model['weights'] = weights
        model['train_acc'] = train_acc
        model['test_acc'] = test_acc
        model['train_acc_top_5'] = train_acc_top_5
        model['test_acc_top_5'] = test_acc_top_5
        print(f"Reg {reg} top 1 train acc {model['train_acc']}, top 1 test acc {model['test_acc']}")
        models.append(model)
    best_model = max(models, key=lambda x: x['test_acc'])
    with open(args.out_name, "wb+") as f:
        f.write(pickle.dumps(best_model))




