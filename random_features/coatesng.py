import concurrent.futures as fs
import gc
import os
import time

import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from scipy.linalg import solve
from sklearn import metrics
from torch import nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

import utils


class IdentityNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BasicCoatesNgNet(nn.Module):
    ''' All image inputs in torch must be C, H, W '''

    def __init__(self, filters, patch_size=6, in_channels=3, pool_size=2, pool_stride=2, bias=1.0, filter_batch_size=1024):
        super().__init__()
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.bias = bias
        self.filter_batch_size = filter_batch_size
        self.filters = filters.copy()
        self.active_filter_set = []
        self.start = None
        self.end = None
        self.gpu = False

    def _forward(self, x):
        # Max pooling over a (2, 2) window
        if 'conv' not in self._modules:
            raise Exception('No filters active, conv does not exist')
        conv = self.conv(x)
        x_pos = F.avg_pool2d(F.relu(conv - self.bias), [self.pool_size, self.pool_size],
                             stride=[self.pool_stride, self.pool_stride], ceil_mode=True)
        x_neg = F.avg_pool2d(F.relu((-1*conv) - self.bias), [self.pool_size, self.pool_size],
                             stride=[self.pool_stride, self.pool_stride], ceil_mode=True)
        return torch.cat((x_pos, x_neg), dim=1)

    def forward(self, x):
        num_filters = self.filters.shape[0]
        activations = []
        for start, end in utils.chunk_idxs_by_size(num_filters, self.filter_batch_size):
            activations.append(self.forward_partial(x, start, end))
        z = torch.cat(activations, dim=1)
        return z

    def forward_partial(self, x, start, end):
        # We do this because gpus are horrible things
        self.activate(start, end)
        return self._forward(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def activate(self, start, end):
        if (self.start == start and self.end == end):
            return self
        self.start = start
        self.end = end
        filter_set = torch.from_numpy(self.filters[start:end])
        if (self.use_gpu):
            filter_set = filter_set.cuda()
        conv = nn.Conv2d(self.in_channels, end - start,
                         self.patch_size, bias=False)
        #print("rebounding nn.Parameter this shouldn't happen that often")
        conv.weight = nn.Parameter(filter_set)
        self.conv = conv
        self.active_filter_set = filter_set
        return self

    def deactivate(self):
        self.active_filter_set = None


class CoatesNgTrained(nn.Module):

    def __init__(self, feed_forward, weights, whitening_weights=None):
        super().__init__()
        self.feed_forward = feed_forward
        self.weights = weights
        self.classifier = torch.nn.Linear(*weights.shape, bias=False)
        self.classifier.weight = nn.Parameter(weights.t())
        self.whitening_weights = nn.Parameter(whitening_weights)

        if ((whitening_weights is None)):
            self.normalize = False
        else:
            self.normalize = True

    def forward(self, x):
        if (self.normalize):
            x = self.whiten(x)
        features = self.feed_forward(x)
        features = features.view(features.size(0), features.size(
            1)*features.size(2)*features.size(3))
        return self.classifier(features)

    def whiten(self, x):
        orig_shape = x.shape
        x = x.view(orig_shape[0], -1)
        row_means = torch.mean(x, dim=1)
        x = x - row_means.unsqueeze(1).expand_as(x)
        row_norms = torch.norm(x, p=2, dim=1)
        x /= row_norms.unsqueeze(1).expand_as(x)
        return torch.mm(x, self.whitening_weights).view(*orig_shape)


def coatesng_featurize(net, X, data_batchsize=128, num_filters=None, filter_batch_size=None, gpu=False, rgb=True):
    assert len(X.shape) == 4
    net.use_gpu = gpu
    if (rgb):
        X = X.transpose(0, 3, 1, 2)
    if (filter_batch_size == None):
        filter_batch_size = net.filter_batch_size
    if (num_filters == None):
        num_filters = len(net.filters)
    X_lift_full = []
    models = []

    for start, end in utils.chunk_idxs_by_size(num_filters, filter_batch_size):
        data_loader = torch.utils.data.DataLoader(
            torch.from_numpy(X), batch_size=data_batchsize)
        X_lift_batch = []
        for j, X_batch in enumerate(data_loader):
            if (gpu):
                X_batch = X_batch.cuda()
            X_var = X_batch
            X_lift = net.forward_partial(X_var, start, end).cpu().data.numpy()
            X_lift_batch.append(X_lift)
        X_lift_full.append(np.concatenate(X_lift_batch, axis=0))
    conv_features = np.concatenate(X_lift_full, axis=1)
    net.deactivate()
    return conv_features.reshape(X.shape[0], -1)


def batched_evaluate(X, net_trained, use_gpu=True, batch_size=128):
    net_trained.feed_forward.use_gpu = use_gpu
    y_pred = []
    X = X.astype('float32')
    for (sidx, eidx) in utils.chunk_idxs(X.shape[0], X.shape[0]//batch_size):
        X_batch = X[sidx:eidx]
        X_torch = torch.from_numpy(X_batch).cuda()
        y_pred.append(np.argmax(net_trained.forward(
            torch.autograd.Variable(X_torch)).data.cpu().numpy(), axis=1))
    return np.hstack(y_pred)
