import ctypes
from numpy.ctypeslib import ndpointer, as_array
import numpy as np
import time
from numpy import asfortranarray as F_ARRAY
from load_shared_lib import load_shared_lib
import pywren
import scipy.misc

ctypes_get_dense_sift = load_shared_lib().ctypes_get_dense_sift
ctypes_get_dense_sift.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_int)]
ctypes_get_dense_sift.restype = ctypes.POINTER(ctypes.c_short)

SIFT_DESC_LENGTH = 131


def sift(img, stride=3, bins=4, num_scales=4, scale_step=1):
    if (img.shape[-1] == 1):
        img = img.reshape(img.shape[0], img.shape[1])
    if (len(img.shape) != 2):
        img = (299/1000)*img[:, :,0] + (587/1000)*img[:, :, 1] + (114/1000)*img[:, :, 2]
    w,h = img.shape
    img = img.astype('float32')
    img = F_ARRAY(img)
    z = np.array([0]).astype('int32')
    sifts = ctypes_get_dense_sift(w,h,stride,bins,num_scales,scale_step,img,img.shape[0],z)
    sifts = as_array(sifts, shape=(z[0],))
    sifts = sifts.reshape(-1, SIFT_DESC_LENGTH)
    return sifts

if __name__ == "__main__":
    img = np.genfromtxt("sift_input.txt", delimiter=",").astype('float32')
    img = img.reshape(32,32).T
    descs = sift(img, stride=3, bins=4, num_scales=4, scale_step=1)[:, :128]
    positions = sift(img, stride=3, bins=4, num_scales=4, scale_step=1)[:, 128:]
    print("POSITIONS")
    print(positions)
    descs_keystone = np.array([[int(float(y)) for y in x.strip().split(",")] for x in open("scala_sifts.txt").read().strip().split("\n")]).T
    print(descs)
    print(descs_keystone)
    print(descs.shape)
    print(descs_keystone.shape)
    assert(np.all(descs == descs_keystone))
    test_img = scipy.misc.imread('test.png', flatten=True)
    descs = sift(test_img, stride=3, bins=4, num_scales=4, scale_step=1)
    print(descs)








