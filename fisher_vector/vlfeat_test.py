import ctypes
from numpy.ctypeslib import ndpointer, as_array
import numpy as np
import time
from load_shared_lib import load_shared_lib

vlfeat  = load_shared_lib().ctypes_get_dense_sift
vlfeat.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_int)]
vlfeat.restype = ctypes.POINTER(ctypes.c_short)
x = np.arange(32*32).astype('float32')
z = np.array([0]).astype('int32')
h,w = 32,32
img = np.genfromtxt("sift_input.txt", delimiter=",").astype('float32')
sifts = vlfeat(h,w,3,4,4,1,img,img.shape[0],z)
sifts = as_array(sifts, shape=(z[0],))
sifts = sifts.reshape(-1, 128).T
sifts_scala = np.array([[int(float(y)) for y in x.strip().split(",")] for x in open("scala_sifts.txt").read().strip().split("\n")])
assert(np.all(sifts_scala == sifts))
print("Python SIFT matches Scala SIFT")


img = np.arange(256*256).astype('float32')
h,w = 256,256
t = time.time()
sifts = vlfeat(h,w,3,4,4,1,img,img.shape[0],z)
e = time.time()
print("SIFT TAKES {0}s for a 256 x 256 img".format(e - t))
