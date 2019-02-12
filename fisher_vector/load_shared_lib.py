import ctypes
import sys
import boto3
import os

if (sys.platform == "linux"):
    END = ".so"
elif (sys.platform == "darwin"):
    END = ".dylib"
else:
    raise Exception("unsupported platform")


def load_shared_lib(key="libImageFeatures", bucket="pictureweb"):
    local_path = '/tmp/libImageFeatures' + END
    key += END
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).download_file(key, local_path)
    return ctypes.cdll.LoadLibrary(local_path)




