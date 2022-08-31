#import sys
#sys.path.append('C:\\Users\\siat\\anaconda3\\envs\\TF1.x\\lib\\site-packages')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["MKL_NUM_THREADS"] = "8"
#os.environ["NUMEXRR_NUM_THREADS"]= "8"
#os.environ["OMP_NUM_THREADS"] = "8"
#os.environ["openmp"] = "True"
import tensorflow as tf
import numpy as np
import h5py
import os
import datetime
from skimage import io
import time
import scipy.io as sio
from numpy.fft import fft2, ifft2, fftshift



def generate_data(x, csm,BATCH_SIZE, shuffle=False):
    """Generate a set of random data."""
    n = len(x)
    ind = np.arange(n)
    if shuffle:
        ind = np.random.permutation(ind)
        x = x[ind]
        csm = csm[ind]
        # mask = mask[ind]

    for j in range(0, n, BATCH_SIZE):
        yield x[j:j + BATCH_SIZE], csm[j:j + BATCH_SIZE]
