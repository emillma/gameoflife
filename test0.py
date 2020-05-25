# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:56:49 2020

@author: emilm
"""
import numpy as np
from scipy.signal import convolve2d
from animate import InteractiveGameOfLife
import time



def gofl_numpy_kernel(init, ksize, valid, n=200):
    shape = init.shape
    kernel = np.ones((ksize,ksize), np.uint8)
    out = np.empty((n, shape[0], shape[1]), dtype=np.int32)
    work = np.empty(shape, dtype=np.uint8)
    out[0] = init

    for i in range(1, n):
        work[:,:] = out[i-1]
        # print(out[n-1])
        score = convolve2d(work, kernel, mode='same', boundary='wrap')
        # out[i] = np.logical_or(score==3, np.logical_and(1,score==2))
        out[i] = valid[score]
    return out

def gofl_numpy_cumsum(init, ksize, valid, n=200):
    shape = init.shape
    workshape = None
    kernel = np.ones((ksize,ksize), np.uint8)
    out = np.empty((n, shape[0], shape[1]), dtype=np.int32)
    work = np.empty(shape, dtype=np.uint8)
    out[0] = init

    for i in range(1, n):
        work[:,:] = out[i-1]
        # print(out[n-1])
        score = convolve2d(work, kernel, mode='same', boundary='wrap')
        # out[i] = np.logical_or(score==3, np.logical_and(1,score==2))
        out[i] = valid[score]
    return out


if __name__ == '__main__':
    init = np.random.randint(0,1,(512,512))
    init[254:258,254:258] = 1

    ksize = 5
    valid = np.zeros(ksize**2+1, np.uint8)
    valid[6:11] = 1

    t_start = time.time()
    output = gofl_numpy_kernel(init, ksize, valid, 500)
    t_end = time.time()
    print(t_end - t_start)
    game = InteractiveGameOfLife(output)