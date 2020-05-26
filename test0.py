# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:56:49 2020

@author: emilm
"""
import numpy as np
from scipy.signal import convolve2d
from animate import InteractiveGameOfLife
import time
import numba as nb

@nb.njit('f4[:,:](f4[:,:], f4[:,:])', parallel=True, cache=True)
def convolve(array, kernel):
    out = np.zeros(array.shape, dtype = np.float32)
    k_2 = np.array((kernel.shape[0] // 2, kernel.shape[0] // 2), np.int64)

    kernel_args = np.argwhere(kernel)
    for kernel_pos in range(kernel_args.shape[0]):
        arg = kernel_args[kernel_pos]
        for i in nb.prange(array.shape[0]):
            for j in nb.prange(array.shape[1]):
                pos0 = (i - arg[0] + k_2[0]) % array.shape[0]
                pos1 = (j - arg[1] + k_2[1]) % array.shape[1]

                out[i, j] += array[pos0, pos1] * kernel[arg[0], arg[1]]

    return out





@nb.njit
def gofl_numpy_kernel(init, ksize, valid, n=200):
    shape = init.shape
    kernel = np.ones((ksize,ksize), np.float32)
    kernel /= np.sum(kernel)
    out = np.empty((n, shape[0], shape[1]), dtype=np.float32)
    work = np.empty(shape, dtype=np.float32)
    out[0] = init

    for i in range(1, n):
        work[:,:] = out[i-1]
        # print(out[n-1])
        # score = convolve2d(work, kernel, mode='same', boundary='wrap')
        score = convolve(work, kernel)
        # out[i] = np.logical_or(score==3, np.logical_and(1,score==2))
        out[i] = score
    return out

def main():
    init = np.random.randint(0,1,(512,512))
    init[254:258,254:258] = 1

    ksize = 5
    valid = np.zeros(ksize**2+10, np.uint8)
    valid[6:11] = 1

    t_start = time.time()
    output = gofl_numpy_kernel(init, ksize, valid, 500)
    t_end = time.time()
    print(t_end - t_start)
    game = InteractiveGameOfLife(output)


if __name__ == '__main__':
    init = np.random.randint(0,1,(512,512))
    init[254:258,254:258] = 1

    ksize = 5
    valid = np.zeros(ksize**2+10, np.uint8)
    valid[6:11] = 1

    t_start = time.time()
    output = gofl_numpy_kernel(init, ksize, valid, 500)
    t_end = time.time()
    print(t_end - t_start)
    game = InteractiveGameOfLife(output)