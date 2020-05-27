# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:15:07 2020

@author: emilm
"""


import numba as nb
import numpy as np
import time
from numba import cuda, float32
from animate import InteractiveGameOfLife



@cuda.jit(device=True)
def a_device_function(a, b):
    return a + b

@cuda.jit('void(b1[:,:],b1[:,:])', cache=True)
def generation(old, new):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time

    line_s = cuda.shared.array(shape=(2,512), dtype=nb.uint8)
    out_s = cuda.shared.array(shape=(2,512), dtype=nb.uint8)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    line_s[tx, ty] = 0
    out_s[tx, ty] = 0
    for i in range(-2, 3):
        line_s[tx, ty] += old[(x + i)% 512, y]
    cuda.syncthreads()

    for i in range(-2, 3):
        out_s[tx, ty] += line_s[tx, (ty + i) % 512]
    cuda.syncthreads()

    if 7 <= out_s[tx, ty] <= 12:
        new[x, y] = 1
    else:
        new[x, y] = 0

def game_of_life(init, n, step=1):
    grid_dim = 256
    block_dim =(2, 512)
    out = cuda.pinned_array((n//step, 512, 512), np.bool)
    d_A = cuda.to_device(init)
    d_B = cuda.device_array((512, 512), dtype=np.bool)

    work_stream = cuda.stream()
    down_stream = cuda.stream()
    for i in range(0, n):
        if i % step == 0:
            d_A.copy_to_host(out[i//step], down_stream)
        generation[grid_dim, block_dim, work_stream](d_A, d_B)
        work_stream.synchronize()
        down_stream.synchronize()
        d_A, d_B = d_B, d_A
    return out


if __name__ == '__main__':
    generations = 4000
    shape = (512, 512)
    init = cuda.pinned_array(shape, np.bool)
    init[254:258,254:258] = 1
    t0 = time.time()
    generations = game_of_life(init, generations, 500)
    print(time.time()-t0)
    game = InteractiveGameOfLife(generations.astype(np.float32))
