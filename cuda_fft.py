# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:23:17 2020

@author: emilm
"""
import numpy as np
import numba as nb
from numba import cuda

butterflies = {}
omegas = {}


def cooley_tukey(data, out, level, omega_arr, butterfly_arr):
    tmp = np.zeros(data.size, dtype=np.complex64)
    for k in range(data.size):
        out[k] = data[butterfly_arr[k]]

    for step in range(level):
        for k in range(data.size):
            sign = 1 - ((1 & k >> step) << 1)
            omega_index = (k % (1 << step)) * data.size >> (1 + step)
            omega_kn = omega_arr[omega_index]

            pair_index = k - (k & (1 << step))
            odd_index = k + (~k & (1 << step))

            tmp[k] = sign * omega_kn * out[odd_index] + out[pair_index]

        for k in range(data.size):
            out[k] = tmp[k]

    return out


@cuda.jit('void(c8[:],c8[:], u1, c8[:], u4[:])', cache=1)
def cuda_cooley_tukey(data, out, level, omega_arr, butterfly_arr):
    s_work = cuda.shared.array(1024, dtype=nb.c8)
    s_tmp = cuda.shared.array(1024, dtype=nb.c8)
    s_omega = cuda.shared.array(1024, dtype=nb.c8)


    x = cuda.grid(1)
    tx = cuda.threadIdx.x
    if x >= data.size:
        return
    if x < omega_arr.size:
        s_omega[x] = omega_arr[x]


    s_work[x] = data[butterfly_arr[x]]
    for step in range(level):
        cuda.syncthreads()
        sign = 1 - ((1 & x >> step) << 1)
        omega_index = (x % (1 << step)) * data.size >> (1 + step)
        omega_kn = s_omega[omega_index]

        pair_index = x - (x & (1 << step))
        odd_index = x + (~x & (1 << step))

        s_tmp[x] =  sign * omega_kn * s_work[odd_index] + s_work[pair_index]
        # s_tmp[x] = omega_kn
        cuda.syncthreads()
        s_work[x] = s_tmp[x]
    # s_work[x] = 1
    out[x] = s_work[x]


@cuda.jit('void(u1, u4[:])', cache=1)
def generate_butterfly(level, out):
    s_work = cuda.shared.array(1024, dtype=nb.u4)
    x = cuda.grid(1)
    tx = cuda.threadIdx.x
    s_work[tx] = 0

    for index in range(level):
        s_work[tx] |= (1 & x >> (level - index - 1)) << index
    out[x] = s_work[tx]


def fft(data):
    size = data.size
    N = 1
    level = 0
    while N < size:
        level += 1
        N <<= 1
    data = np.concatenate((data, data[:N-size]))
    out = np.zeros(data.size, dtype=np.complex64)
    k_arr = k_arr = np.arange(N, dtype=np.uint16)
    if N not in butterflies:
        butterflie = np.zeros(N, dtype=np.uint16)
        for index in range(level):
            butterflie |= (1 & k_arr >> (level - index - 1)) << index
        butterflies[N] = butterflie

    if N not in omegas:
        omegas[N] = np.exp(-2j * np.pi * k_arr[:N//2] / N)

    cooley_tukey(data, out, level, omegas[N], butterflies[N])
    return out


def ifft(data):
    data = np.conjugate(data)
    out = fft(data)
    return np.conjugate(out) / out.size


# data = np.arange(2<<15)
# out = ifft(fft(data))
data = np.random.random(1024).astype(np.complex64)
d_data = cuda.to_device(data)

out = cuda.device_array(1024, dtype=np.complex64)
level = np.uint8(10)
omega = np.exp(-2j * np.pi * np.arange(1024)[:1024//2] / 1024).astype(np.complex64)
d_butter = cuda.device_array(1024, dtype=np.uint32)
generate_butterfly[1,1024](10, d_butter)

but = d_butter.copy_to_host()

cuda_cooley_tukey[1,1024](d_data, out, level, omega, d_butter)

g = out.copy_to_host()
g2 = fft(data)