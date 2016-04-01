# -*- coding: utf-8 -*-
"""
Calculate E-field of caustic beams by using fourier transform.

Author: Benjamin Wiegand, highwaychile@zoho.com
"""
import numpy as np
from numpy.fft import fft2, ifftshift, fftshift, ifft2
from matplotlib import pyplot as plt
from scipy.misc import imresize


LIMITS = [50, 50]
RESOLUTION = [7000, 7000]
THRESHOLD = 0.05
FACTOR = 10


xlim = float(RESOLUTION[0]) / (LIMITS[0]*FACTOR) * np.pi / 2
ylim = float(RESOLUTION[1]) / (LIMITS[1]*FACTOR) * np.pi / 2

x = np.linspace(-xlim, xlim, RESOLUTION[0]).astype(complex)
y = np.linspace(-ylim, ylim, RESOLUTION[1]).astype(complex)

X, Y = np.meshgrid(x, y)

mat = X ** 2 - Y
idx = abs(mat) < THRESHOLD
mat[idx] = np.exp(1j * (X**4))[idx]
mat[~idx] = 0

del x, y, X, Y, idx

ft = fftshift(fft2(mat))

# crop according to FACTOR
new_res = [r/FACTOR for r in RESOLUTION]
pos = [r/2 - n/2 for r, n in zip(RESOLUTION, new_res)]
ft = ft.T[pos[0]:pos[0]+new_res[0], pos[1]:pos[1]+new_res[1]]

# plots

plt.figure(31)
plt.clf()

# don't plot huge matrices
plt.pcolormesh(imresize(abs(mat)**2, (400,400)))

def _do_plot(mat):
    plt.clf()
    xran = np.linspace(-LIMITS[0], LIMITS[0], RESOLUTION[0] / FACTOR)
    yran = np.linspace(-LIMITS[1], LIMITS[1], RESOLUTION[1] / FACTOR)

    plt.pcolormesh(xran, yran, mat)

    plt.xlim(xran[0], xran[-1])
    plt.ylim(yran[0], yran[-1])
    plt.xlabel('X')
    plt.ylabel('Y')

plt.figure(30)
_do_plot(abs(ft.T)**2)

plt.figure(32)
_do_plot(np.angle(ft.T))
plt.colorbar()