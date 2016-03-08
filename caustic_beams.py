# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:44:16 2016

@author: ben
S"""
import sympy as sp
import numpy as np
import pickle
from time import time
from copy import copy
from string import lowercase
from matplotlib import pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy import interpolate
# spyder always imports the numpy version of sum, but we need the builtin
# version
from __builtin__ import sum

# ================== CONFIGURATION ==================

# the order of the catastrophe
ORDER = 4
# list of tuples with 2 elements each, corresponding to the limits of
# the corresponding axis
LIMITS = [
    (-50,50),   # a axis
    (-50,50),   # b axis
    #(0,)*2,  # c axis
    #(0,)*2,  # d axis
]#
# number of data points for each axis
RESOLUTION = [200,200]
# number of weights for legendre gauss integration
NUM_WEIGHTS = 500
# neglected integral is proportional to exp(-D)
D = 100
# a grid of N_ROOTS**2 roots is calculated and then interpolated.
# N_ROOTS=25 should be enough
N_ROOTS = 25
# in case of pearcey beam:
# should the results of method of steepest descent and integration method be
# merged together? This reduces the number of weights needed.
# Otherwise, only integration method is used.
USE_MASK = True

# ==================================================
assert (len(LIMITS) == ORDER - 2) and (len(RESOLUTION) == 2), \
    "not the right number of parameters"

# expand RESOLUTION to ORDER-2 dimensions
for i, l in enumerate(LIMITS):
    if l[0] == l[1]:
        RESOLUTION.insert(i, 1)
# which axes to plot?
PLOT_AXES = list(np.where(np.array(RESOLUTION) > 1)[0])
# limits of the axes to plot
PLOT_LIMITS = [(LIMITS[i][0], LIMITS[i][1], RESOLUTION[i]) for i in PLOT_AXES]
#%%
# ================== ANALYTICAL CALCULATIONS ==================
print 'ANALYTICAL CALCULATIONS'
t = time()

# generate catastrophe

# list of generated variables
var = []
S = sp.Symbol('S')

catastrophe = S ** (ORDER)

for i in xrange(ORDER - 2):
    tmp_var = sp.Symbol(lowercase[i])
    catastrophe += tmp_var * S ** (i+1)
    var.append(tmp_var)

# calculate derivatives
expr_d = sp.diff(catastrophe, S)
expr_dd = sp.diff(expr_d, S)
expr_ddd = sp.diff(expr_dd, S)

if ORDER == 4:
    # find saddle points (dep. of S)
    saddles_s = sp.solve(expr_d, S)
    
    # calculate the value of the function and its derivatives at each saddle point
    def _plug_in(expr, S, saddle_s, var):
        expr = expr.subs(S, saddle_s)
        return sp.lambdify(var, expr, [{'ImmutableMatrix': np.array}, 'numpy'])
    
    val = [_plug_in(catastrophe, S, saddle_s, var) for saddle_s in saddles_s]
    d = [_plug_in(expr_d, S, saddle_s, var) for saddle_s in saddles_s]
    dd = [_plug_in(expr_dd, S, saddle_s, var) for saddle_s in saddles_s]
    ddd = [_plug_in(expr_ddd, S, saddle_s, var) for saddle_s in saddles_s]

# generate an integral representation for the caustic beam
caustic_beam = sp.exp(1j * catastrophe)
caustic_beam = sp.lambdify([S] + var, caustic_beam, 'numexpr', dummify=False)

print '%f seconds' % (time() - t)
#%%
# ================== INTEGRATION METHOD ==================
print 'INTEGRATION METHOD'
t = time()

# get legendre gauss coefficients
# use a cache because calculating them for more than 1000 weights is very
# time consuming
cache = {}
try:
    f = file('gauss_cache', 'r')
    cache = pickle.load(f)
    s_values, weights = cache[NUM_WEIGHTS]
    f.close()
except (IOError, KeyError), e:
    f = file('gauss_cache', 'w')
    s_values, weights = leggauss(NUM_WEIGHTS)
    cache[NUM_WEIGHTS] = (s_values, weights)
    pickle.dump(cache, f)
    f.close()

# change position range from [-1,1] to [0,1]
s_values = (s_values + 1) * 0.5

# generate grid
slices = [slice(lim[0], lim[1], res*1j) for lim, res in zip(LIMITS, RESOLUTION)]
grid_fine = np.mgrid[slices].astype(np.complex)
range_fine = [np.linspace(*l) for l in PLOT_LIMITS]

# generade coarse grid for root calculation
slices2 = [slice(lim[0], lim[1], res > 1 and (N_ROOTS*1j) or 1j) for lim, res in zip(LIMITS, RESOLUTION)]
grid_coarse = np.mgrid[slices2].astype(np.float)
cp = copy(grid_coarse)
range_coarse = [np.linspace(l[0], l[1], N_ROOTS) for l in PLOT_LIMITS]

def _get_coeff(pos, neg=False):
    # factor that determines whether integration takes place below or above the
    # real axis in complex space
    if neg and ((ORDER) % 2 != 0):
        fac = -1
    else:
        fac = 1

    # prefactor that determines whether integration takes place towars inf or
    # -inf
    pref = neg and -1 or 1
    
    coeff = np.zeros(ORDER + 1)
    
    # zeroth order
    coeff[-1] = -D

    for i in xrange(ORDER - 2):
        order = (i+1)
        idx = -order - 1
        coeff[idx] = (pref ** order) * pos[i] * \
            np.sin(fac * order * np.pi / (2*ORDER))

    # highest order    
    coeff[0] = 1
    return coeff

# calculate a coarse grid of roots
for c in zip(*[np.nditer(coord, op_flags=['readwrite']) for coord in grid_coarse]):
    r1 = np.roots(_get_coeff(c))
    r2 = np.roots(_get_coeff(c, True))
    c[0][...] = max(r1[abs(np.imag(r1)) == 0])
    c[1][...] = max(r2[abs(np.imag(r2)) == 0])

roots_coarse = np.squeeze(grid_coarse[0])
roots_coarse_neg = np.squeeze(grid_coarse[1])

# create fine grid for the roots by means of interpolation
f = interpolate.RectBivariateSpline(range_coarse[0], range_coarse[1], roots_coarse)
f2 = interpolate.RectBivariateSpline(range_coarse[0], range_coarse[1], roots_coarse_neg)
roots_fine = f(range_fine[0], range_fine[1])
roots_fine_neg = f2(range_fine[0], range_fine[1])

# we have to expand the dimension of the grid again
for i, l in enumerate(LIMITS):
    if i not in PLOT_AXES:
        roots_fine = np.expand_dims(roots_fine, axis=i)
        roots_fine_neg = np.expand_dims(roots_fine_neg, axis=i)

# the results will be stored here
mat = np.zeros(RESOLUTION).astype(np.complex)
mat_neg = copy(mat)
mat2 = copy(mat)
mat2_neg = copy(mat)

# at which angle is the endpoint of the integration along the arc
angle = np.pi / (2 * ORDER)
angle_neg = ((-1) ** ORDER) * np.pi / (2 * ORDER)

for s, w in zip(s_values, weights):
    # integration from 0 to R1
    mat += w * caustic_beam(s * roots_fine, *grid_fine)
    # integration from 0 to -R2
    mat_neg += w * caustic_beam(-s * roots_fine_neg, *grid_fine)

    # integration from R1 along the arc
    s_val = roots_fine * np.exp(1j * s * angle)    
    mat2 += w * caustic_beam(s_val, *grid_fine)

    # integration from -R2 along the arc
    s_val_neg = -roots_fine_neg * np.exp(1j * s * angle_neg)
    mat2_neg += w * caustic_beam(s_val_neg, *grid_fine)

# proper scaling for legendre gauss integration
mat *= 0.5 * roots_fine
mat_neg *= 0.5 * roots_fine_neg
mat2 *= 0.5 * roots_fine * abs(angle) / (2*np.pi)
mat2_neg *= 0.5 * roots_fine_neg * abs(angle_neg) / (2*np.pi)

# remove extra dimensions
E_i = np.squeeze(mat + mat_neg + mat2 + mat2_neg).T

print '%f seconds' % (time() - t)
#%%
# ================== METHOD OF STEEPEST DESCENT ==================
if ORDER == 4:
    # which axis is symmetric? Speeds up calculations for steepest descent
    SYMMETRY = 0

    print 'METHOD OF STEEPEST DESCENT'
    t = time()
    limits_s = copy(LIMITS)
    resolution_s = copy(RESOLUTION)

    if SYMMETRY is not None:
        # calculate only half of the image for symmetry reasons
        limits_s[SYMMETRY] = (limits_s[SYMMETRY][0], 0)
        assert RESOLUTION[SYMMETRY] % 2 == 0, 'Resolution in symmetry direction has to be even'
        resolution_s[SYMMETRY] = resolution_s[SYMMETRY] / 2
    
    
    def _saddle_contribution(val, dd, saddle_i, *positions):
        """
        Calculates the contribution to the E-field of each saddle.
        """
        val = val[saddle_i](*positions)
        deriv = dd[saddle_i](*positions)
    
        ang = np.angle(deriv)
        theta_ = (np.pi - ang) / 2

        if saddle_i == 1:
            theta_[theta_ > np.pi/2] -= np.pi
    
        return np.exp(1j * (val + theta_)) * np.sqrt(2*np.pi/abs(deriv))
    
    # create the grid
    slices = [slice(lim[0], lim[1], res*1j) for lim, res in zip(limits_s, resolution_s)]
    positions = np.mgrid[slices].astype(complex)
    
    # calculate E field
    E_s = sum(_saddle_contribution(val, dd, i, *positions) for i,s in enumerate(saddles_s))
    E_s[np.isnan(E_s)] = 0
    
    # use symmetry
    stack = (PLOT_AXES.index(SYMMETRY) == 0) and np.vstack or np.hstack
    E_s = stack((E_s, np.flipud(E_s))).T
    
    print '%f seconds' % (time() - t)

#%%
# ================== PLOTS ==================

# how far away from the caustic should the two matrices get joined?
CAUSTIC_DISTANCE = 15

axes = []

for i, (lim, res) in enumerate(zip(LIMITS, RESOLUTION)):
    if i in PLOT_AXES:
        axes.append(np.linspace(lim[0], lim[1], res))

if USE_MASK and ORDER == 4:
    mask = np.zeros(E_s.shape)
    
    # position of the caustic y=f(x)
    f = lambda x: -(27.0/8*(abs(x)+CAUSTIC_DISTANCE)**2)**(1.0/3)

    for i, x_ in enumerate(axes[0]):
        for j, y_ in enumerate(axes[1]):
            if f(x_) > y_:
                mask[i,j] = 1

    mask = mask.T

    E_s_mask = np.ma.masked_array(E_s, mask==0)
    E_i_mask = np.ma.masked_array(E_i, mask)

    # merge matrices
    E = E_s_mask.filled(0) + E_i_mask.filled(0) * np.exp(1j*np.pi/4)
else:
    E = E_i

I = abs(E)**2
ang = np.angle(E)

plt.clf()

def _plot(mat, vmin, vmax, i):
    plt.subplot(2, 1, i)
    plt.pcolormesh(axes[0], axes[1], mat, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.xlim(min(axes[0]), max(axes[0]))
    plt.ylim(min(axes[1]), max(axes[1]))
    plt.xlabel(lowercase[PLOT_AXES[0]])
    plt.ylabel(lowercase[PLOT_AXES[1]])

_plot(I, 0, 6, 1)
_plot(ang, -np.pi, np.pi, 2)