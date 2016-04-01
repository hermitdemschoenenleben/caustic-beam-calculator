# -*- coding: utf-8 -*-
"""
Calculate position of caustic of swallowtail beam.

Author: Benjamin Wiegand, highwaychile@zoho.com
"""
from string import lowercase
import sympy as sp
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat

LIMITS = [
    (-50,50),
    (-50,50),
    (-50,50),
]

RESOLUTION = (300,300,15)

LIMITS2D = list(np.copy(LIMITS))
RESOLUTION2D = list(np.copy(RESOLUTION))
LIMITS2D.pop(2)
RESOLUTION2D.pop(2)

ORDER = 5
AXIS = 2 # XXX: can't change this without breaking code

RECALCULATE = True
SAVE = True

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

# coefficients of S in derivative
saddle_coefficients = sp.lambdify(var, sp.Poly(expr_d,S).all_coeffs(), 'numpy')

# "compile" expressions for fast computation
d = sp.lambdify([S] + var, expr_d, 'numexpr')
dd = sp.lambdify([S] + var, expr_dd, 'numexpr')


X = [[] for i in xrange(ORDER-1)]
Y = [[] for i in xrange(ORDER-1)]
Z = [[] for i in xrange(ORDER-1)]

def export():
    ranges = [np.linspace(l[0], l[1], r) for l, r in zip(LIMITS, RESOLUTION)]
    grid = np.meshgrid(*ranges)

    result = [np.zeros(RESOLUTION) for s in X]

    for saddle_i, points in enumerate(zip(X, Y, Z)):
        for pos in zip(*points):
            idx = []
            for i, coord in enumerate(pos):
                idx.append(np.where(ranges[i] == coord))
            result[saddle_i][idx] = 1

        savemat('punkte_sattel_%d' % (saddle_i+1), {'caustic': result[saddle_i]})

    savemat('alle_punkte.mat', {
        'caustic':  np.sum(result, axis=0)
    })
    return result

def find_min(arr):
    mn = argrelextrema(arr, np.less)
    mn = (np.diff(np.sign(np.diff(arr))) > 0).nonzero()[0] + 1
    arr = arr * 0
    arr[mn] = 1
    return arr

z_values = np.linspace(LIMITS[2][0], LIMITS[2][1], RESOLUTION[2]).astype(complex)
slices = [slice(lim[0], lim[1], res*1j) for lim, res in zip(LIMITS2D, RESOLUTION2D)]
grid2d = np.squeeze(np.mgrid[slices].astype(np.complex))

if SAVE and RECALCULATE:
    saddles_save = [np.zeros(RESOLUTION).astype(complex) for p in xrange(ORDER-1)]

for i, z in enumerate(z_values):
    break
    """if i < 60 or i > 95:
        continue"""
    saddles = [np.zeros(RESOLUTION2D).astype(complex) for p in xrange(ORDER-1)]

    if RECALCULATE:
        print 'step 1: %f %% ' % (float(i) * 100 / len(z_values))
        for idx2d in np.ndindex(*RESOLUTION2D):
            coord = [g[idx2d] for g in grid2d] + [z]

            for j, s in enumerate(np.roots(saddle_coefficients(*coord))):
                saddles[j][idx2d] = s
                if SAVE:
                    saddles_save[j][idx2d[0], idx2d[1], i] = s
    else:
        for j, s in enumerate(saddles_save):
            saddles[j] = saddles_save[j][:, :, i]

    for saddle_i, saddle in enumerate(saddles):
        dd_2d = np.array(
            abs(dd(saddle, grid2d[0], grid2d[1], z))
        )
        d_2d = np.array(
            abs(d(saddle, grid2d[0], grid2d[1], z))
        )

        result_col = np.copy((dd_2d))
        result_row = np.copy((dd_2d)).T

        for n_col, col in enumerate(result_col):
            result_col[n_col] = find_min(abs(col))

        for n_row, row in enumerate(result_row):
            result_row[n_row] = find_min(abs(row))

        result = result_row + result_col.T
        result[result < 1] = 0
        result[result > 0] = 1
        result[abs(dd_2d.T)>10] = 0

        result[np.isnan(result)] = 0

        # filter out small artefacts
        """if len(result[result>0]) < RESOLUTION[0]*RESOLUTION[1] / 5000:
            result *= 0
            print '?'
        else:
            print '!'"""

        caustic_idx = result > 0

        height = (z,) * len(caustic_idx[caustic_idx])

        X[saddle_i] += list(grid2d[0][caustic_idx])
        Y[saddle_i] += list(grid2d[1][caustic_idx])
        Z[saddle_i] += list(height)

        """if saddle_i == 3 and i == 0:
            print len(result[result>0])
            print RESOLUTION[0]*RESOLUTION[1] / 5000
            plt.clf()
            plt.pcolormesh((dd_2d.T), vmax=5)
            plt.colorbar()
            plt.pause(0.1)
            raw_input('')
            plt.clf()
            d_2d = ndimage.filters.gaussian_filter(d_2d, 1, mode='nearest')
            plt.pcolormesh((d_2d.T))
            plt.colorbar()
            plt.pause(0.1)
            raw_input('')
            plt.pcolormesh(result)
            plt.pause(0.1)
            raw_input('')"""
"""plt.clf()
plt.pcolormesh(res)
plt.pause(0.1)
raw_input(value)
continue"""
#result = ndimage.filters.gaussian_filter(result, 1, mode='nearest')

fig = plt.gcf()
plt.clf()
ax = Axes3D(fig)


def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=5, linewidth=0)
    scalarMap.set_array(cs)
    plt.show()

def do_scatter(num, cmap=None):
    x = np.array(X[num])
    y = np.array(Y[num])
    z = np.array(Z[num])

    skip = 1
    scatter3d(x[::skip], z[::skip], y[::skip], y[::skip], cmap)

#do_scatter(1)

def nan_helper(w):
    return np.isnan(w), lambda q: q.nonzero()[0]

def do_solid(num, interpolate, half, options):
    x = np.array(X[num])
    y = np.array(Y[num])
    z = np.array(Z[num])

    new = np.zeros((RESOLUTION[2], RESOLUTION[0]))
    new[:] = np.nan

    for x_, y_, z_ in zip(x,y,z):
        idx1 = np.where(grid2d[0][:,0] == x_)
        idx2 = np.where(z_values == z_)
        test = new[idx2,idx1]
        if np.isnan(test):
            if half is None or ((half == 0) and (y_ > 0)) or ((half == 1) and (y_ < 0)):
                new[idx2,idx1] = y_
        else:
            print 'doppelt'

    if interpolate:
        for i, row in enumerate(new):
            nans, x_ = nan_helper(row)
            try:
                row[nans]= np.interp(x_(nans), x_(~nans), row[~nans])
            except:
                pass

        for r, row in enumerate(new):
            start = row[0]
            end = row[-1]

            for c, cell in enumerate(row):
                if cell != start:
                    new[r, 0:c] = np.nan
                    break
                if c == len(row) - 1:
                    new[r, :] = np.nan

            for c, cell in enumerate(reversed(row)):
                if cell != end:
                    new[r, -c:] = np.nan
                    break

    new_grid = np.meshgrid(grid2d[0][:,0], z_values)
    #new = ndimage.filters.gaussian_filter(new, 1, mode='nearest')
    """plt.clf()
    new[np.isnan(new)] = 20
    plt.pcolormesh(new)
    asd"""
    #test = np.copy(new)
    #test[np.isnan(test)] = 0
    #norm = Normalize(vmin = np.min(test), vmax = np.max(test), clip = False)
    ax.plot_wireframe(new_grid[0], new_grid[1], new, **options)

options = {
    'alpha':        0.5,
    'rstride':      5,
    'cstride':      5,
    'color':        'b',
}


"""do_solid(3, True, 0, options)
do_solid(3, True, 1, options)
do_solid(1, False, None, {
    'color':        'r',
    'alpha':        0.5,
})"""

do_scatter(1, 'Reds')
do_scatter(3, 'Blues')

ax.set_xlabel('A1')
ax.set_ylabel('A3')
ax.set_zlabel('A2')