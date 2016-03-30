from string import lowercase
import sympy as sp
import numpy as np
import scipy.ndimage.filters as filters
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
LIMITS = [
    (-100,100),
    (-100,100),
    (-100,100),
]
ORDER = 5

RESOLUTION = [100,100,20]

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
dd = sp.lambdify([S] + var, expr_dd, 'numexpr')

X = [[] for i in xrange(ORDER-1)]
Y = [[] for i in xrange(ORDER-1)]
Z = [[] for i in xrange(ORDER-1)]


saddles = []

for i, value in enumerate(vals):
    LIMITS[2] = (value, value)
    slices = [slice(lim[0], lim[1], res*1j) for lim, res in zip(LIMITS, RESOLUTION)]
    grid = np.squeeze(np.mgrid[slices].astype(np.complex))

    saddles_m = [np.zeros(grid[0].shape).astype(complex) for p in xrange(ORDER-1)]
    print i

    test = np.zeros(grid[0].shape)
    for idx in np.ndindex(grid[0].shape):
        coord = [g[idx] for g in grid]
        for i, s in enumerate(np.roots(saddle_coefficients(*coord))):
            saddles_m[i][idx] = s

    saddles.append(saddles_m)
    continue

    saddles_m = saddles[i]

    dd_evaluated = (np.array(
        [abs(dd(s, *grid)) for s in saddles_m]
    ))

    def find_min(arr):
        """return (np.r_[True, arr[1:] < arr[:-1]] &
                np.r_[arr[:-1] < arr[1:], True]).astype(int)"""

        mn = argrelextrema(arr, np.less)
        mn = (np.diff(np.sign(np.diff(arr))) > 0).nonzero()[0] + 1
        arr = arr * 0
        arr[mn] = 1
        return arr

        arr = filters.minimum_filter1d(arr, 5) == arr
        return arr.astype(int)
    results = []
    for k, dd_ in enumerate(dd_evaluated):
        res = np.zeros(grid[0].shape)
        result_col = np.copy((dd_))
        result_row = np.copy((dd_)).T

        for n_col, col in enumerate(result_col):
            result_col[n_col] = find_min(abs(col))

        for n_row, row in enumerate(result_row):
            result_row[n_row] = find_min(abs(row))

        result = result_row + result_col.T
        result[result < 1] = 0
        result[result > 0] = 1
        result[abs(dd_.T)>20] = 0
        res[result>0] = 1
        results += [res]

        idx = res > 0
        height = (value,)*len(idx[idx])
        X[k] += list(grid[0][idx])
        Y[k] += list(grid[1][idx])
        Z[k] += list(height)
        """plt.clf()
        plt.pcolormesh((dd_.T), vmin=-10, vmax=10)
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

    #res *= (i+1)
    #v.append(res)
    """idx = res > 0
    height = (value,)*len(idx[idx])
    X += list(grid[0][idx])
    Y += list(grid[1][idx])
    Z += list(height)"""


fig = plt.gcf()
plt.clf()
ax = fig.add_subplot(111, projection='3d')


def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=2, linewidth=0)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()

def do_scatter(num, cmap=None):
    x = X[num]
    y = Y[num]
    z = Z[num]
    skip = 1
    scatter3d(x[::skip], z[::skip], y[::skip], y[::skip], cmap)

do_scatter(1)

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def do_solid(num, half):
    x = X[num]
    y = Y[num]
    z = Z[num]

    new = np.zeros((len(vals), grid[0].shape[0]))
    new[:] = np.nan

    for x_, y_, z_ in zip(x,y,z):
        idx1 = np.where(grid[0][:,0] == x_)
        idx2 = np.where(vals==z_)
        if np.isnan(new[idx2,idx1]):
            new[idx2,idx1] = y_
        else:
            print 'doppelt'

    N = 24
    def sl1(arr):
        if half is None:
            return arr
        return arr[half*N:(half+1)*N-1,:]
    def sl2(arr):
        if half is None:
            return arr
        return arr[half*N:(half+1)*N-1]

    new = sl1(new)

    for i, row in enumerate(new.T):
        nans, x= nan_helper(row)
        row[nans]= np.interp(x(nans), x(~nans), row[~nans])
        new[:,i] = row

    new_grid = np.meshgrid(grid[0][:,0], sl2(vals))

    test = np.copy(new)
    test[np.isnan(test)] = 0
    norm = Normalize(vmin = np.min(test), vmax = np.max(test), clip = False)

    ax.plot_surface(new_grid[0], new_grid[1], new,
        rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0, antialiased=False, norm=norm)

#do_solid(3,0)
#do_solid(3,1)
do_scatter(3)
#do_solid(1,None)

ax.set_xlabel('A1')
ax.set_ylabel('A3')
ax.set_zlabel('A2')

"""ax.plot_surface(new_grid[0][:, 101:], new_grid[1][:, 101:], new2[:, 101:],
                rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0, antialiased=False, norm=norm)"""


#ax.plot_surface(new_grid[0], new_grid[1], new2)
#plt.pcolormesh(new2)
asd


v2 = np.max(v, axis=0)
v2[v2==0] = np.nan
v2 = pd.DataFrame(v2).interpolate(method='linear', axis=0).values
#v2[np.isnan(v2)] = 0
test = np.copy(v2)
test[np.isnan(test)] = 0
norm = Normalize(vmin = np.min(test), vmax = np.max(test), clip = False)
ax.plot_surface(grid[0][:, 0:100], grid[1][:, 0:100], v2[:, 0:100],
                rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0, antialiased=False, norm=norm)
ax.plot_surface(grid[0][:, 101:], grid[1][:, 101:], v2[:, 101:],
                rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0, antialiased=False, norm=norm)
#ax.plot_surface(grid[0][:, 101:], grid[1][:, 101:], v2[:, 101:])
ax.set_xlabel('A1')
ax.set_ylabel('A2')
ax.set_zlabel('A3')
#clf()
#v[isnan(v)] = 0
#pcolormesh(v)