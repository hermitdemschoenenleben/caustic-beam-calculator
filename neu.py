from string import lowercase
import sympy as sp
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

LIMITS = [
    (-50,50),
    (-50,50),
    (-50,50),
]
"""PLOT_LIMITS = [
    (-50,50),
    (-50,50),
    (-50,50),
]"""
PLOT_LIMITS = LIMITS
RESOLUTION = [300,300,25]
ORDER = 5
AXIS = 2 # XXX: can't change this without breaking code

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


"""X = [[] for i in xrange(ORDER-1)]
Y = [[] for i in xrange(ORDER-1)]
Z = [[] for i in xrange(ORDER-1)]"""

def find_min(arr):
    mn = argrelextrema(arr, np.less)
    mn = (np.diff(np.sign(np.diff(arr))) > 0).nonzero()[0] + 1
    arr = arr * 0
    arr[mn] = 1
    return arr

slices = [slice(lim[0], lim[1], res*1j) for lim, res in zip(LIMITS, RESOLUTION)]
grid = np.squeeze(np.mgrid[slices].astype(np.complex))

"""saddles = [np.zeros(grid[0].shape).astype(complex) for p in xrange(ORDER-1)]

indices = list(np.ndindex(grid[0].shape))
for i, idx in enumerate(indices):
    if i % 100000 == 0:
        print 'step 1: %f %% ' % (float(i) * 100 / len(indices))

    coord = [g[idx] for g in grid]
    for i, s in enumerate(np.roots(saddle_coefficients(*coord))):
        saddles[i][idx] = s"""


for saddle_i, saddle in enumerate(saddles):
    break
    dd_evaluated = np.array(
        abs(dd(saddle, *grid))
    )

    results = []

    for i in xrange(dd_evaluated.shape[AXIS]):
        dd_2d = dd_evaluated[:,:,i] # XXXXXXXXXXXXXXXXX
        value = grid[2][0,0,i]

        res = np.zeros(dd_2d[0].shape)
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

        idx = result > 0

        height = (value,) * len(idx[idx])

        X[saddle_i] += list(grid[0][idx,i])
        Y[saddle_i] += list(grid[1][idx,i])
        Z[saddle_i] += list(height)
        """plt.clf()
        plt.pcolormesh((dd_2d.T))
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
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')


def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=5, linewidth=0)
    scalarMap.set_array(cs)
    #fig.colorbar(scalarMap)
    plt.show()

def do_scatter(num, cmap=None):
    x = np.array(X[num])
    y = np.array(Y[num])
    z = np.array(Z[num])

    def plot_range_filter(*pos):
        pos = list(pos)
        for i, l in enumerate(PLOT_LIMITS):
            idx = (pos[i] > l[0]) & (pos[i] < l[1])
            for j, p in enumerate(pos):
                pos[j] = pos[j][idx]

        return pos

    x, y, z = plot_range_filter(x, y, z)
    skip = 1
    scatter3d(x[::skip], z[::skip], y[::skip], y[::skip], cmap)

#do_scatter(1)

def nan_helper(w):
    return np.isnan(w), lambda q: q.nonzero()[0]

def do_solid(num, interpolate, half, options):
    x = np.array(X[num])
    y = np.array(Y[num])
    z = np.array(Z[num])

    new = np.zeros((grid[0].shape[2], grid[0].shape[0]))
    new[:] = np.nan

    for x_, y_, z_ in zip(x,y,z):
        idx1 = np.where(grid[0][:,0,0] == x_)
        idx2 = np.where(grid[2][0,0,:] == z_)
        test = new[idx2,idx1]
        if np.isnan(test):
            if half is None or ((half == 0) and (y_ > 0)) or ((half == 1) and (y_ < 0)):
                new[idx2,idx1] = y_
        else:
            pass
            #print 'doppelt'


    def sl(arr):
        N = RESOLUTION[AXIS] / 2 - 1

        if half is None:
            return arr
        if len(arr.shape) > 1:
            return arr[half*N:(half+1)*N-1,:]
        else:
            return arr[half*N:(half+1)*N-1]

    #new = sl(new)
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

    new_grid = np.meshgrid(grid[0][:,0,0], grid[2][0,0,:])
    #new = ndimage.filters.gaussian_filter(new, 1, mode='nearest')

    """plt.clf()
    new[np.isnan(new)] = 20
    plt.pcolormesh(new)
    asd"""
    test = np.copy(new)
    test[np.isnan(test)] = 0
    norm = Normalize(vmin = np.min(test), vmax = np.max(test), clip = False)
    ax.plot_surface(new_grid[0], new_grid[1], new, **options)

options = {
    'alpha':        0.5,
    'rstride':      5,
    'cstride':      5,
    'color':        'b',
}


#do_solid(3, True, 0, options)
#do_solid(3, True, 1, options)
"""do_solid(1, False, None, {
    'color':        'r',
    'alpha':        0.5,
})"""

do_scatter(1, 'Reds')
do_scatter(3, 'Blues')

ax.set_xlabel('A1')
ax.set_ylabel('A3')
ax.set_zlabel('A2')