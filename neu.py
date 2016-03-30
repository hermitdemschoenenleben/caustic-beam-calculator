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
    0,
]
ORDER = 5

# convert integer limits to tuples
LIMITS = [(l,l) if type(l) is int else l for l in LIMITS]

RESOLUTION = [400,400]
# expand RESOLUTION to ORDER-2 dimensions
for i, l in enumerate(LIMITS):
    if l[0] == l[1]:
        RESOLUTION.insert(i, 1)

fig = plt.gcf()
plt.clf()
ax = fig.add_subplot(111, projection='3d')


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

# generate an integral representation for the caustic beam
caustic_beam = sp.exp(1j * catastrophe)
# "compile for fast computation
caustic_beam = sp.lambdify([S] + var, caustic_beam, 'numexpr')

# coefficients of S in derivative
saddle_coefficients = sp.lambdify(var, sp.Poly(expr_d,S).all_coeffs(), 'numpy')

# "compile" expressions for fast computation
val = sp.lambdify([S] + var, catastrophe, 'numexpr')
d = sp.lambdify([S] + var, expr_d, 'numexpr')
dd = sp.lambdify([S] + var, expr_dd, 'numexpr')
ddd = sp.lambdify([S] + var, expr_ddd, 'numexpr')

#vals = [-20, 0, 20]
vals = np.linspace(-100,100,50)


#saddles = []
"""X = []
Y = []
Z = []
"""

for i, value in enumerate(vals):
    if i < 20:
        continue
    LIMITS[2] = (value, value)
    slices = [slice(lim[0], lim[1], res*1j) for lim, res in zip(LIMITS, RESOLUTION)]
    grid = np.squeeze(np.mgrid[slices].astype(np.complex))

    """saddles_m = [np.zeros(grid[0].shape).astype(complex) for p in xrange(ORDER-1)]
    print i

    test = np.zeros(grid[0].shape)
    for idx in np.ndindex(grid[0].shape):
        coord = [g[idx] for g in grid]
        for i, s in enumerate(np.roots(saddle_coefficients(*coord))):
            saddles_m[i][idx] = s

    saddles.append(saddles_m)
    continue"""

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

    res = np.zeros(grid[0].shape)
    for k, dd_ in enumerate(dd_evaluated):
        result_col = np.copy((dd_))
        result_row = np.copy((dd_)).T

        for n_col, col in enumerate(result_col):
            result_col[n_col] = find_min(abs(col))

        for n_row, row in enumerate(result_row):
            result_row[n_row] = find_min(abs(row))

        result = result_row + result_col.T
        result[result < 1] = 0
        result[result > 0] = 1
        result[abs(dd_.T)>10] = 0
        res[result>0] = 1
        plt.clf()
        plt.pcolormesh((dd_.T), vmin=-10, vmax=10)
        plt.colorbar()
        plt.pause(0.1)
        raw_input('')
        plt.pcolormesh(result)
        plt.pause(0.1)
        raw_input('')
    """plt.clf()
    plt.pcolormesh(res)
    plt.pause(0.1)
    raw_input(value)
    continue"""
    #result = ndimage.filters.gaussian_filter(result, 1, mode='nearest')

    #res *= (i+1)
    #v.append(res)
    idx = res > 0
    height = (value,)*len(idx[idx])
    X += list(grid[0][idx])
    Y += list(grid[1][idx])
    Z += list(height)

def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=2, linewidth=0)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()
print '!PLOT'
skip = 1
scatter3d(X[::skip], Z[::skip], Y[::skip], Y[::skip])

ax.set_xlabel('A1')
ax.set_ylabel('A3')
ax.set_zlabel('A2')

new = np.zeros((len(vals), grid[0].shape[0]))
new[:] = np.nan

for x, y, z in zip(X, Y, Z):
    idx1 = np.where(grid[0][:,0] == x)
    idx2 = np.where(abs(vals-z)<0.001)
    if np.isnan(new[idx2,idx1]) or new[idx2,idx1] < y:
        new[idx2,idx1] = y

#plt.clf()
new2 = pd.DataFrame(new).interpolate(method='linear', axis=0).values
#new2[np.isnan(new2)] = 0

new_grid = np.meshgrid(grid[0][:,0], vals)

test = np.copy(new2)
test[np.isnan(test)] = 0
norm = Normalize(vmin = np.min(test), vmax = np.max(test), clip = False)

ax.plot_surface(new_grid[0][25:,:], new_grid[1][25:,:], new2[25:,:],
                rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0, antialiased=False, norm=norm)
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