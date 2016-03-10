# -*- coding: utf-8 -*-
"""
Calculate caustic beams.

Author: Benjamin Wiegand, highwaychile@zoho.com
"""
import sympy as sp
import numpy as np
import pickle
from time import time
from string import lowercase
from matplotlib import pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy import interpolate
# spyder always imports the numpy version of sum, but we need the builtin
# version
from __builtin__ import sum

# ================== CONFIGURATION ==================

# the order of the catastrophe
ORDER = 6

# list of tuples with 2 elements each, corresponding to the limits of
# the corresponding axis.
LIMITS = [
    (-150,150),   # a axis
    -10,        # c axis
    (-150,150),   # b axis
    -10,        # d axis
]

# number of data points for each axis
RESOLUTION = [1920,1080]

# number of weights for integration. Set this to 0 in order to use the
# continuous mode.
NUM_WEIGHTS = 0

# a grid of N_ROOTS**2 roots is calculated and then interpolated.
# N_ROOTS=25 should be enough
N_ROOTS = 25

# display integration path in the complex plain for the first coordinate
SHOW_INTEGRATION_PATH = False

# use Legendre-Gauss method? Usually, for high values of NUM_WEIGHTS, False
# is better
GAUSS = False

# in case of pearcey beam:
# should the results of method of steepest descent and integration method be
# merged together? This reduces the number of weights needed.
# Otherwise, only integration method is used.
USE_MASK = True

# the contribution of the integral along the arc is normally very small.
# Use less steps for faster calculation
CALCULATE_ARC_EVERY_N_STEPS = 100

# neglected integral is proportional to exp(-D)
D = 100

# ==================================================

assert (len(LIMITS) == ORDER - 2) and (len(RESOLUTION) == 2), \
    "not the right number of parameters"

# convert integer limits to tuples
LIMITS = [(l,l) if type(l) is int else l for l in LIMITS]

if SHOW_INTEGRATION_PATH:
    RESOLUTION = [2,2]

# expand RESOLUTION to ORDER-2 dimensions
for i, l in enumerate(LIMITS):
    if l[0] == l[1]:
        RESOLUTION.insert(i, 1)

# which axes to plot?
PLOT_AXES = list(np.where(np.array(RESOLUTION) > 1)[0])

# limits of the axes to plot
PLOT_LIMITS = [(LIMITS[i][0], LIMITS[i][1], RESOLUTION[i]) for i in PLOT_AXES]

if NUM_WEIGHTS == 0:
    NUM_WEIGHTS = 10
    CONTINUOUS = True
    assert not GAUSS, 'continuous mode and gauss not possible'
else:
    CONTINUOUS = False

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
caustic_beam = sp.lambdify([S] + var, caustic_beam, 'numexpr')

print '%f seconds' % (time() - t)

# ========================== INTEGRATION METHOD ==========================
def get_gauss_coefficients(N):
    """
    Returns N legendre gauss coefficients.
    use a cache because calculating them for more than 1000 weights is very
    time consuming.
    """
    cache = {}
    try:
        f = file('gauss_cache', 'r')
        cache = pickle.load(f)
        s_values, weights = cache[N]
        f.close()
    except (IOError, KeyError), e:
        f = file('gauss_cache', 'w')
        s_values, weights = leggauss(N)
        cache[N] = (s_values, weights)
        pickle.dump(cache, f)
        f.close()

    # change position range from [-1,1] to [0,1]
    s_values = (s_values + 1) * 0.5
    return s_values, weights


def plot_integration_point(s):
    """
    Place a dot in the complex plain for SHOW_INTEGRATION_PATH.
    """
    # get the first element in every dimension
    while s.shape:
        s = s[0]
    plt.plot(np.real(s),np.imag(s),'o',markersize=4, mfc='red')
    return 0


def get_coeff(pos, neg=False):
    """
    Returns the coefficients of the polynomial used for finding the integration
    limits. Set neg=True for the negative uintegral.
    """
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


def get_highest_real_root(pos):
    roots_pos = np.roots(get_coeff(pos))
    roots_neg = np.roots(get_coeff(pos, True))
    return max(roots_pos[abs(np.imag(roots_pos)) == 0]), \
           max(roots_neg[abs(np.imag(roots_neg)) == 0])


def integration_method():
    print 'INTEGRATION METHOD'
    t = time()

    # generate fine grid
    slices = [slice(lim[0], lim[1], res*1j) for lim, res in zip(LIMITS, RESOLUTION)]
    grid_fine = np.mgrid[slices].astype(np.complex)
    range_fine = [np.linspace(*l) for l in PLOT_LIMITS]

    # generade coarse grid for root calculation
    slices2 = [slice(lim[0], lim[1], res > 1 and (N_ROOTS*1j) or 1j) for lim, res in zip(LIMITS, RESOLUTION)]
    grid_coarse = np.mgrid[slices2].astype(np.float)
    range_coarse = [np.linspace(l[0], l[1], N_ROOTS) for l in PLOT_LIMITS]

    # calculate a coarse grid of roots
    for c in zip(*[np.nditer(coord, op_flags=['readwrite']) for coord in grid_coarse]):
        r_pos, r_neg = get_highest_real_root(c)
        # use first and second coordinate of the grid to save the positive
        # and negative root
        c[0][...] = r_pos
        c[1][...] = r_neg

    roots_coarse = np.squeeze(grid_coarse[0])
    roots_coarse_neg = np.squeeze(grid_coarse[1])

    # the hack above changed grid_coarse
    del grid_coarse

    # create fine grid for the roots by means of interpolation
    f = interpolate.RectBivariateSpline(range_coarse[0], range_coarse[1], roots_coarse)
    f2 = interpolate.RectBivariateSpline(range_coarse[0], range_coarse[1], roots_coarse_neg)
    roots_fine_pos = f(range_fine[0], range_fine[1])
    roots_fine_neg = f2(range_fine[0], range_fine[1])

    # we have to expand the dimension of the grid again
    for i, l in enumerate(LIMITS):
        if i not in PLOT_AXES:
            roots_fine_pos = np.expand_dims(roots_fine_pos, axis=i)
            roots_fine_neg = np.expand_dims(roots_fine_neg, axis=i)

    # results of integration along the real axis
    mat_line = np.zeros(RESOLUTION).astype(np.complex)
    # results of integration along the arcs
    mat_arc_pos = np.copy(mat_line)
    mat_arc_neg = np.copy(mat_line)

    # at which angle is the endpoint of the integration along the arc
    angle_pos = np.pi / (2 * ORDER)
    angle_neg = ((-1) ** ORDER) * np.pi / (2 * ORDER)

    counter = 0
    s_values_used = []

    try:
        while True:
            if GAUSS:
                s_values, weights = get_gauss_coefficients(NUM_WEIGHTS)
                # proper scaling for legendre gauss integration
                fact_line = fact_arc = 0.5
            else:
                n_weights = NUM_WEIGHTS*2**counter
                print 'Calculating %d weights' % n_weights
                fact_line = 1.0 / n_weights
                fact_arc = fact_line / CALCULATE_ARC_EVERY_N_STEPS
                s_values = np.arange(0, 1, 1.0/n_weights)

            # evaluate the function only at values of s that were not yet
            # evaluated
            s_values_new = np.setdiff1d(s_values, s_values_used)
            N = len(s_values_new)

            if not GAUSS:
                weights = (1,) * len(s_values_new)

            for i, s, w in zip(range(N), s_values_new, weights):
                # display percentage
                if i > 0 and i % (N/10) == 0:
                    print str(int(float(i) / N * 100)) + '%'

                if SHOW_INTEGRATION_PATH:
                    # don't evaluate the function but plot the s value in the
                    # complex plain instead
                    expr = lambda x, *args: plot_integration_point(x)
                else:
                    expr = caustic_beam

                # integration from -R_neg to R_pos
                s_val = -roots_fine_neg + s * (roots_fine_pos+roots_fine_neg)
                mat_line += w * expr(s_val, *grid_fine)

                # use less steps for integration along the arc
                if GAUSS or i % CALCULATE_ARC_EVERY_N_STEPS == 0:
                    # integration from R_pos along the arc
                    s_val_pos = roots_fine_pos * np.exp(1j * s * angle_pos)
                    mat_arc_pos += w * expr(s_val_pos, *grid_fine)
                    # integration from -R_neg along the arc
                    s_val_neg = -roots_fine_neg * np.exp(1j * s * angle_neg)
                    mat_arc_neg += w * expr(s_val_neg, *grid_fine)

            if SHOW_INTEGRATION_PATH:
                return

            # calculate E field and remove empty extra dimensions
            E = np.squeeze(
                mat_line * fact_line * (roots_fine_pos + roots_fine_neg) +
                mat_arc_pos * fact_arc * roots_fine_pos * abs(angle_pos) / (2*np.pi) +
                mat_arc_neg * fact_arc * roots_fine_neg * abs(angle_neg) / (2*np.pi)
            ).T

            plot(E, 0, preview=True)
            plt.pause(0.1)

            if not CONTINUOUS and not GAUSS and not SHOW_INTEGRATION_PATH:
                question = ('Anzahl Gewichte auf %d verdoppeln (j/N)? ' %
                            (NUM_WEIGHTS*2**(counter+1)))
                answer = raw_input(question) not in ['y', 'j']
            else:
                answer = False

            if SHOW_INTEGRATION_PATH or GAUSS or answer:
                break

            s_values_used += list(s_values)
            counter += 1

    except KeyboardInterrupt:
        if not CONTINUOUS:
            raise KeyboardInterrupt()

    print '%f seconds' % (time() - t)
    return E


# ======================= METHOD OF STEEPEST DESCENT =======================


def steepest_descent():
    # which axis is symmetric? Speeds up calculations
    SYMMETRY = 0

    print 'METHOD OF STEEPEST DESCENT'
    t = time()
    limits_s = np.copy(LIMITS)
    resolution_s = np.copy(RESOLUTION)

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

    # make use of symmetry
    stack = (PLOT_AXES.index(SYMMETRY) == 0) and np.vstack or np.hstack

    print '%f seconds' % (time() - t)
    return stack((E_s, np.flipud(E_s))).T


# ============================= PLOTS =============================


def plot(E_i, E_s, preview=False):
    # how far away from the caustic should the result of integration method
    # and method of steepest descent be joined?
    CAUSTIC_DISTANCE = 15

    axes = []

    for i, (lim, res) in enumerate(zip(LIMITS, RESOLUTION)):
        if i in PLOT_AXES:
            axes.append(np.linspace(lim[0], lim[1], res))

    if USE_MASK and ORDER == 4 and not preview:
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
        plt.subplot(1, 2, i)
        plt.pcolormesh(axes[0], axes[1], mat, vmax=vmax, vmin=vmin)
        plt.colorbar()
        plt.xlim(min(axes[0]), max(axes[0]))
        plt.ylim(min(axes[1]), max(axes[1]))
        plt.xlabel(lowercase[PLOT_AXES[0]])
        plt.ylabel(lowercase[PLOT_AXES[1]])

    _plot(I, 0, None, 1)
    _plot(ang, -np.pi, np.pi, 2)


def show_complex_plain():
    res = 1000
    pos = [l[0] for l in LIMITS]

    root_pos, root_neg = get_highest_real_root(pos)

    xl = (-root_neg - 1, root_pos + 1)
    yl = xl

    # create complex grid
    g = np.mgrid[xl[0]:xl[1]:res*1j, yl[0]:yl[1]:res*1j]
    g = (g[0] + 1j*g[1]).T

    # evaluate function in the complex plain
    res = (np.real(caustic_beam(g, *pos)))

    xran = np.linspace(xl[0],xl[1],1000)
    yran = np.linspace(yl[0],yl[1],1000)

    plt.clf()
    plt.pcolormesh(xran,yran,res,vmin=0,vmax=1)
    plt.xlim(*xl)
    plt.ylim(*yl)
    plt.colorbar()


if __name__ == '__main__':
    if SHOW_INTEGRATION_PATH:
        show_complex_plain()
        plt.set_cmap('gray')
    else:
        plt.set_cmap('jet')

    E_i = integration_method()

    if not SHOW_INTEGRATION_PATH and ORDER == 4:
        E_s = steepest_descent()
    else:
        E_s = 0

    if not SHOW_INTEGRATION_PATH:
        plot(E_i, E_s)
