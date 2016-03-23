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
from scipy.interpolate import RectBivariateSpline
from itertools import combinations
# spyder always imports the numpy version of sum, but we need the builtin
# version
from __builtin__ import sum

# ================== CONFIGURATION ==================

# the order of the catastrophe
ORDER = 6

# list of tuples with 2 elements each, corresponding to the limits of
# the corresponding axis.
LIMITS = [
    0,
    (-50,50),
    0,
    (-50,50),
]

# number of data points for each axis
#RESOLUTION = [100,100]
RESOLUTION = [400, 400]

# for integration method:
# a grid of N_ROOTS_INT**2 roots is calculated and then interpolated.
# N_ROOTS_INT=100 should be enough
N_ROOTS = 100

# for method of steepest descent:
# a grid of N_ROOTS_STEEP**2 saddle points is calculated and then interpolated.
# if you see artefacts, use a higher number
# (best results with N_ROOTS_STEEP == RESOLUTION)
N_SADDLES = RESOLUTION[0]

# Which THRESHOLDS to use for merging the results of steepest descent method
# and integration method?
# "saddle_distance" and "second_derivative" is possible
# second part of the tuple is a threshold, higher values means that more pixels
# are handled by integration method
#THRESHOLDS='saddle_distance', 0.5
#THRESHOLDS='second_derivative', 10
THRESHOLDS=0.5, 30

# number of weights for integration. Set this to 0 in order to use the
# continuous mode.
NUM_WEIGHTS = 0

# display integration path in the complex plain for the first coordinate
SHOW_INTEGRATION_PATH = False

# use Legendre-Gauss method? Usually, for high values of NUM_WEIGHTS, False
# is better
GAUSS = False

# the contribution of the integral along the arc is normally very small.
# Use less steps for faster calculation
CALCULATE_ARC_EVERY_N_STEPS = 100

# neglected integral is proportional to exp(-D)
D = 100

# the maximum value in the plot
PLOT_MAX = 5
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
    NUM_WEIGHTS = 100
    CONTINUOUS = True
    assert not GAUSS, 'continuous mode and gauss not possible'
else:
    CONTINUOUS = False

THRESHOLDS = list(THRESHOLDS)
# ================== ANALYTICAL CALCULATIONS ==================
print 'ANALYTICAL CALCULATIONS'
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

# generate an integral representation for the caustic beam
caustic_beam = sp.exp(1j * catastrophe)
# "compile for fast computation
caustic_beam = sp.lambdify([S] + var, caustic_beam, 'numexpr')

# coefficients of S in derivative
saddle_coefficients = [sp.lambdify(var, e) for e in sp.Poly(expr_d,S).all_coeffs()]

# "compile" expressions for fast computation
val = sp.lambdify([S] + var, catastrophe, 'numexpr')
d = sp.lambdify([S] + var, expr_d, 'numexpr')
dd = sp.lambdify([S] + var, expr_dd, 'numexpr')
ddd = sp.lambdify([S] + var, expr_ddd, 'numexpr')

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

    # generate the grids
    grid_fine, range_fine, grid_coarse, range_coarse = generate_grids(N_ROOTS)

    roots_coarse_pos = np.zeros(grid_coarse[0].shape)
    roots_coarse_neg = np.copy(roots_coarse_pos)

    # calculate a coarse grid of roots
    for idx in np.ndindex(grid_coarse[0].shape):
        coord = [g[idx] for g in grid_coarse]
        r_pos, r_neg = get_highest_real_root(coord)

        roots_coarse_pos[idx] = r_pos
        roots_coarse_neg[idx] = r_neg

    # create fine grid for the roots by means of interpolation
    roots_fine_pos = interpolate(range_coarse, range_fine, roots_coarse_pos)
    roots_fine_neg = interpolate(range_coarse, range_fine, roots_coarse_neg)

    # create flat arrays instead of matrices in order to only calculate the
    # points included in mask
    idx = mask == 0
    shape = len(mask[idx])
    roots_fine_neg = roots_fine_neg[idx]
    roots_fine_pos = roots_fine_pos[idx]
    grid_fine = [g[idx] for g in grid_fine]

    # results of integration along the real axis
    mat_line = np.zeros(shape).astype(np.complex)
    # results of integration along the arcs
    mat_arc_pos = np.copy(mat_line)
    mat_arc_neg = np.copy(mat_line)

    # at which angle is the endpoint of the integration along the arc
    angle_pos = np.pi / (2 * ORDER)
    angle_neg = ((-1) ** ORDER) * np.pi / (2 * ORDER)

    counter = 0
    s_values_used = []

    # index of pixels that did not converge yet
    not_converged = slice(None)
    last = []

    fact_line = np.ones(mat_line.shape)

    try:
        while True:
            if GAUSS:
                s_values, weights = get_gauss_coefficients(NUM_WEIGHTS)
                # proper scaling for legendre gauss integration
                fact_line = fact_arc = 0.5
            else:
                n_weights = NUM_WEIGHTS*2**counter
                print 'Calculating %d weights' % n_weights
                fact_line[not_converged] = 1.0 / n_weights

                fact_arc = fact_line / CALCULATE_ARC_EVERY_N_STEPS
                s_values = np.arange(0, 1, 1.0/n_weights)

            #
            mat_line_work = np.copy(mat_line[not_converged])
            mat_arc_pos_work = np.copy(mat_arc_pos[not_converged])
            mat_arc_neg_work = np.copy(mat_arc_neg[not_converged])
            roots_fine_pos_work = np.copy(roots_fine_pos[not_converged])
            roots_fine_neg_work = np.copy(roots_fine_neg[not_converged])
            grid_fine_work = [np.copy(g[not_converged]) for g in grid_fine]

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
                s_val = -roots_fine_neg_work + s * (roots_fine_pos_work+roots_fine_neg_work)
                mat_line_work += w * expr(s_val, *grid_fine_work)

                # use less steps for integration along the arc
                if GAUSS or i % CALCULATE_ARC_EVERY_N_STEPS == 0:
                    # integration from R_pos along the arc
                    s_val_pos = roots_fine_pos_work * np.exp(1j * s * angle_pos)
                    mat_arc_pos_work += w * expr(s_val_pos, *grid_fine_work)
                    # integration from -R_neg along the arc
                    s_val_neg = -roots_fine_neg_work * np.exp(1j * s * angle_neg)
                    mat_arc_neg_work += w * expr(s_val_neg, *grid_fine_work)

            if SHOW_INTEGRATION_PATH:
                return

            mat_line[not_converged] = mat_line_work
            mat_arc_pos[not_converged] = mat_arc_pos_work
            mat_arc_neg[not_converged] = mat_arc_neg_work

            threshold = 0.001

            if counter >= 2:
                test = mat_line * fact_line
                not_converged = (abs(test - last[-2]) > threshold) | \
                                    (abs(test - last[-1]) > threshold)
                remaining = len(not_converged[not_converged])
            else:
                remaining = mat_line.shape[0]

            last.append(mat_line*fact_line);

            print '%d pixels remaining' % remaining

            # calculate E field and remove empty extra dimensions
            E = (
                mat_line * fact_line * (roots_fine_pos + roots_fine_neg) +
                mat_arc_pos * fact_arc * roots_fine_pos * abs(angle_pos) / (2*np.pi) +
                mat_arc_neg * fact_arc * roots_fine_neg * abs(angle_neg) / (2*np.pi)
            )

            plot(E, E_s)
            plt.pause(0.2)

            if not CONTINUOUS and not GAUSS and not SHOW_INTEGRATION_PATH:
                question = ('Anzahl Gewichte auf %d verdoppeln (j/N)? ' %
                            (NUM_WEIGHTS*2**(counter+1)))
                answer = raw_input(question) not in ['y', 'j']
            else:
                answer = False

            if remaining == 0:
                print "all pixels converged!"
                break

            if SHOW_INTEGRATION_PATH or GAUSS or answer:
                break

            s_values_used += list(s_values)
            counter += 1

    except KeyboardInterrupt:
        if not CONTINUOUS:
            raise KeyboardInterrupt()

    print '%f seconds' % ((time() - t) / 1000)
    return E


def steepest_descent():
    print 'METHOD OF STEEPEST DESCENT'
    t = time()

    # generate the grid
    grid_fine, range_fine, grid_coarse, range_coarse = generate_grids(N_SADDLES)

    # a list of ORDER-1 matrices containing the saddle positions
    saddles_coarse = [np.zeros(grid_coarse[0].shape).astype(complex) for i in xrange(ORDER-1)]
    # contains a matrix of the minimum saddle distance
    saddle_distance_coarse = np.zeros(grid_coarse[0].shape)

    # calculate a coarse grid of saddle point positions
    for idx in np.ndindex(grid_coarse[0].shape):
        coord = [g[idx] for g in grid_coarse]
        # get saddle points
        saddles = np.roots([coeff(*coord) for coeff in saddle_coefficients])
        # sort saddles
        saddles = sorted(saddles, key=lambda x: (np.imag(x), np.real(x), abs(x)))

        for i, saddle in enumerate(saddles):
            saddles_coarse[i][idx] = saddle

        # calculate the distance between saddle points pairwise and find the
        # minimal one
        saddle_distance_coarse[idx] = min(
            abs(co[0] - co[1]) for co in combinations(saddles, 2)
        )

    # interpolate coarse matrices
    saddle_distance_fine = interpolate(range_coarse, range_fine, saddle_distance_coarse)
    saddles_fine = [
        interpolate(range_coarse, range_fine, np.real(s)) +
        1j * interpolate(range_coarse, range_fine, np.imag(s))
        for s in saddles_coarse
    ]

    # evaluate the second derivative as criterium whether saddle approximation
    # is valid
    dd_evaluated = np.array(
        [abs(np.squeeze(dd(r, *grid_fine))) for r in saddles_fine]
    ).min(axis=0)

    def _saddle_contribution(saddle_i, grid_fine):
        """
        Calculates the contribution to the E-field of each saddle.
        """
        value = val(saddles_fine[saddle_i], *grid_fine)
        deriv = dd(saddles_fine[saddle_i], *grid_fine)

        ang = np.angle(deriv)
        theta_ = (np.pi - ang) / 2

        # smoothen the angles (otherwise you can see sharp jumps from -pi to pi)
        theta_ = np.arcsin(np.sin(theta_))

        # apply method of steepest descent
        ret = np.exp(1j * (value + theta_)) * np.sqrt(2*np.pi/np.abs(deriv))

        # ignore contributions of imaginary saddles
        ret[abs(np.imag(saddles_fine[saddle_i])) > 0.01] = 0
        ret[np.isnan(ret)] = 0
        return ret

    # calculate E field as sum of saddle contributions
    E_s = sum(_saddle_contribution(i, grid_fine) for i in xrange(ORDER-1))

    print '%f seconds' % (time() - t)

    return E_s.T, saddle_distance_fine.T, dd_evaluated.T


def generate_grids(coarse_steps):
    # generate fine grids
    slices = [slice(lim[0], lim[1], res*1j) for lim, res in zip(LIMITS, RESOLUTION)]
    grid_fine = np.squeeze(np.mgrid[slices].astype(np.complex))
    range_fine = [np.linspace(*l) for l in PLOT_LIMITS]

    # generade coarse grid for root calculation
    slices2 = [slice(lim[0], lim[1], res > 1 and (coarse_steps*1j) or 1j) for lim, res in zip(LIMITS, RESOLUTION)]
    grid_coarse = np.squeeze(np.mgrid[slices2].astype(np.float))
    range_coarse = [np.linspace(l[0], l[1], coarse_steps) for l in PLOT_LIMITS]

    return grid_fine, range_fine, grid_coarse, range_coarse


def interpolate(range_coarse, range_fine, mat):
    f = RectBivariateSpline(range_coarse[0], range_coarse[1], mat)
    return f(range_fine[0], range_fine[1])


# ============================= PLOTS =============================
def _plot(mat, vmin, vmax, n):
    axes = []

    for i, (lim, res) in enumerate(zip(LIMITS, RESOLUTION)):
        if i in PLOT_AXES:
            axes.append(np.linspace(lim[0], lim[1], res))

    plt.subplot(1, 2, n)
    plt.pcolormesh(axes[0], axes[1], mat, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.xlim(min(axes[0]), max(axes[0]))
    plt.ylim(min(axes[1]), max(axes[1]))
    plt.xlabel(lowercase[PLOT_AXES[0]])
    plt.ylabel(lowercase[PLOT_AXES[1]])


def plot(E_i_flat, E_s):
    # restore original shape of E_i from flat array
    E_i = np.zeros(E_s.shape).astype(complex).T
    E_i[mask == 0] = E_i_flat
    E_i = E_i.T

    E_s_mask = np.ma.masked_array(E_s, mask.T==0)
    E_i_mask = np.ma.masked_array(E_i, mask.T)

    # merge matrices
    E = E_s_mask.filled(0) + E_i_mask.filled(0) * np.exp(1j*np.pi/4)

    I = abs(E)**2
    ang = np.angle(E)

    plt.clf()

    _plot(I, 0, PLOT_MAX, 1)
    _plot(ang, -np.pi, np.pi, 2)

    return E


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


def get_mask():
    while True:
        # pixels == 1 are handled by steepest descent method,
        # pixels == 0 by integration method
        mask = np.ones(E_s.shape)
        # mask pixels with small minimal saddle distance
        mask[saddle_distance < THRESHOLDS[0]] = 0
        # mask pixels with small second derivative
        mask[abs(dd_evaluated) < THRESHOLDS[1]] = 0
        # mask diverging pixels
        mask[abs(E_s)**2 > 10] = 0

        # plot intensity and mask
        plt.clf()
        _plot(abs(E_s)**2, 0, PLOT_MAX, 1)
        _plot(mask, 0, 1, 2)
        plt.pause(0.5)

        try:
            inp = raw_input('Werte %f und %f ok? ' % (THRESHOLDS[0], THRESHOLDS[1]))
            if not inp:
                break
            THRESHOLDS[0], THRESHOLDS[1] = [float(x) for x in inp.split(' ')]
        except:
            print 'Zwei Werte mit Leerzeichen getrennt eingeben'
            continue

    return mask.T


if __name__ == '__main__':
    if SHOW_INTEGRATION_PATH:
        show_complex_plain()
        plt.set_cmap('gray')
    else:
        plt.set_cmap('jet')

    if not SHOW_INTEGRATION_PATH:
        E_s, saddle_distance, dd_evaluated = steepest_descent()
        mask = get_mask()
    else:
        E_s = 0

    E_i = integration_method()

    if not SHOW_INTEGRATION_PATH:
        E = plot(E_i, E_s)
