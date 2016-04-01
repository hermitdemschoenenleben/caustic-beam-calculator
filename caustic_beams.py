# -*- coding: utf-8 -*-
"""
Calculate E-field of caustic beams.

Author: Benjamin Wiegand, highwaychile@zoho.com
"""
import sympy as sp
import numpy as np
from time import time
from string import lowercase
from scipy.io import savemat
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
from itertools import combinations
from multiprocessing import Process, Queue
from copy import copy

# ================== CONFIGURATION ==================
# list of tuples with 2 elements each, corresponding to the limits of
# the corresponding axis. The length of the LIMITS list determines whether
# pearcey beam (2 elements), swallowtail (3) or butterfly (4) is calculated.
LIMITS = [
    (-200,200),
    (-200,200),
    100,
]

# save results to file
FILENAME = 'results.mat'

# number of data points for each axis
RESOLUTION = [400,400]
#RESOLUTION = [1920, 1080]

# the maximum value in the plot
PLOT_MAX = 5

# for integration method:
# a grid of N_ROOTS roots is calculated and then interpolated.
# N_ROOTS=[100,100] should be enough, in case of artefacts use more points
N_ROOTS = [100,100]

# for method of steepest descent:
# a grid of N_SADDLES saddle points is calculated and then interpolated.
# For high resolutions you can use a number lower than resolution to speed up
# calculations dramatically, if you see artefacts, use a higher number
# (best results with N_SADDLES == copy(RESOLUTION))
N_SADDLES = copy(RESOLUTION)
#N_SADDLES = [100, 100]

# Which thresholds to use for merging the results of steepest descent method
# and integration method?
# Two methods are used:
#   - minimum saddle distance: if the minimum distance between two saddles is
#     smaller than the first threshold, the point is masked for later
#     re-calculation by means of the integration method
#   - second derivative: if the evaluation of the second derivative at a point
#     is lower than the second threshold for one or more saddles, this point
#     will be calculated using integration method.
THRESHOLDS=0.5, 30

# number of weights for integration. Set this to 0 in order to use the
# continuous mode.
NUM_WEIGHTS = 0

# display integration path in the complex plain
SHOW_INTEGRATION_PATH = False

# the contribution of the integral along the arc is normally very small.
# Use less steps for faster calculation
CALCULATE_ARC_EVERY_N_STEPS = 100

# number of threads to use for calculation of roots (method of steepest descent)
N_THREADS = 4

# neglected integral is proportional to exp(-D)
D = 100

# integration method: if a point changes less than this quantity during two
# iterations, it is regarded as converged
CONVERGENCE_THRESHOLD = 0.001

# =============================================================================

ORDER = len(LIMITS) + 2

try:
    beam = ['Pearcey', 'Swallowtail', 'Butterfly'][ORDER-4]
except IndexError:
    beam = ['Caustic beam of order %d' % ORDER]

print 'CALCULATING %s' % beam

assert (len(LIMITS) == ORDER - 2) and (len(RESOLUTION) == 2), \
    "not the right number of parameters"

# convert integer limits to tuples
LIMITS = [(l,l) if type(l) is int else l for l in LIMITS]

if SHOW_INTEGRATION_PATH:
    RESOLUTION = [2,2]

PLOT_RESOLUTION = np.copy(RESOLUTION)

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


def start():
    if SHOW_INTEGRATION_PATH:
        show_complex_plain()
        plt.set_cmap('gray')
    else:
        plt.set_cmap('jet')

    if not SHOW_INTEGRATION_PATH:
        E_s, saddle_distance, dd_evaluated = steepest_descent()
        mask = ask_for_mask(E_s, saddle_distance, dd_evaluated)
    else:
        E_s = 0
        mask = np.zeros((2,2))

    E_i, new_mask = integration_method(E_s, mask)

    if not SHOW_INTEGRATION_PATH:
        E = plot(E_i, E_s, mask, new_mask)
        dct = copy(locals())
        dct.update(copy(globals()))
        if FILENAME:
            vars_to_save = ['E', 'E_i', 'E_s', 'saddle_distance', 'dd_evaluated',
                'mask', 'LIMITS', 'RESOLUTION', 'N_ROOTS', 'N_SADDLES', 'THRESHOLDS',
                'NUM_WEIGHTS', 'SHOW_INTEGRATION_PATH', 'CALCULATE_ARC_EVERY_N_STEPS',
                'N_THREADS', 'D', 'ORDER']
            savemat(FILENAME, {
                key: dct[key] for key in vars_to_save
            })
    else:
        E = None

    return E, E_i, E_s

# ========================== INTEGRATION METHOD ==========================
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


def integration_method(E_s, mask):
    print 'INTEGRATION METHOD'
    t = time()

    # generate the grids
    grid_fine, range_fine, grid_coarse, range_coarse = generate_grids(N_ROOTS)

    # these variables will contain the integration endpoints
    roots_coarse_pos = np.zeros(grid_coarse[0].shape)
    roots_coarse_neg = np.copy(roots_coarse_pos)

    # calculate a coarse grid of roots (integration endpoints)
    for idx in np.ndindex(grid_coarse[0].shape):
        coord = [g[idx] for g in grid_coarse]
        r_pos, r_neg = get_highest_real_root(coord)

        roots_coarse_pos[idx] = r_pos
        roots_coarse_neg[idx] = r_neg

    # create fine grid for the roots (integration endpoints) by means of
    # interpolation
    roots_fine_pos = interpolate(range_coarse, range_fine, roots_coarse_pos)
    roots_fine_neg = interpolate(range_coarse, range_fine, roots_coarse_neg)

    # filter the grid as well as the matrices for the integration endpoints:
    # only pixels that were not calculated well by method of steepest descent
    # should be included.
    # This generates flat arrays instead of matrices.
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

    # number of iterations
    counter = 0
    # list of s values that were already evaluated
    s_values_used = []

    # index of pixels that did not converge yet
    # is slice(None) at the beginning which means that all pixels are included
    # at the beginning
    not_converged = slice(None)

    # A list containing the result for the E field after each iteration step.
    # This list is used to find out, which pixels did already converge.
    history = []

    # prefactor for integral calculation
    fact_line = np.ones(mat_line.shape)

    try:
        while True:
            n_weights = NUM_WEIGHTS*2**counter
            print 'Calculating %d weights' % n_weights

            # prefactor for integral calculation using `n_weights` weights
            fact_line[not_converged] = 1.0 / n_weights
            fact_arc = fact_line / CALCULATE_ARC_EVERY_N_STEPS

            # the s values where the function should be evaluated
            s_values = np.arange(0, 1, 1.0/n_weights)
            # evaluate the function only at values of s that were not yet
            # evaluated: subtract the s_values_used from s_values
            s_values_new = np.setdiff1d(s_values, s_values_used)
            # number of evaluation points during this iteration
            N = len(s_values_new)

            # the _work variables contain only pixels that did not yet converge
            mat_line_work = mat_line[not_converged]
            mat_arc_pos_work = mat_arc_pos[not_converged]
            mat_arc_neg_work = mat_arc_neg[not_converged]
            roots_fine_pos_work = roots_fine_pos[not_converged]
            roots_fine_neg_work = roots_fine_neg[not_converged]
            grid_fine_work = [g[not_converged] for g in grid_fine]

            for i, s in enumerate(s_values_new):
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
                # beispiel: Integriere von 0 bis positiven R
                # s_val = s * (roots_fine_pos_work)
                # -R bis 0:
                # s_val = -roots_fine_neg_work + s * (roots_fine_neg_work)
                mat_line_work += expr(s_val, *grid_fine_work)

                # use less steps for integration along the arc
                if i % CALCULATE_ARC_EVERY_N_STEPS == 0:
                    # integration from R_pos along the arc
                    s_val_pos = roots_fine_pos_work * np.exp(1j * s * angle_pos)
                    mat_arc_pos_work += expr(s_val_pos, *grid_fine_work)
                    # integration from -R_neg along the arc
                    s_val_neg = -roots_fine_neg_work * np.exp(1j * s * angle_neg)
                    mat_arc_neg_work += expr(s_val_neg, *grid_fine_work)

            if SHOW_INTEGRATION_PATH:
                return

            mat_line[not_converged] = mat_line_work
            mat_arc_pos[not_converged] = mat_arc_pos_work
            mat_arc_neg[not_converged] = mat_arc_neg_work

            # check which pixels converged
            if counter >= 2:
                test = mat_line * fact_line
                not_converged = (abs(test - history[-2]) > CONVERGENCE_THRESHOLD) | \
                                    (abs(test - history[-1]) > CONVERGENCE_THRESHOLD)
                remaining = len(not_converged[not_converged])
                new_mask_flat = np.ones(mat_line.shape)
                new_mask_flat[not_converged] = 0
            else:
                remaining = mat_line.shape[0]
                new_mask_flat = None

            history.append(mat_line*fact_line);

            print '%d pixels remaining' % remaining

            # calculate E field
            """
            Beispiel: Positive half
            E = (
                mat_line * fact_line * (roots_fine_pos) +
                mat_arc_pos * fact_arc * roots_fine_pos * abs(angle_pos) / (2*np.pi)
            )
            """
            E = (
                mat_line * fact_line * (roots_fine_pos + roots_fine_neg) +
                mat_arc_pos * fact_arc * roots_fine_pos * abs(angle_pos) / (2*np.pi) +
                mat_arc_neg * fact_arc * roots_fine_neg * abs(angle_neg) / (2*np.pi)
            )

            plot(E, E_s, mask, new_mask_flat)

            if not CONTINUOUS and not SHOW_INTEGRATION_PATH:
                question = ('Anzahl Gewichte auf %d verdoppeln (j/N)? ' %
                            (NUM_WEIGHTS*2**(counter+1)))
                answer = raw_input(question) not in ['y', 'j']
            else:
                answer = False

            if remaining == 0:
                print "all pixels converged!"
                break

            if SHOW_INTEGRATION_PATH or answer:
                break

            s_values_used += list(s_values)
            counter += 1

    except KeyboardInterrupt:
        if not CONTINUOUS:
            raise KeyboardInterrupt()

    print '%f seconds' % ((time() - t) / 1000)
    return E, new_mask_flat


def steepest_descent():
    print 'METHOD OF STEEPEST DESCENT'
    t = time()

    # generate the grid
    grid_fine, range_fine, grid_coarse, range_coarse = generate_grids(N_SADDLES)

    # a list of ORDER-1 matrices containing the saddle positions
    saddles_coarse = [np.zeros(grid_coarse[0].shape).astype(complex) for i in xrange(ORDER-1)]
    # contains a matrix of the minimum saddle distance
    saddle_distance_coarse = np.zeros(grid_coarse[0].shape)

    # multithreading
    processes = []
    index_list = []
    all_indices = list(np.ndindex(grid_coarse[0].shape))
    per_thread = int(round(len(all_indices) / N_THREADS))

    def calculate_saddles(q, ind_thread, thread_num):
        """
        One thread: calculate saddle points for a list of indices.
        """
        saddles_coarse_t = np.copy(saddles_coarse)
        saddle_distance_coarse_t = np.copy(saddle_distance_coarse)

        N = len(ind_thread)

        for i, idx in enumerate(ind_thread):
            # display percentage
            if thread_num == 0 and i > 0 and i % (N/10) == 0:
                print 'Thread 1: ', str(int(float(i) / N * 100)) + '%'

            coord = [g[idx] for g in grid_coarse]
            # get saddle points
            saddles = np.roots(saddle_coefficients(*coord))
            # sort saddles
            saddles = sorted(saddles, key=lambda x: (np.imag(x), np.real(x), abs(x)))

            for i, saddle in enumerate(saddles):
                saddles_coarse_t[i][idx] = saddle

            # calculate the distance between saddle points pairwise and find the
            # minimal one
            saddle_distance_coarse_t[idx] = min(
                abs(co[0] - co[1]) for co in combinations(saddles, 2)
            )

        q.put([saddles_coarse_t, saddle_distance_coarse_t])

    for thread_num in xrange(N_THREADS):
        # index list for this thead
        index_list.append(
            all_indices[thread_num*per_thread:((thread_num+1)*per_thread - 1)]
        )

        q = Queue()
        p = Process(target=calculate_saddles, args=(q, index_list[-1], thread_num))
        p.start()
        processes.append((q,p))

    # collect results of threads
    for i, qp in enumerate(processes):
        q, p = qp
        saddles_coarse_t, saddle_distance_coarse_t = q.get()

        ind_thread = index_list[i]
        # this loop would be not necessary, but it doesn't work without
        for idx in ind_thread:
            saddle_distance_coarse[idx] = saddle_distance_coarse_t[idx]

            for j, saddle in enumerate(saddles_coarse_t):
                saddles_coarse[j][idx] = saddle[idx]
        p.join()

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
        [abs(dd(s, *grid_fine)) for s in saddles_fine]
    ).min(axis=0)

    def saddle_contribution(saddle_i, grid_fine):
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
    E_s = sum(saddle_contribution(i, grid_fine) for i in xrange(ORDER-1))

    print '%f seconds' % (time() - t)

    return E_s.T, saddle_distance_fine.T, dd_evaluated.T


def generate_grids(coarse_steps):
    # generate fine grids
    range_fine = [np.linspace(*l) for l in PLOT_LIMITS]
    slices = [slice(lim[0], lim[1], res*1j) for lim, res in zip(LIMITS, RESOLUTION)]
    grid_fine = np.squeeze(np.mgrid[slices].astype(np.complex))

    # generade coarse grid for root calculation
    range_coarse = [np.linspace(l[0], l[1], coarse_steps[i]) for i, l in enumerate(PLOT_LIMITS)]
    slices2 = [slice(lim[0], lim[1], res > 1 and (coarse_steps.pop(0)*1j) or 1j) for lim, res in zip(LIMITS, RESOLUTION)]
    grid_coarse = np.squeeze(np.mgrid[slices2].astype(np.float))
    return grid_fine, range_fine, grid_coarse, range_coarse


def interpolate(range_coarse, range_fine, mat):
    f = RectBivariateSpline(range_coarse[0], range_coarse[1], mat)
    return f(range_fine[0], range_fine[1])


# ============================= PLOTS =============================
def plot_single_mat(mat, vmin, vmax, n, title):
    axes = []

    for i, (lim, res) in enumerate(zip(LIMITS, RESOLUTION)):
        if i in PLOT_AXES:
            axes.append(np.linspace(lim[0], lim[1], res))

    plt.figure(n)
    plt.clf()
    plt.pcolormesh(axes[0], axes[1], mat, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.xlim(min(axes[0]), max(axes[0]))
    plt.ylim(min(axes[1]), max(axes[1]))
    plt.xlabel(lowercase[PLOT_AXES[0]])
    plt.ylabel(lowercase[PLOT_AXES[1]])
    plt.title(title)


def plot(E_i, E_s, mask_start, new_mask=None):
    if E_i is not None:
        if len(E_i.shape) == 1:
            # restore original shape of E_i from flat array
            E_i_mat = np.zeros(PLOT_RESOLUTION).astype(complex).T
            E_i_mat[mask_start == 0] = E_i
            E_i = E_i_mat.T

        E_s_mask = np.ma.masked_array(E_s, mask_start.T==0)
        E_i_mask = np.ma.masked_array(E_i, mask_start.T)

        # merge matrices
        E = E_s_mask.filled(0) + E_i_mask.filled(0) * np.exp(1j*np.pi/4)
    else:
        E = E_s

    I = abs(E)**2
    ang = np.angle(E)

    plot_single_mat(I, 0, PLOT_MAX, 1, 'Intensity')
    plot_single_mat(ang, -np.pi, np.pi, 2, 'Phase')

    if new_mask is not None:
        if len(new_mask.shape) == 1:
            # restore original shape of mask from flat array
            new_mask_mat = np.ones(PLOT_RESOLUTION).astype(complex).T
            new_mask_mat[mask_start == 0] = new_mask
            new_mask = new_mask_mat
        plot_single_mat(np.ones(new_mask.T.shape) - new_mask.T, 0, 1, 3,
            'Mask of not converged pixels')
    else:
        plot_single_mat(np.ones(mask_start.T.shape) - mask_start.T, 0, 1, 3,
            'Mask of not converged pixels')

    plt.pause(0.5)

    return E


def show_complex_plain():
    # resolution for commplex plain
    res = 1000

    pos = [l[0] for l in LIMITS]
    root_pos, root_neg = get_highest_real_root(pos)

    # x and y limits for complex plain plot
    xl = (-root_neg - 1, root_pos + 1)
    yl = xl

    # create grid for complex plain
    g = np.mgrid[xl[0]:xl[1]:res*1j, yl[0]:yl[1]:res*1j]
    g = (g[0] + 1j*g[1]).T

    # evaluate function in the complex plain
    res = (np.real(caustic_beam(g, *pos)))

    xran = np.linspace(xl[0],xl[1],1000)
    yran = np.linspace(yl[0],yl[1],1000)

    plt.clf()
    plt.pcolormesh(xran,yran,res,vmin=0,vmax=1)
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.xlim(*xl)
    plt.ylim(*yl)
    plt.colorbar()


def ask_for_mask(E_s, saddle_distance, dd_evaluated):
    while True:
        # pixels == 1 are handled by steepest descent method,
        # pixels == 0 by integration method
        mask = np.ones(PLOT_RESOLUTION)
        # mask pixels with small minimal saddle distance
        mask[saddle_distance < THRESHOLDS[0]] = 0
        # mask pixels with small second derivative
        mask[abs(dd_evaluated) < THRESHOLDS[1]] = 0
        # mask diverging pixels
        mask[abs(E_s)**2 > 5] = 0

        # plot intensity and mask
        plot(None, E_s, mask.T)

        try:
            inp = raw_input('Masken-Schwellwerte %f und %f ok? ' % (THRESHOLDS[0], THRESHOLDS[1]))
            if not inp:
                break
            THRESHOLDS[0], THRESHOLDS[1] = [float(x) for x in inp.split(' ')]
        except:
            print 'Zwei Integer / Floats mit Leerzeichen getrennt eingeben'
            continue

    return mask.T


if __name__ == '__main__':
    E, E_i, E_s = start()