import copy

import numpy as np
import time

from phi import vis, math
from phi.field import CenteredGrid
from phi.geom import Box
from phi.math import extrapolation, tensor
from phi.physics._effect import FieldEffect
from scipy.optimize import minimize
from scipy.optimize import Bounds
from tqdm import tqdm

from src.env.physics.burgers import Burgers

h = 1e-2  # Time step of integrator
facU = 1  # Factor by which dt is larger
dt = facU * h  # Coarse time step for SUR
T = 1.0  # Final time
dimY = 2  # dimension of state space

u_min = -10.0  # lower bound for control
u_max = 10.0  # upper bound for control

V = [u_min, u_max]  # finite set of admissible controls in (II) and (III)
nu = len(V)  # dimension of V

Q = [1.0, 0.1]  # weights on the diagonal of the Q-matrix in the objective function

y0 = [1.0, 0.0]  # initial condition for y

nt = round(T / h) + 1  # number of time steps on fine grid
nt2 = round(T / dt) + 1  # number of time steps on coarser grid for SUR

t = np.linspace(0.0, T, nt)  # array of time steps (fine grid)
t2 = np.linspace(0.0, T, nt2)  # array of time steps (coarse grid)

u0 = np.zeros(nt, )  # initial guess for control u

y_ref = np.zeros((nt, 2))  # reference trajectory on fine grid
y_ref2 = y_ref[::facU, :]  # reference trajectory on coarse grid


def rhs(y_, u_):
    alpha, beta, delta = -1.0, 1.0, -0.1
    return np.array([y_[1], -delta * y_[1] - alpha * y_[0] - beta * y_[0] * y_[0] * y_[0] + u_])


def ODE(u_, y0_):
    y_ = np.zeros((u_.shape[0], dimY), dtype=float)
    y_[0, :] = y0_
    for ii in range(u_.shape[0] - 1):
        k1 = rhs(y_[ii, :], u_[ii])
        k2 = rhs(y_[ii, :] + h / 2 * k1[:], u_[ii])
        k3 = rhs(y_[ii, :] + h / 2 * k2[:], u_[ii])
        k4 = rhs(y_[ii, :] + h * k3[:], u_[ii])
        y_[ii + 1, :] = y_[ii, :] + h / 6 * (k1[:] + 2 * k2[:] + 2 * k3[:] + k4[:])
    return y_


def Phi(u_, y0_):
    # Integration with constant input over one time step of the coarse grid
    u2 = u_ * np.ones((facU + 1), dtype=float)
    y_ = ODE(u2, y0_)
    return y_[-1, :]


def coarseGridToFine(x_):
    if facU == 1:
        return x_
    y_ = np.zeros(nt, dtype=float)
    for ii in range(nt2 - 1):
        y_[facU * ii: facU * (ii + 1)] = x_[ii]
    y_[-1] = x_[-1]
    return y_


p = 20  # Prediction horizon on coarse grid
T = 10.0  # New final time
nt2 = round(T / dt) + 1  # number of time steps on coarser grid for SUR
t2 = np.linspace(0.0, T, nt2)  # array of time steps (coarse grid)

# new reference trajectory
y_ref2 = np.zeros((nt2, 2))
y_ref2[np.where(t2 <= 8.0), 0] = -0.5
y_ref2[np.where(t2 <= 6.0), 0] = 0.5
y_ref2[np.where(t2 <= 4.0), 0] = -1.0
y_ref2[np.where(t2 <= 2.0), 0] = 1.0

print('Solve (I) via MPC with T = {:.1f} ...'.format(T))


# Redefine J_I with shorter reference trajectory
def J_I_MPC(u_, y0_, y_ref_):
    # calculate trajectory using the time-T-map Phi
    y_ = np.zeros((p + 1, 2))
    y_[0, :] = y0_
    for ii in range(p):
        y_[ii + 1, :] = Phi(u_[ii], y_[ii, :])

    # fill up reference trajectory if necessary
    y_ref_2 = np.zeros(y_.shape)
    y_ref_2[:y_ref_.shape[0], :] = y_ref_

    # calculate weighted difference between trajectories
    dy = y_ - y_ref_2
    dyQ = np.zeros(dy.shape[0], dtype=float)
    for ii in range(dy.shape[1]):
        dyQ += Q[ii] * np.power(dy[:, ii], 2)

    return dt * np.sum(dyQ)


def GaussianClash(x):
    batch_size = 32
    leftloc = np.random.uniform(0.2, 0.4, batch_size)
    leftamp = np.random.uniform(0, 3, batch_size)
    leftsig = np.random.uniform(0.05, 0.15, batch_size)
    rightloc = np.random.uniform(0.6, 0.8, batch_size)
    rightamp = np.random.uniform(-3, 0, batch_size)
    rightsig = np.random.uniform(0.05, 0.15, batch_size)
    left = tensor(leftamp, x.shape[0]) * math.exp(
        -0.5 * (x.x.tensor - tensor(leftloc, x.shape[0])) ** 2 / tensor(leftsig, x.shape[0]) ** 2)
    right = tensor(rightamp, x.shape[0]) * math.exp(
        -0.5 * (x.x.tensor - tensor(rightloc, x.shape[0])) ** 2 / tensor(rightsig, x.shape[0]) ** 2)
    result = left + right
    return result


def GaussianForce(x):
    batch_size = 32
    loc = np.random.uniform(0.4, 0.6, batch_size)
    amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
    sig = np.random.uniform(0.1, 0.4, batch_size)
    result = tensor(amp, x.shape[0]) * math.exp(
        -0.5 * (x.x.tensor - tensor(loc, x.shape[0])) ** 2 / tensor(sig, x.shape[0]) ** 2)
    return result


# initialize arrays for MPC solution
yI_MPC = np.zeros((nt2, 2))
yI_MPC[0, :] = y0
uI_MPC = np.zeros(nt2)

# initial guess for first optimization problem
u0 = 0.5 * (u_max + u_min) * np.ones(p) + 0.1 * (np.random.rand(p) - 0.5) * (u_max - u_min)
N = 32
viscosity = 0.01 / (N * np.pi)
domain_dict = dict(x=N, bounds=Box[-1:1], extrapolation=extrapolation.PERIODIC)
physics = Burgers(default_viscosity=viscosity)
y_initial = CenteredGrid(GaussianClash, **domain_dict)
forces = FieldEffect(CenteredGrid(GaussianForce, **domain_dict), ['velocity'])
# calculate reference trajectories for minimizer
gt_y = copy.deepcopy(y_initial)
y_reference = [gt_y]
print("calculating reference trajectories\n")
for i in tqdm(range(nt2)):
    gt_y = physics.step(gt_y, effects=(forces,))
    y_reference.append(gt_y)


# cost function, weighted difference between trajectories
# p is the prediction horizon
def J(u_, y0_, y_ref_):

    # step1: update the environment for the prediction horizon p using given control u
    y_ = [y0_]
    for _ in range(p):
        y0_ = CenteredGrid(y0_, **domain_dict)
        f_ = FieldEffect(CenteredGrid(u_, **domain_dict), ['velocity'])
        y0_ = physics.step(y0_, dt=dt, effects=(f_,))
        y_.append(y0_.data.native("vector,x")[0])
    y_ = np.array(y_)

    # step2: select the reference trajectories
    y_ref_2 = np.array(y_ref_[:y_.shape[0]])

    # step3: calculate weighted difference between trajectories
    dy = y_ - y_ref_2
    dyQ = np.zeros(dy.shape[0], dtype=float)
    for ii in range(dy.shape[1]):
        dyQ += Q[ii] * np.power(dy[:, ii], 2)

    return dt * np.sum(dyQ)


# box constraints u_min <= u <= u_max
bounds = Bounds(u_min * np.ones(p, dtype=float), u_max * np.ones(p, dtype=float))
# MPC loop
print("Starting MPC loop\n")
for i in tqdm(range(nt2)):

    # determine maximum entry of reference trajectory
    # if ie - i < p, then the remaining entries are
    # constant and identical to the last given one
    ie = np.min([nt2, i + p + 1])
    ny_init = y_initial.data.native("vector,x")[0]
    ny_ref = y_reference[i].data.native("vector,x")[0]
    # call optimizer
    res = minimize(lambda utmp: J_I_MPC(utmp, ny_init, ny_ref), u0, method='SLSQP', bounds=bounds)

    # retrieve first entry of u and apply it to the plant
    uI_MPC[i] = res.x[0]
    if i < nt2 - 1:
        yI_MPC[i + 1, :] = Phi(uI_MPC[i], yI_MPC[i, :])

    # update initial guess
    u0[:-1] = res.x[1:]
    u0[-1] = res.x[-1]