import copy
import numpy as np
from tqdm import tqdm

from phi.geom import Box
from phi import vis, math
from phi.field import CenteredGrid
from phi.math import extrapolation, tensor
from phi.physics._effect import FieldEffect

from scipy.optimize import minimize
from scipy.optimize import Bounds

from src.env.physics.burgers import Burgers

# *[1] todo: Explain what the GaussianForce function represents here.
# *[2] todo: Explain the cost function
"""
    what is the problem? 
        For a   domain D (of varying sizes [32,64,..]),
                timestep of 0.01, 
                the initial state is sampled from GaussianClashFunction*[1]
        control the evolution of environment, by minimizing the following cost function.*[2]
"""
N = 32
dt = 0.01
visc = 0.003  # viscosity
ph = 40  # prediction horizon
domain_dict = dict(x=N, bounds=Box[-1:1], extrapolation=extrapolation.PERIODIC)
physics = Burgers(default_viscosity=visc)

p = 20  # Prediction horizon on coarse grid
T = 1.0  # New final time
nt2 = round(T / dt) + 1  # number of time steps on coarser grid for SUR
u_min = -10.0  # lower bound for control
u_max = 10.0  # upper bound for control
Q = [1.0, 0.1]  # weights on the diagonal of the Q-matrix in the objective function


# ** what is the initial state? : is defined based on the goal. One could start from a suboptimal initial state and
# learn its way to goal state.
def GaussianClash(x):
    batch_size = N
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


# intial state
y0_pf = CenteredGrid(GaussianClash, **domain_dict)
y0_native = y0_pf.data.native("vector,x")[0]


# ** what is the goal state? : goal of the problem and also decides on how to choose a cost function.

def GaussianForce(x):
    batch_size = N
    loc = np.random.uniform(0.4, 0.6, batch_size)
    amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
    sig = np.random.uniform(0.1, 0.4, batch_size)
    result = tensor(amp, x.shape[0]) * math.exp(
        -0.5 * (x.x.tensor - tensor(loc, x.shape[0])) ** 2 / tensor(sig, x.shape[0]) ** 2)
    return result


# GaussianForce as phiflow.FieldEffect.
g_f_pf = FieldEffect(CenteredGrid(GaussianForce, **domain_dict), ['velocity'])
# calculate ground truth trajectories
gt_trajectories = []
y_gt_pf = copy.deepcopy(y0_pf)
# def calc_gt_trajectories(x0_):
for i in tqdm(range(nt2)):
    y_gt_pf = physics.step(y_gt_pf, dt=dt, effects=(g_f_pf,))
    gt_trajectories.append(y_gt_pf.data.native("vector,x")[0])
gt_traj_np = np.array(gt_trajectories)


# ** What is the cost function? : simply, the weighted difference between trajectories

def J(u_, y_0_native: np.array, y_gt_native: np.ndarray):
    """
        step1 -> update the environment for finite number of prediction horizon in the presence of control element u_
        step2 -> extract target trajectories from y_gt_native of size = prediction horizon
        step3 -> calculate weighted difference
    """

    # step1: update the environment for the prediction horizon p using given control u
    y0_ = copy.deepcopy(y_0_native)
    y_ = [y0_]
    for _ in tqdm(range(p)):
        y00_ = math.tensor(y0_.reshape(y0_pf.data._native.shape), y0_pf.shape)
        y0_ = CenteredGrid(y00_, **domain_dict)
        u0_ = math.tensor(u_.reshape(y0_pf.data._native.shape), y0_pf.shape)
        f_ = FieldEffect(CenteredGrid(u0_, **domain_dict), ['velocity'])
        y0_ = physics.step(y0_, dt=dt, effects=(f_,))
        y0_ = y0_.data.native("vector,x")[0]
        y_.append(y0_)
    y_ = np.array(y_)

    # step2: select the reference trajectories
    y_ref = np.array(y_gt_native[:y_.shape[0]])

    # step3: calculate weighted difference between trajectories
    dy = y_ - y_ref
    dyQ = np.zeros(dy.shape[0], dtype=float)
    # for ii in range(dy.shape[1]):
    for ii in range(1):
        dyQ += Q[ii] * np.power(dy[:, ii], 2)

    return dt * np.sum(dyQ)


# ** MPC loop

# intial guess for first optimization problem
u0 = CenteredGrid(GaussianForce, **domain_dict).data.native("vector,x")[0]
# box constraints u_min <= u <= u_max
bounds = Bounds(u_min * np.ones(N, dtype=float), u_max * np.ones(N, dtype=float))

y_ = copy.deepcopy(y0_native)
u_ = np.zeros((nt2, N))
y_traj = np.zeros((nt2, N))
y_traj[0] = y_
print("shape", u0.shape)
for i in tqdm(range(nt2)):

    # call optimizer
    res = minimize(lambda utmp: J(utmp, y_, gt_traj_np), u0, method='SLSQP', bounds=bounds)

    # retrieve first entry of u and apply it to the plant
    u_[i] = res.x[0]
    # apply above control on environment
    if i < nt2 - 1:
        y_ = physics.step(y_, dt=dt, effects=(u_[i],))
        y_traj[i + 1] = copy.deepcopy(y_)
    # update initial guess
    u0[:-1] = res.x[1:]
    u0[-1] = res.x[-1]

# visualize results
for y in y_traj:
    vis.show(y)