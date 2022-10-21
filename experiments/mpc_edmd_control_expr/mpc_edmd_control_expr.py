import time

import numpy as np
from matplotlib import pyplot as plt
from phi.flow import *
from scipy.optimize import Bounds, minimize
from stable_baselines3 import DDPG, PPO
from tqdm import tqdm

from src.agent.MPCAgent import MPCAgent
from src.agent.RandomAgent import RandomAgent
from src.env.HeatPhysicsGym import HeatPhysicsGym


def plotGrid(listU, domain: int, dx: float, label: list[str]):
    x = np.arange(0, domain, dx)
    plt.tick_params(axis='x', which='minor', length=10)
    plt.grid(True, linestyle='--', which='both')
    for u, lab in zip(listU, label):
        plt.plot(x, u, label=lab)
    plt.xlim(0, domain)
    plt.ylim(-3, 3)
    plt.xlabel("domain")
    plt.ylabel("range")
    plt.legend()
    plt.show()


def run(_env=None, agent=None, title: str = 'Uncontrolled', N: int = 1, extra_step_count: int = 0):
    reward = 0.0
    total_rewards = []
    total_actions = np.zeros((N, _env.step_count), dtype=float)
    finalstateOfEachCycle = []

    # perform cycles
    for i in range(nt2):
        obs = _env.reset()
        done = False
        while not done:
            if agent is None:
                action = [0]
                if env.step_idx < 8:
                    action = [-1.0]
            else:
                action = agent.predict(y_=obs)
                total_actions[i][env.step_idx] = action[0]
            obs, reward, done, info = _env.step(action)

    print(f'Total mean reward for {title} agent: {np.array(total_rewards).mean()}')
    print(f'Actions: \n {total_actions}')
    return np.array(finalstateOfEachCycle).mean(axis=0)


def run_mpc(N: int = 1, ph: int = 10):
    agent_mpc = MPCAgent(ph=ph, u_min=-1.0, u_max=1.0)

    return run(_env=env, agent=agent_mpc, title='MPC', N=N)


dx = 0.5
h = 1e-2
facU = 1
dt = 0.05
domain = 1
N = int(domain / dx)

u_min = -10.0
u_max = 10.0

Q = [1.0, 0.1]

y0 = [1.0, 0.0]


def rhs(y_, u_):
    alpha, beta, delta = -1.0, 1.0, -0.1
    return np.array([y_[1], -delta * y_[1] - alpha * y_[0] - beta * y_[0] * y_[0] * y_[0] + u_])


def ODE(u_, y0_):
    y_ = np.zeros((u_.shape[0], N), dtype=float)
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

    out = dt * np.sum(dyQ)
    print(f"{u_},{out}")
    return out


p = 10
T = 1.0
nt2 = round(T / dt) + 1
t2 = np.linspace(0.0, T, nt2)
# new reference trajectory
y_ref2 = np.zeros((nt2, 2))
y_ref2[np.where(t2 <= 8.0), 0] = -0.5
y_ref2[np.where(t2 <= 6.0), 0] = 0.5
y_ref2[np.where(t2 <= 4.0), 0] = -1.0
y_ref2[np.where(t2 <= 2.0), 0] = 1.0

yI_MPC = np.zeros((nt2, 2))
yI_MPC[0, :] = y0
uI_MPC = np.zeros(nt2)

u0 = 0.5 * (u_max + u_min) * np.ones(p) + 0.1 * \
     (np.random.rand(p) - 0.5) * (u_max - u_min)
bounds = Bounds(u_min * np.ones(p, dtype=float), u_max * np.ones(p, dtype=float))
# MPC loop
t0 = time.time()
for i in range(nt2):

    # determine maximum entry of reference trajectory
    # if ie - i < p, then the remaining entries are
    # constant and identical to the last given one
    ie = np.min([nt2, i + p + 1])

    # call optimizer
    res = minimize(lambda utmp: J_I_MPC(utmp, yI_MPC[i, :], y_ref2[i: ie, :]),
                   u0, method='SLSQP', bounds=bounds)

    # retrieve first entry of u and apply it to the plant
    uI_MPC[i] = res.x[0]
    if i < nt2 - 1:
        yI_MPC[i + 1, :] = Phi(uI_MPC[i], yI_MPC[i, :])

    # update initial guess
    u0[:-1] = res.x[1:]
    u0[-1] = res.x[-1]

tI_MPC = time.time() - t0
print('-> Done in {:.2f} seconds.\n'.format(tI_MPC))

plt.plot(yI_MPC)
plt.show()

# print("MPC")
# mpc_state = run_mpc(N=1, ph=10)
#
# print("visualization")
# plotGrid([mpc_state], domain=domain, dx=dx, label=['mpc state'])
