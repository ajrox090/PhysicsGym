import time
from phi.flow import *
from src.env.HeatPhysicsGym import HeatPhysicsGym
from src.util import run_experiment

N = 1
ph = 5
dxdt = 5
dt = 0.01
dx = 0.25
seed = 43
domain = 3
epoch = 10
lr = 0.0001
step_count = 300
diffusivity = 2.0

env = HeatPhysicsGym(domain=domain, dx=dx,
                     domain_dict=dict(x=int(domain / dx), bounds=Box[0:1],
                                      extrapolation=extrapolation.BOUNDARY),
                     dt=dt, step_count=step_count,
                     diffusivity=diffusivity, dxdt=dxdt)


def run_rl():
    agentPath = f'results/ddpgAgent1_dxdt{dxdt}_burgers_{epoch}epochs'
    figName = f'results/experiment_heat_rl'
    print("train-store")
    x = time.time()
    run_experiment(learn=True, lr=lr, save_model=agentPath, _env=env,
                   agent='ddpg', n_epochs=epoch, render=False)
    print(f'{time.time() - x} seconds elapsed for training with {epoch} epochs.')
    print("--------------------------------------------------------------------------------")
    print("Test")
    if seed is not None:
        np.random.seed(seed)
    x = time.time()
    run_experiment(load_model=agentPath, _env=env, agent='ddpg',
                   saveFig=figName, render=False,
                   ymax_states=0.5, ymin_states=-0.1, linelabels=True)
    print(f'{time.time() - x} seconds elapsed for testing with N={N}.')
    print("--------------------------------------------------------------------------------")


def run_baseline():
    if seed is not None:
        np.random.seed(seed)
    figName = f'results/experiment_heat_baseline'
    run_experiment(_env=env, agent='baseline', saveFig=figName, render=False,
                   ymax_states=0.5, ymin_states=-0.1, linelabels=True)


def run_mpc():
    if seed is not None:
        np.random.random(seed)
    figName = f'results/experiment_heat_mpc'
    print("Test")
    x = time.time()
    run_experiment(_env=env, agent='mpc', saveFig=figName, ph=ph,
                   render=False, ymax_states=0.5, ymin_states=-0.1, linelabels=True)
    print(f'{time.time() - x} seconds elapsed for testing.')
    print("--------------------------------------------------------------------------------")


print("RL")
run_rl()

print("Baseline")
run_baseline()

print("MPC")
run_mpc()
