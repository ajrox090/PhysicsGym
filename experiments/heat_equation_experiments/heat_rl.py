import time
from phi.flow import *
from src.env.HeatPhysicsGym import HeatPhysicsGym
from src.util import run_experiment, plotGrid

np.random.seed(43)
N = 1
dxdt = 5
dt = 0.01
dx = 0.25
domain = 3
epoch = 1000
lr = 0.0001
step_count = 200
diffusivity = 2.0
figName = f'results/experiment_heat_rl'

env = HeatPhysicsGym(domain=domain, dx=dx,
                     domain_dict=dict(x=int(domain / dx), bounds=Box[0:1],
                                      extrapolation=extrapolation.BOUNDARY),
                     dt=dt, step_count=step_count,
                     diffusivity=diffusivity, dxdt=dxdt)

agentPath = f'results/ddpgAgent1_heat_{epoch}epochs_{step_count}steps'
# print("train-store")
# x = time.time()
# run_experiment(learn=True, lr=lr, save_model=agentPath, _env=env,
#                agent='ddpg', n_epochs=epoch)
# print(f'{time.time() - x} seconds elapsed for training with {epoch} epochs.')
# print("--------------------------------------------------------------------------------")
print("Test")
x = time.time()
# agentPath = "ddpgAgent_working"
env.step_count = 400
run_experiment(load_model=agentPath, _env=env, agent='ddpg', saveFig=figName, linelabels=True)
print(f'{time.time() - x} seconds elapsed for testing with N={N}.')
print("--------------------------------------------------------------------------------")
