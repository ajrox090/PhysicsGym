import time

from phi.flow import *
from src.env.BurgersPhysicsGym import BurgersPhysicsGym
from src.util import run_experiment


dxdt = 5
dt = 0.01
dx = 0.25
domain = 4
step_count = 200
viscosity = 0.03
N = 1
lr = 0.0001
epochs = 100
figName = f'results/experiment_burgers_rl'

domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.PERIODIC)
env = BurgersPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                        dt=dt, step_count=step_count,
                        viscosity=viscosity, dxdt=dxdt)
agentPath = f'results/ddpgAgent1_burgers_{epochs}epochs_{step_count}steps'
print("Train-Store")
x = time.time()
run_experiment(learn=True, lr=lr, save_model=agentPath, _env=env,
               agent='ddpg', n_epochs=epochs)
print(f'{time.time() - x} seconds elapsed for training with {epochs} epochs.')
print("--------------------------------------------------------------------------------")
print("Test")
np.random.random(43)
x = time.time()
env.step_count = 400
run_experiment(load_model=agentPath, _env=env, agent='ddpg', saveFig=figName, linelabels=True)
print(f'{time.time() - x} seconds elapsed for testing with N={N}.')
print("--------------------------------------------------------------------------------")
