import time

from phi.flow import *
from src.env.BurgersPhysicsGym import BurgersPhysicsGym
from src.util import run_experiment as run

np.random.seed(43)

dxdt = 5
dt = 0.01
dx = 0.25
domain = 4
step_count = 200
viscosity = 0.03
N = int(domain / dx)
lr = 0.0001

domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.PERIODIC)
env = BurgersPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                        dt=dt, step_count=step_count,
                        viscosity=viscosity, dxdt=dxdt)
print("DDPG")
for epoch in [10]:
    agentPath = f'results/ddpgAgent1_dxdt{dxdt}_burgers_{epoch}epochs'
    # print("train-store")
    # x = time.time()
    # run(learn=True, save_model=agentPath, _env=env, agent='ddpg', n_epochs=epoch, render=False)
    # print(f'{time.time() - x} seconds elapsed for training with {epoch} epochs.')
    # print("--------------------------------------------------------------------------------")
    print("Test")
    x = time.time()
    ddpg_state = run(load_model=agentPath, _env=env, agent='ddpg', saveFig=agentPath)
    print(f'{time.time() - x} seconds elapsed for testing with N={N}.')
    print("--------------------------------------------------------------------------------")
