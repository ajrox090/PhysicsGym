import time

from phi.flow import *

from src.env.BurgersPhysicsGym import BurgersPhysicsGym
from src.util import run_experiment

np.random.seed(43)

N = 1
ph = 10
dxdt = 5
dt = 0.01
dx = 0.25
domain = 4
step_count = 400
viscosity = 0.03
figName = f'results/experiment_burgers_mpc'

domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.PERIODIC)
env = BurgersPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                        dt=dt, step_count=step_count,
                        viscosity=viscosity, dxdt=dxdt)
print("Test")
x = time.time()
run_experiment(_env=env, agent='mpc', saveFig=figName, ph=ph, linelabels=True)
print(f'{time.time() - x} seconds elapsed for testing.')
print("--------------------------------------------------------------------------------")
