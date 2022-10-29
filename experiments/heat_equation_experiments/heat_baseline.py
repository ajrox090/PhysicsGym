from phi.flow import *
from src.env.HeatPhysicsGym import HeatPhysicsGym
from src.util import run_experiment

np.random.seed(43)

N = 1
dxdt = 5
dt = 0.01
dx = 0.25
domain = 3
step_count = 400
diffusivity = 2.0
figName = f'results/experiment_heat_baseline'

env = HeatPhysicsGym(domain=domain, dx=dx, dt=dt, step_count=step_count, diffusivity=diffusivity, dxdt=dxdt,
                     domain_dict=dict(x=int(domain / dx), bounds=Box[0:1], extrapolation=extrapolation.BOUNDARY))

run_experiment(_env=env, agent='baseline', saveFig=figName, render=False, linelabels=True)
