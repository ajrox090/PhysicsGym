from phi.flow import *
from src.env.BurgersPhysicsGym import BurgersPhysicsGym
from src.util import run_experiment

np.random.seed(43)


dxdt = 5
dt = 0.01
dx = 0.25
domain = 4
step_count = 400
viscosity = 0.03
N = int(domain / dx)

domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.PERIODIC)
env = BurgersPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                        dt=dt, step_count=step_count,
                        viscosity=viscosity, dxdt=dxdt)

figName = f'results/experiment_burgers_uncontrolled'
run_experiment(_env=env, render=False, saveFig=figName, linelabels=True)
