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

domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.PERIODIC)
env = BurgersPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                        dt=dt, step_count=step_count,
                        viscosity=viscosity, dxdt=dxdt)
print("MPC")
for ph in [5]:
    agentPath = f'results/mpc_burgers_{ph}epochs'
    print("Test")
    x = time.time()
    mpc_state = run(_env=env, agent='mpc', saveFig=agentPath, ph=ph)
    print(f'{time.time() - x} seconds elapsed for testing.')
    print("--------------------------------------------------------------------------------")
