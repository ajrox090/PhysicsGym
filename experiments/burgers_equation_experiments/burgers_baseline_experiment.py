from phi.flow import *

from src.agent.BaselineAgent import BaselineAgent
from src.env.BurgersPhysicsGym import BurgersPhysicsGym
from src.util import plotGrid

dxdt = 5
dt = 0.01
dx = 0.25
domain = 4
step_count = 100
viscosity = 0.03
N = int(domain / dx)

domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.PERIODIC)
env = BurgersPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                        dt=dt, step_count=step_count,
                        viscosity=viscosity, dxdt=dxdt,
                        xlim=0, ylim=2.0)

observation = env.reset()
statesList = []
statelabels = []
actionList = []
actionLabels = []
agent = BaselineAgent(env=env, u_min=-1.0, u_max=1.0)
for _ in range(step_count):
    actions = agent.predict(observation)
    # if _ % ((step_count - 1) // 5) == 0:
    #     env.enable_rendering()
    observation, reward, done, info = env.step(actions)
    # if _ % ((step_count - 1) // 3) == 0:
    if _ in [1, 15, 30, 60, 99]:
        statesList.append(observation)
        statelabels.append(f't={env.step_idx/100}')
        actionList.append(env.forces.field.data.numpy("vector,x")[0])
        actionLabels.append(f't={env.step_idx/100}')
    # if env._render:
    #     env.disable_rendering()

# env.disable_rendering()
plotGrid(listU=statesList, domain=domain, dx=dx, label=statelabels,
         ylim_min=-2.0, ylim_max=2.0)
plotGrid(listU=actionList, domain=domain, dx=dx, label=actionLabels,
         xlim_max=1.5, ylim_max=1.0, ylim_min=-1.0)
