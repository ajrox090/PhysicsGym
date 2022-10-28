from phi.flow import *
from src.env.BurgersPhysicsGym import BurgersPhysicsGym
from src.util import plotGrid

dxdt = 5
dt = 0.01
dx = 0.25
domain = 4
step_count = 100
viscosity = 0.003
N = int(domain / dx)

domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.PERIODIC)
env = BurgersPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                        dt=dt, step_count=step_count,
                        viscosity=viscosity, dxdt=dxdt,
                        xlim=0, ylim=2.0)

observation = env.reset()
# env.enable_rendering()
results = []
labels = []
for _ in range(step_count):
    if _ < 0:
        a = -1.0
    else:
        a = 0.0
    actions = np.array([a])
    # if _ % ((step_count - 1) // 5) == 0:
    #     env.enable_rendering()
    if _ % ((step_count - 1) // 3) == 0:
        results.append(observation)
        labels.append(f'state at t={env.step_idx/100}')
    observation, reward, done, info = env.step(actions)
    # if env._render:
    #     env.disable_rendering()

# env.disable_rendering()
plotGrid(listU=results, domain=domain, dx=dx, label=labels)
