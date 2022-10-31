import time

from phi.flow import *
from src.env.HeatPhysicsGym import HeatPhysicsGym
from src.util import plotGrid, run_experiment

ph = 5
dxdt = 5
dt = 0.01
dx = 0.25
domain = 3
step_count = 400
diffusivity = 2.0

env = HeatPhysicsGym(domain=domain, dx=dx,
                     domain_dict=dict(x=int(domain / dx), bounds=Box[0:1],
                                      extrapolation=extrapolation.BOUNDARY),
                     dt=dt, step_count=step_count,
                     diffusivity=diffusivity, dxdt=dxdt)

np.random.seed(np.random.randint(0, 1000))
x = time.time()

print("Uncontrolled")
uncontrolled_state, uncontrolled_actions = run_experiment(_env=env, linelabels=True, eval=True)

print("Baseline")
baseline_state, baseline_actions = run_experiment(_env=env, agent='baseline', eval=True)

print("RL")
agentPath = f'results/ddpgAgent1_heat_1000epochs_200steps'
rl_state, rl_actions = run_experiment(load_model=agentPath, _env=env, agent='ddpg', eval=True)

print("MPC")
mpc_state, mpc_actions = run_experiment(_env=env, agent='mpc', ph=ph, eval=True)

figName = f'results/experiment_heat_eval'
states = [uncontrolled_state, baseline_state, rl_state, mpc_state]
actions = [uncontrolled_actions, baseline_actions, rl_actions, mpc_actions]

plotGrid(listU=states, domain=domain, dx=dx, label=["uncontrolled", "baseline", "RL", "MPC"],
         ylim_min=round(np.min(states) - 0.05, 3), ylim_max=round(np.max(states) + 0.05, 3),
         saveFig=figName + "_states",
         linelabels=True)

plotGrid(listU=actions, label=["uncontrolled", "baseline", "RL", "MPC"], saveFig=figName + "_actions",
         xlabel="t", ylabel="actions", xlim_max=env.step_count,
         ylim_min=np.min(actions), ylim_max=np.max(actions) + 0.1, linelabels=True)
print(f'Experiment finished. \nTotal time taken: {time.time() - x} seconds.')

# save actions to file for later plotting
for idx, acts in enumerate(actions):
    fName = None
    if idx == 0:
        fName = f'results/uncontrolled_actions.txt'
    elif idx == 1:
        fName = f'results/baseline_actions.txt'
    elif idx == 2:
        fName = f'results/rl_actions.txt'
    elif idx == 3:
        fName = f'results/mpc_actions.txt'
    with open(fName, 'w') as f:
        for j, act in enumerate(acts):
            if j == 0:
                f.write(fName.split("/")[1] + "\n")
            if j < len(acts) - 1:
                f.write(str(act) + ", ")
            else:
                f.write(str(act) + "\n\n")
