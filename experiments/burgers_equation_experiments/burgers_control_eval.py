import time

from phi.flow import *

from src.env.BurgersPhysicsGym import BurgersPhysicsGym
from src.util import run_experiment, plotGrid


N = 8
ph = 5
dxdt = 5
dt = 0.01
dx = 0.25
domain = 4
step_count = 200
viscosity = 0.03

domain_dict = dict(x=int(domain / dx), bounds=Box[0:1], extrapolation=extrapolation.PERIODIC)
env = BurgersPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict, dt=dt, step_count=step_count,
                        viscosity=viscosity, dxdt=dxdt)

print("Uncontrolled")
x = time.time()
uncontrolled_state, uncontrolled_actions, uncontrolled_rewards = run_experiment(N=N, _env=env,
                                                                                linelabels=True, eval=True,
                                                                                multiprocessing=True)

print(f'Uncontrolled finished. \nTotal time taken: {time.time() - x} seconds.')

print("Baseline")
x = time.time()
baseline_state, baseline_actions, baseline_rewards = run_experiment(N=N, _env=env, agent='baseline', eval=True,
                                                                    multiprocessing=True)
print(f'baseline finished. \nTotal time taken: {time.time() - x} seconds.')

print("RL")
x = time.time()
agentPath = f'results/ddpgAgent1_burgers_100epochs_200steps'
rl_state, rl_actions, rl_rewards = run_experiment(N=N, load_model=agentPath, _env=env, agent='ddpg', eval=True)
print(f'rl finished. \nTotal time taken: {time.time() - x} seconds.')

print("MPC")
x = time.time()
mpc_state, mpc_actions, mpc_rewards = run_experiment(N=N, _env=env, agent='mpc', ph=ph, eval=True,
                                                     multiprocessing=True)
print(f'mpc finished. \nTotal time taken: {time.time() - x} seconds.')


x = time.time()
figName = f'results/experiment_heat_eval_{step_count}'
states = [uncontrolled_state, baseline_state, rl_state, mpc_state]
actions = [uncontrolled_actions, baseline_actions, rl_actions, mpc_actions]
rewards = [uncontrolled_rewards, baseline_rewards, rl_rewards, mpc_rewards]
labels = ["uncontrolled", "baseline", "RL", "MPC"]
plotGrid(listU=states, domain=domain, dx=dx, label=labels,
         ylim_min=round(np.min(states) - 0.05, 3), ylim_max=round(np.max(states) + 0.05, 3),
         saveFig=figName + "_states",
         linelabels=True)

plotGrid(listU=actions, label=labels, saveFig=figName + "_actions",
         xlabel="t", ylabel="actions", xlim_max=env.step_count,
         ylim_min=np.min(actions), ylim_max=np.max(actions) + 0.1, linelabels=True)

plotGrid(listU=rewards, xlabel="t", ylabel="rewards", label=labels,
         xlim_max=env.step_count,
         ylim_min=np.min(rewards), ylim_max=np.max(rewards) + 0.1,
         saveFig=f'{figName}_rewards',
         linelabels=True)
print(f'plotting finished. \nTotal time taken: {time.time() - x} seconds.')

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
