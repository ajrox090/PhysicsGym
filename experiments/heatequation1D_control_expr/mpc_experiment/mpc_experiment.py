import time
from matplotlib import pyplot as plt

from phi.flow import *
from src.agent.MPCAgent import MPCAgent
from src.env.HeatPhysicsGym import HeatPhysicsGymNoRMS


def plotGrid(listU, domain: int, dx: float, label: list[str]):
    x = np.arange(0, domain, dx)
    plt.tick_params(axis='x', which='minor', length=10)
    plt.grid(True, linestyle='--', which='both')
    for u, lab in zip(listU, label):
        plt.plot(x, u, label=lab)
    plt.xlim(0, domain)
    plt.ylim(-3, 3)
    plt.xlabel("domain")
    plt.ylabel("range")
    plt.legend()
    plt.show()


def run(N: int = 1, _env=None, ph: int = 5, label: str = "",
        u_min: float = -1.0, u_max: float = 1.0):
    reward = 0.0
    rewards = []
    final_states = []
    total_actions = np.zeros((N, _env.step_count), dtype=float)

    agent = MPCAgent(env=env, ph=ph, u_min=u_min, u_max=u_max)
    # perform cycles
    for i in range(N):
        obs = _env.reset()
        done = False
        while not done:
            action = agent.predict(observation=obs)
            total_actions[i][_env.step_idx-1] = action[0]
            # if _env.step_idx % int(_env.step_count // 20) == 0:
            # if _env.step_idx < _env.step_count:
            if _env.step_idx in [1, 15, 30, 60, 120, 299]:
                _env.enable_rendering()  # only show final state during testing
            obs, reward, done, info = _env.step(action)
            if _env._render:
                _env.disable_rendering()
        rewards.append(reward)
        final_states.append(obs)

    print(f'Average reward for {label}: {np.array(rewards).mean()}')
    with open("output.txt", "a") as f:
        f.write(f'\nactions for {label}: \n{total_actions}\n')
        f.write("-------------------------------------------------------------------------------\n")
    return np.array(final_states).mean(axis=0)


N = 2
step_count = 300
dxdt = 5
dx = 0.25
domain = 3
domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.BOUNDARY)
env = HeatPhysicsGymNoRMS(domain=domain, dx=dx, domain_dict=domain_dict,
                          dt=0.01, step_count=step_count,
                          diffusivity=2.0, dxdt=dxdt,
                          saveFig=True, title="experiment1", plotFolder="plots")

print("MPC")
ph = 4

label = 'mpc_experiment1'
print(label)
x = time.time()
mpc_state = run(_env=env, label=label, u_min=-1.0, u_max=1.0, ph=ph)
print(f'{time.time() - x} seconds elapsed for mpc_experiment.py.')
print("--------------------------------------------------------------------------------")

print("visualization")
plotGrid([mpc_state], domain=domain, dx=dx, label=['mpc_state'])
