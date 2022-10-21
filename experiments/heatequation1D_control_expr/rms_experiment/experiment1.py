import time
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from phi.flow import *
from stable_baselines3 import DDPG

from src.agent.RandomAgent import RandomAgent
from src.env.HeatPhysicsGym import HeatPhysicsGym, HeatPhysicsGymNoRMS


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


def run(N: int = 1, _env=None, agent=None, save_model: str = None, load_model: str = None, learn: bool = False,
        n_epochs: int = 5, label: str = ""):

    if type(agent) == str:
        if agent == 'ddpg':
            if load_model is not None:
                agent = DDPG.load(load_model)
                agent.set_env(_env)
            else:
                agent = DDPG(verbose=0, env=_env, learning_rate=lr, policy='MlpPolicy')
        elif agent == 'random':
            agent = RandomAgent(env=_env)

    if learn:  # train
        # _env.enable_rendering()
        agent = agent.learn(total_timesteps=step_count * n_epochs)
        if save_model is not None:
            agent.save(save_model)
        return agent
    else:  # test
        reward = 0.0
        rewards = []
        final_states = []
        total_actions = np.zeros((N, _env.step_count), dtype=float)
        # perform cycles
        for i in range(N):
            obs = _env.reset()
            done = False
            while not done:
                if agent is None:
                    action = [0.0]
                else:
                    action = agent.predict(observation=obs)
                    total_actions[i][_env.step_idx-1] = action[0]
                if _env.step_idx in [1, 15, 30, 60, 120, 240, 299]:
                    _env.enable_rendering()  # only show final state during testing
                obs, reward, done, info = _env.step(action)
                if _env._render:
                    _env.disable_rendering()
                # print(f'Reward for cycle {i}')
            rewards.append(reward)
            final_states.append(obs)

        print(f'Average reward for {label}: {np.array(rewards).mean()}')
        # print(f'Actions: \n {total_actions}')
        with open("output.txt", "a") as f:
            f.write("\n")
            f.write(f'actions for {label}: \n{total_actions}\n')
            f.write("-------------------------------------------------------------------------------")
        return np.array(final_states).mean(axis=0)


lr = 0.0001
N = 10
step_count = 300
dxdt = 5
dx = 0.25
domain = 3
domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.BOUNDARY)
env = HeatPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                     dt=0.01, step_count=step_count,
                     diffusivity=2.0, dxdt=dxdt)
env_norms = HeatPhysicsGymNoRMS(domain=domain, dx=dx, domain_dict=domain_dict,
                                dt=0.01, step_count=step_count,
                                diffusivity=2.0, dxdt=dxdt)

print("DDPG")
ddpg_state = []
for epoch in [10]:
    # agentPath = f'ddpgAgent1_RMS_{epoch}epochs'
    # print("train-store")
    # x = time.time()
    # run(learn=True, save_model=agentPath, _env=env, agent='ddpg', n_epochs=epoch, label=agentPath)
    # print(f'{time.time() - x} seconds elapsed for training with {epoch} epochs.')
    # print("--------------------------------------------------------------------------------")
    # print("Test")
    # x = time.time()
    # ddpg_state_withRMS = run(load_model=agentPath, _env=env, agent='ddpg', label=agentPath)
    # print(f'{time.time() - x} seconds elapsed for testing with N={N}.')
    # ddpg_state.append(ddpg_state_withRMS)
    # print("--------------------------------------------------------------------------------")

    agentPath = f"ddpgAgent3_noRMS_{epoch}epochs"
    print("train-store")
    # x = time.time()
    # run(learn=True, save_model=agentPath, _env=env_norms, agent='ddpg', label=agentPath)
    # print(f'{time.time() - x} seconds elapsed for training with {epoch} epochs.')
    # print("--------------------------------------------------------------------------------")

    print("Test")
    x = time.time()
    ddpg_state_noRMS = run(load_model=agentPath, _env=env_norms, agent='ddpg', label=agentPath)
    print(f'{time.time() - x} seconds elapsed for testing with N={N}.')
    ddpg_state.append(ddpg_state_noRMS)
    print("--------------------------------------------------------------------------------")

print("visualization")
plotGrid([
    ddpg_state[0],
    env.reference_state_np,
    # ddpg_state[1],

],
    domain=domain, dx=dx,
    label=[
        # 'ddpg state no RMS 5 epochs',
        'ddpg state no RMS 10 epochs',
        'reference state',
    ])
