import copy
import time

from matplotlib import pyplot as plt

from phi.flow import *
from stable_baselines3 import DDPG

from src.agent.RandomAgent import RandomAgent
from src.env.HeatPhysicsGym import HeatPhysicsGym


def plotGrid(listU, domain: int, dx: float, label: list[str]):
    x = np.arange(0, domain, dx)
    plt.tick_params(axis='x', which='minor', length=10)
    plt.grid(True, linestyle='--', which='both')
    for u, lab in zip(listU, label):
        plt.plot(x, u, label=lab)
    plt.xlim(0, domain)
    plt.ylim(-3, 3)
    plt.legend()
    plt.show()


lr = 0.0001
dx = 0.1
domain = 3
step_count = 200
n_epochs = 5
domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.BOUNDARY)

print("DDPG")
env = HeatPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                     dt=0.01, step_count=step_count,
                     diffusivity=2.0)

env.reset()
agent_ddpg = DDPG(verbose=1, env=env, learning_rate=lr, policy='MlpPolicy', tensorboard_log='./log')
agent_ddpg.learn(total_timesteps=step_count * n_epochs, tb_log_name=f'{time.time()}_ddpg')
# test
obs = env.reset()
done = False
while not done:
    obs, _, done, _ = env.step(agent_ddpg.predict(observation=obs))
ddpg_state = copy.deepcopy(obs)

print("random agent")
agent_random = RandomAgent(env=env, learning_rate=lr, tensorboard_log='./log')
env.reset()
done = False
while not done:
    obs, _, done, _ = env.step(agent_random.predict(obs))
random_state = copy.deepcopy(obs)

print("uncontrolled")
env.reset()
done = False
while not done:
    obs, _, done, _ = env.step([0])
uncontrolled_state = copy.deepcopy(obs)

print("visualization")
plotGrid([
    ddpg_state,
    random_state,
    uncontrolled_state],
    domain=domain, dx=dx,
    label=[
        'ddpg state',
        'random state',
        'uncontrolled state'
    ])
