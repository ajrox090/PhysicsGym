import time
from matplotlib import pyplot as plt

from phi.flow import *
from stable_baselines3 import DDPG, PPO

from src.agent.MPCAgent import MPCAgent
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
    plt.xlabel("domain")
    plt.ylabel("range")
    plt.legend()
    plt.show()


def run(_env=None, agent=None, title: str = 'Uncontrolled'):
    done = False
    rew = 0.0
    rews = []
    obsList = []
    for _ in range(5):
        obs = _env.reset()
        while not done:
            a = agent.predict(observation=obs)
            print(a)
            obs, rew, done, info = _env.step(a if agent is not None else [0])
        rews.append(rew)
        obsList.append(obs)
    print(f'Average reward for {title} agent: {np.array(rews).mean()}')

    return np.array(obsList).mean(axis=0)


def run_ddpg():
    env.reset()
    # agent_ddpg = DDPG(verbose=1, env=env, learning_rate=lr, policy='MlpPolicy', tensorboard_log='./log')
    # agent_ddpg.learn(total_timesteps=step_count * n_epochs, tb_log_name=f'{time.time()}_ddpg')
    agent_ddpg = DDPG(verbose=0, env=env, learning_rate=lr, policy='MlpPolicy')
    agent_ddpg.learn(total_timesteps=step_count * n_epochs)

    return run(_env=env, agent=agent_ddpg, title='DDPG')


def run_ppo():
    env.reset()
    # agent_ppo = PPO(verbose=1, env=env, learning_rate=lr, policy='MlpPolicy', tensorboard_log='/log')
    # agent_ppo.learn(total_timesteps=step_count * n_epochs, tb_log_name=f'{time.time()}_ppo')
    agent_ppo = PPO(verbose=0, env=env, learning_rate=lr, policy='MlpPolicy')
    agent_ppo.learn(total_timesteps=step_count * n_epochs)

    return run(_env=env, agent=agent_ppo, title='PPO')


def run_random():
    env.reset()
    agent_random = RandomAgent(env=env)

    return run(_env=env, agent=agent_random, title='Random')


def run_uncontrolled():
    env.reset()
    return run(_env=env)


def run_mpc():
    env.reset()
    agent_mpc = MPCAgent(env=env, ph=5, u_min=-1.0, u_max=1.0)

    return run(_env=env, agent=agent_mpc, title='MPC')


lr = 0.0001
dx = 0.5
domain = 4
step_count = 200
n_epochs = 10
domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.BOUNDARY)
env = HeatPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                     dt=0.1, step_count=step_count,
                     diffusivity=2.0, dxdt=10)

# print("PPO")
# ppo_state = run_ppo()

# print("DDPG")
# ddpg_state = run_ddpg()
#
# print("random agent")
# random_state = run_random()
#
# print("uncontrolled")
# uncontrolled_state = run_uncontrolled()

print("MPC")
mpc_state = run_mpc()

print("visualization")
plotGrid([
    # ddpg_state,
    # ppo_state,
    # random_state,
    # uncontrolled_state,
    mpc_state
],
    domain=domain, dx=dx,
    label=[
        # 'ddpg state',
        # 'ppo state',
        # 'random state',
        # 'uncontrolled state',
        'mpc state'
    ])
