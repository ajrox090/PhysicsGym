from matplotlib import pyplot as plt
from phi.flow import *
from stable_baselines3 import DDPG, PPO
from tqdm import tqdm

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


def run(_env=None, agent=None, title: str = 'Uncontrolled', N: int = 1, extra_step_count: int = 0):
    reward = 0.0
    done = False
    total_rewards = []
    finalstateOfEachCycle = []

    # perform cycles
    for _ in tqdm(range(N)):
        observation = _env.reset()
        done = False
        pbar = tqdm(total=step_count)
        while not done:
            if agent is None:
                action = [0]
                if env.step_idx < 8:
                    action = [-1.0]
            else:
                action = agent.predict(observation=observation)
            observation, reward, done, info = _env.step(action)
            pbar.update(1)
            # if done:
            #     for _ in range(extra_step_count):
            #         observation, _, _, _ = _env.step([0.0])
        total_rewards.append(reward)
        finalstateOfEachCycle.append(observation)
        pbar.close()

    print(f'Total mean reward for {title} agent: {np.array(total_rewards).mean()}')
    return np.array(finalstateOfEachCycle).mean(axis=0)


def run_ddpg(save_model: str = None, load_model: str = None,
             learn: bool = False, N: int = 1):
    env.reset()
    if load_model is not None:
        agent_ddpg = DDPG.load(load_model)
        agent_ddpg.set_env(env)
    else:
        agent_ddpg = DDPG(verbose=0, env=env, learning_rate=lr, policy='MlpPolicy')
    if learn:
        print("training begin")
        agent_ddpg.learn(total_timesteps=step_count * n_epochs)
        print("training complete")
        if save_model is not None:
            agent_ddpg.save(save_model)
    return run(_env=env, agent=agent_ddpg, title='DDPG', extra_step_count=0, N=N)


def run_ppo(N: int = 1):
    env.reset()
    # agent_ppo = PPO(verbose=1, env=env, learning_rate=lr, policy='MlpPolicy', tensorboard_log='/log')
    # agent_ppo.learn(total_timesteps=step_count * n_epochs, tb_log_name=f'{time.time()}_ppo')
    agent_ppo = PPO(verbose=0, env=env, learning_rate=lr, policy='MlpPolicy')
    agent_ppo.learn(total_timesteps=step_count * n_epochs)

    return run(_env=env, agent=agent_ppo, title='PPO', N=N)


def run_random(N: int = 1):
    env.reset()
    agent_random = RandomAgent(env=env)

    return run(_env=env, agent=agent_random, title='Random', N=N)


def run_uncontrolled(N: int = 1):
    env.reset()
    return run(_env=env, N=N)


def run_mpc(N: int = 1):
    env.reset()
    agent_mpc = MPCAgent(env=env, ph=10, u_min=-1.0, u_max=1.0)

    return run(_env=env, agent=agent_mpc, title='MPC', N=N)


class Aagent:
    def __init__(self, actions, env):
        self.actions = actions
        self.env = env

    def predict(self, observation=None):
        return np.array([self.actions[env.step_idx]])


def run_manual_control(actions):
    env.reset()
    agent = Aagent(actions, env)
    run(env, agent, title='manual')


lr = 0.0001
n_epochs = 5

dx = 0.25
domain = 3
step_count = 10
domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.BOUNDARY)
env = HeatPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                     dt=0.01, step_count=step_count,
                     diffusivity=2.0, dxdt=5)
env.reset()
init_state = env.init_state.data.native("vector,x")[0]
# optimal_actions = [(-0.2 if i < step_count else 0.0) for i in range(step_count)]
# run_manual_control(optimal_actions)
N = 1

# print("PPO")
# ppo_state = run_ppo()
# print("DDPG")
# ddpg_state = run_ddpg(load_model="ddpgAgent.zip", N=N,
# #                       learn=True, save_model="ddpgAgent")
# ddpg_state = run_ddpg(load_model="ddpgAgent_working", N=N)
# # ddpg_state = run_ddpg(N=N, learn=True, save_model="ddpgAgent")
# print("random agent")
# random_state = run_random(N=N)
# print("uncontrolled")
# uncontrolled_state = run_uncontrolled(N=N)
print("MPC")
mpc_state = run_mpc(N=N)

# print("visualization")
plotGrid([
    init_state,
    # ddpg_state,
    # ppo_state,
    # random_state,
    # uncontrolled_state,
    mpc_state
],
    domain=domain, dx=dx,
    label=[
        'initial state',
        # 'ddpg state',
        # 'ppo state',
        # 'random state',
        # 'uncontrolled state',
        'mpc state'
    ])
