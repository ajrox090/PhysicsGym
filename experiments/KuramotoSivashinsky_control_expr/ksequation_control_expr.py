import copy
from tqdm import tqdm
from matplotlib import pyplot as plt

from phi.flow import *
from stable_baselines3 import PPO, DDPG
from stable_baselines3.ppo import MlpPolicy as MlpPolicy_ppo
from stable_baselines3.ddpg import MlpPolicy as MlpPolicy_ddpg

from src.env.BurgersPhysicsGym import BurgersPhysicsGym
from src.env.KSPhysicsGym import KuramotoSivashinskyPhysicsGym


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


dx = 0.25
domain = 128
step_count = 100
domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.BOUNDARY)
dt = 0.01

env_krargs = dict(domain=domain, dx=dx, domain_dict=domain_dict,
                  dt=dt, step_count=step_count)

agent_krargs_ppo = dict(verbose=2, policy=MlpPolicy_ppo,
                        n_steps=step_count,
                        n_epochs=10000,
                        learning_rate=0.0001,
                        batch_size=step_count)

agent_krargs_ddpg = dict(verbose=2, policy=MlpPolicy_ddpg,
                         learning_rate=0.0001,
                         batch_size=step_count)

print("DDPG")
env_ddpg = KuramotoSivashinskyPhysicsGym(**env_krargs)
# env_ddpg.init_state = copy.deepcopy(env_ppo.init_state)
_ = env_ddpg.reset()
agent_ddpg = DDPG(env=env_ddpg, **agent_krargs_ddpg)
agent_ddpg.learn(total_timesteps=step_count)
# env_ddpg.render(mode='final', title='rl control ddpg')

print("PPO")
env_ppo = KuramotoSivashinskyPhysicsGym(**env_krargs)
env_ppo.init_state = copy.deepcopy(env_ddpg.init_state)
_ = env_ppo.reset()
agent = PPO(env=env_ppo, **agent_krargs_ppo)
agent.learn(total_timesteps=step_count)
# # env.render(mode='final', title='rl control ppo')

print("random agent")
env_random = KuramotoSivashinskyPhysicsGym(**env_krargs)
env_random.init_state = copy.deepcopy(env_ddpg.init_state)
# env_random.init_state = copy.deepcopy(env_ppo.init_state)
_ = env_random.reset()
for i in tqdm(range(step_count)):
    action = [env_random.action_space.sample()]
    obs, _, _, _ = env_random.step(action)
# env_random.render(mode='final', title='random control')

print("uncontrolled")
env_uncontrolled = KuramotoSivashinskyPhysicsGym(**env_krargs)
env_uncontrolled.init_state = copy.deepcopy(env_ddpg.init_state)
# env_uncontrolled.init_state = copy.deepcopy(env_ppo.init_state)
_ = env_uncontrolled.reset()
u = env_uncontrolled.init_state
for i in tqdm(range(step_count)):
    u = env_uncontrolled.physics.step(u, dt=dt)

print("visualization")
plotGrid([
    env_ppo.init_state.data.native("vector,x")[0],
    env_ppo.final_state.data.native("vector,x")[0],
    env_ddpg.final_state.data.native("vector,x")[0],
    env_random.final_state.data.native("vector,x")[0],
    u.data.native("vector,x")[0],
    # env_ddpg.reference_state_phi.data.native("vector,x")[0]],
    env_ppo.reference_state_phi.data.native("vector,x")[0]],
    domain=domain, dx=dx,
    label=[
        'initial state',
        'rl ppo final',
        'rl ddpg final',
        'random agent final',
        'uncontrolled final',
        'target state'])

plt.plot(env_ppo.previous_rew, label='ppo reward')
plt.plot(env_ddpg.previous_rew, label='ddpg reward')
plt.legend()
plt.title("RL Rewards for PPO and DDPG")
plt.show()
