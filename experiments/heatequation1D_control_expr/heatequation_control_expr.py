import copy

from matplotlib import pyplot as plt
from phi.flow import *
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from tqdm import tqdm

from src.env.PhysicsGym import HeatPhysicsGym

dx = 0.25
domain = 5
step_count = 100
domain_dict = dict(x=int(domain/dx), bounds=Box[0:1],
                   extrapolation=extrapolation.BOUNDARY)
# dt = 1. / step_count
dt = 0.01
# viscosity = 0.01 / (N * np.pi)

env_krargs = dict(domain=domain, dx=dx, domain_dict=domain_dict,
                  dt=dt, step_count=step_count,
                  diffusivity=0.3,
                  final_reward_factor=32)

num_epochs = 1000
lr = 0.0001
batch_size = step_count

agent_krargs_ppo = dict(verbose=0, policy=MlpPolicy,
                        n_steps=step_count,
                        n_epochs=num_epochs,
                        learning_rate=lr,
                        batch_size=batch_size)

env = HeatPhysicsGym(**env_krargs)

agent = PPO(env=env, **agent_krargs_ppo)

obs = env.reset()
print("PPO")
agent.learn(total_timesteps=step_count)
x = np.arange(0, domain, dx)
plt.tick_params(axis='x', which='minor', length=10)
plt.grid(True, linestyle='--', which='both')
plt.plot(x, env.initial_state.data.native("vector,x")[0], label='init state')
plt.plot(x, env.final_state.data.native("vector,x")[0], label='rl state')
plt.plot(x, env.reference_state_np, label='reference state')
plt.xlim(0, domain)
plt.ylim(-3, 3)
plt.legend()
plt.show()

# print("random agent")
# env2 = HeatPhysicsGym(**env_krargs)
# _ = env2.reset()
# env2.init_state = copy.deepcopy(env.init_state)
# env2.cont_state = copy.deepcopy(env.init_state)
# vis.show(env2.init_state, env2.cont_state)
# for i in tqdm(range(step_count)):
#     action = [env2.action_space.sample()]
#     obs, _, _, _ = env2.step(action)
#
#
# print("uncontrolled")
# env3 = HeatPhysicsGym(**env_krargs)
# _ = env3.reset()
# env3.init_state = copy.deepcopy(env.init_state)
# env3.cont_state = copy.deepcopy(env.init_state)
# vis.show(env3.init_state, env3.cont_state)
# u = env3.init_state
# for i in tqdm(range(step_count)):
#     u = env3.physics.step(u, dt=dt)
# x = np.arange(0, domain, dx)
# plt.tick_params(axis='x', which='minor', length=10)
# plt.grid(True, linestyle='--', which='both')
#
# plt.plot(x, u.data.native("vector,x")[0], label='uncontrolled state')
# plt.plot(x, env.cont_state.data.native("vector,x")[0], label='rl control state')
# plt.plot(x, env2.cont_state.data.native("vector,x")[0], label='random agent control state')
#
# plt.plot(x, env.reference_state_np, label='reference state')
# plt.xlim(0, domain)
# plt.ylim(-3, 3)
# plt.legend()
# plt.show()

