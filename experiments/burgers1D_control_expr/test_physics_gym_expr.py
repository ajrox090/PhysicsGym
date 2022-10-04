import copy
from typing import Optional
import gym
from phi.flow import *
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.ppo import MlpPolicy
from tqdm import tqdm

from src.env.PhysicsGym import TestPhysicsGym
# from stable_baselines3.ddpg import MlpPolicy

from src.env.burgers_env_gym import Burgers1DEnvGym
from src.env.heat_env_gym import Heat1DEnvGym
from src.env.physics.heat import Heat
from src.networks import RES_UNET, CNN_FUNNEL
from src.policy import CustomActorCriticPolicy

# env
N = 8
step_count = 8
domain_dict = dict(x=N, bounds=Box[0:1],
                   extrapolation=extrapolation.PERIODIC)
dt = 1. / step_count
viscosity = 0.01 / (N * np.pi)

diffusion_substeps = 1
final_reward_factor = 8
reward_rms: Optional[RunningMeanStd] = None

env_krargs = dict(N=N, domain_dict=domain_dict, dt=dt, step_count=step_count,
                  diffusivity=viscosity,
                  final_reward_factor=final_reward_factor, reward_rms=reward_rms)
# env_krargs = dict(N=N, domain_dict=domain_dict, dt=dt, step_count=step_count,
#                   viscosity=viscosity, diffusion_substeps=diffusion_substeps,
#                   final_reward_factor=final_reward_factor, reward_rms=reward_rms)

# agent
num_epochs = 10
lr = 0.0001
batch_size = step_count

agent_krargs_ppo = dict(verbose=0, policy=MlpPolicy,
                        n_steps=step_count,
                        n_epochs=num_epochs,
                        learning_rate=lr,
                        batch_size=batch_size)

# 1) Create an instance of Burgers' environment defined in phiflow/Burgers.py  with above parameters.
env = TestPhysicsGym(**env_krargs)

# 2) Create default PPO agent without any external NNs.
agent = PPO(env=env, **agent_krargs_ppo)

obs = env.reset()
u = env.init_state
vis.show(u)
# for _ in view('state, obs', framerate=1, namespace=globals()).range():
# # for i in tqdm(range(10000)):
#     actions = (np.random.uniform(-1.0, 1.0),)
#     obs, rew, done, info = env.step(np.array(actions))
#     state = env.cont_state
# 3) train the agent to learn the distribution of actions using an optimization algorithm
# i.e. maximising the following reward,
#           reward = -(current_state - gt_state)**2
print("training begins")
agent.learn(total_timesteps=1000)
print("training complete")
