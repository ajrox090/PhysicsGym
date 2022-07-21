import unittest
from typing import Optional
import gym
from phi.flow import *
from stable_baselines3 import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.ppo import MlpPolicy

from src.env.burgers_env_gym import BurgersEnvGym
from src.runner import RLRunner

runner = RLRunner(path_config="../experiment.yml")

# env
N = runner.config['env']['N']
num_envs = runner.config['env']['num_envs']
step_count = runner.config['env']['step_count']
domain_dict = dict(x=64, bounds=Box[0:1])  # , extrapolation=extrapolation.PERIODIC)
dt = 1. / step_count
viscosity = 0.01 / (N * np.pi)
if 'viscosity' in runner.config['env'].keys():
    viscosity = runner.config['env']['viscosity']
diffusion_substeps = runner.config['env']['diffusion_substeps']
final_reward_factor = runner.config['env']['final_reward_factor']
reward_rms: Optional[RunningMeanStd] = None

# agent
num_epochs = runner.config['agent']['num_epochs']
lr = runner.config['agent']['lr']
batch_size = step_count
env_krargs = dict(N=N, num_envs=num_envs, domain_dict=domain_dict, dt=dt,
                  viscosity=viscosity, diffusion_substeps=diffusion_substeps,
                  final_reward_factor=final_reward_factor, reward_rms=reward_rms)
agent_krargs = dict(verbose=0, policy=MlpPolicy,
                    n_steps=step_count,
                    n_epochs=num_epochs,
                    learning_rate=lr,
                    batch_size=batch_size)

# 1) Create an instance of Burgers' environment defined in phiflow/Burgers.py  with above parameters.
env = BurgersEnvGym(**env_krargs)
# changed the env interface from stable_baselines3.VecEnv -> gym.Env
assert isinstance(env, gym.Env)
# 2) Create default PPO agent without any external NNs.
agent = PPO(env=env, **agent_krargs)

# 3) train the agent to learn the distribution of actions using an optimization algorithm
# i.e. maximising the following reward,
#           reward = -(current_state - gt_state)**2
print("training begins")
agent.learn(total_timesteps=32)
print("training complete")

# Note: Here the term 'State' refers to phiflow.CenteredGrid/.StaggeredGrid/.PointCloud,...
# which is a fancy way of representing phiflow.Field implementations which are used to store interesting values like,
# velocity, pressure, temperature, etc.
#
# 3.0) Initial State:
# During training, the agent interacts with the environment in the form of performing
# actions. The actions in general could be discrete or continuous, but here we consider only continuous actions
# which are sampled from a simple 'Gaussian function'. The sampled action is used only initially and then the NN updates
# it in the direction of maximizing the reward above.
#
# 3.1) Target State:
# The 'render' function of the Env also plots individual states and shows how each state progresses in time in
# comparison to its ground truth value, decided by the target function. The target function here is 'Gaussian force'
# function defined in util/burgers_util.py

# 4) Testing:
#
# 4.1) Reset:
# let's test how the agent performs on 'a simple gaussian function' with different parameters.
# Maybe, I should probably use a different/ similar function, but it all depends on the problem at hand.
obs = env.reset()
# 4.2) Pre-processing:
# On resetting the environment, we get the observation which is a tuple of (current_state, ground_truth_state, time)
# obs2 = obs[:, :, :2] # 2D
obs2 = obs[:, :1]  # 1D  # extract current_state
# Now, numpy.ndarray cannot be directly used as values in CenterGrid so, convert it to 'phi.math.tensor'
curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)
vis.plot(curr_state)
vis.show()
curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)

# 4.3) Play :D
# The view below is a very nice interactive viewer by phiflow, this basically plots every phi.Field objects in an
# interactive plot which opens in a browser. The objects for plotting can also be described as parameters.
# For the below example, the supported object is curr_state.
for i in view(play=True, namespace=globals()).range(32):
    curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)
    act = agent.predict(obs)[0]
    obs, reward, done, info = env.step(act)
