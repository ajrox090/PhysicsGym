from typing import Optional
import gym
from phi.flow import *
from stable_baselines3 import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.ppo import MlpPolicy

from src.env.heat_env_gym import Heat1DEnvGym
from src.runner import RLRunner

runner = RLRunner(path_config="experiment.yml")
rc_env = runner.config['env']
rc_agent = runner.config['agent']
# env
N = rc_env['N']
step_count = rc_env['step_count']
domain_dict = dict(x=N, bounds=Box[-1:1], extrapolation=extrapolation.PERIODIC)
dt = 1. / step_count
diffusivity = 0.01 / (N * np.pi)
if 'diffusivity' in rc_env.keys():
    diffusivity = rc_env['diffusivity']
final_reward_factor = rc_env['final_reward_factor']
reward_rms: Optional[RunningMeanStd] = None

# agent
num_epochs = rc_agent['num_epochs']
lr = rc_agent['lr']
batch_size = step_count
env_krargs = dict(N=N, domain_dict=domain_dict, dt=dt, step_count=step_count,
                  diffusivity=diffusivity,
                  final_reward_factor=final_reward_factor, reward_rms=reward_rms)
agent_krargs = dict(verbose=0, policy=MlpPolicy,
                    n_steps=step_count,
                    n_epochs=num_epochs,
                    learning_rate=lr,
                    batch_size=batch_size)

# 1) Create an instance of Burgers' environment defined in phiflow/Burgers.py  with above parameters.
env = Heat1DEnvGym(**env_krargs)
# changed the env interface from stable_baselines3.VecEnv -> gym.Env
assert isinstance(env, gym.Env)
# 2) Create default PPO agent without any external NNs.
agent = PPO(env=env, **agent_krargs)

# 3) train the agent to learn the distribution of actions using an optimization algorithm
# i.e. maximising the following reward,
#           reward = -(current_state - gt_state)**2
print("training begins")
env.enable_rendering()
agent.learn(total_timesteps=32)
print("training complete")

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
# vis.plot(curr_state)
# vis.show()
curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)

# 4.3) Play :D
# The view below is a very nice interactive viewer by phiflow, this basically plots every phi.Field objects in an
# interactive plot which opens in a browser. The objects for plotting can also be described as parameters.
# For the below example, the supported object is curr_state.
for i in view(play=True, namespace=globals()).range(32):
    curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)
    act = agent.predict(obs)[0]
    obs, reward, done, info = env.step(act)
