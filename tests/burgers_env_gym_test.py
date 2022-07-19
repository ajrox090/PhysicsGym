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
domain_dict = dict(x=32, bounds=Box[0:1])  # , extrapolation=extrapolation.PERIODIC)
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

env = BurgersEnvGym(**env_krargs)
assert isinstance(env, gym.Env)
agent = PPO(env=env, **agent_krargs)
assert agent is not None

print("training begin")
agent.learn(total_timesteps=32)
print("training complete")

obs = env.reset()
# obs2 = obs[:, :, :2] # 2D
obs2 = obs[:, :1]  # 1D
curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)
vis.plot(curr_state)
vis.show()
curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)

for i in view(play=True, namespace=globals()).range(32):
    curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)
    act = agent.predict(obs)[0]
    obs, reward, done, info = env.step(act)
