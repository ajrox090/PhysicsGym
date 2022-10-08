from typing import Optional
import gym
from phi.flow import *
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.ddpg import MlpPolicy

from src.env.BurgersPhysicsGym import Burgers1DEnvGym
from src.runner import RLRunner

runner = RLRunner(path_config="experiment.yml")
rc_env = runner.config['env']
rc_agent = runner.config['agent']
# env
N = rc_env['N']
step_count = rc_env['step_count']

bound_x = rc_env['bounds']
domain_dict = dict(x=N, bounds=Box[0:1], extrapolation=extrapolation.PERIODIC)
dt = 1. / step_count
viscosity = 0.01 / (N * np.pi)
if 'viscosity' in rc_env.keys():
    viscosity = rc_env['viscosity']
diffusion_substeps = rc_env['diffusion_substeps']
final_reward_factor = rc_env['final_reward_factor']
reward_rms: Optional[RunningMeanStd] = None

# agent
num_epochs = rc_agent['num_epochs']
lr = rc_agent['lr']
batch_size = step_count
env_krargs = dict(N=N, domain_dict=domain_dict, dt=dt, step_count=step_count,
                  viscosity=viscosity, diffusion_substeps=diffusion_substeps,
                  final_reward_factor=final_reward_factor, reward_rms=reward_rms)
agent_krargs_ppo = dict(verbose=0, policy=MlpPolicy,
                        n_steps=step_count,
                        n_epochs=num_epochs,
                        learning_rate=lr,
                        batch_size=batch_size)
agent_krargs_ddpg = dict(verbose=0, policy=MlpPolicy,
                         # n_steps=step_count,
                         # n_epochs=num_epochs,
                         learning_rate=lr,
                         batch_size=batch_size)

# 1) Create an instance of Burgers' environment defined in phiflow/Burgers.py  with above parameters.
env = Burgers1DEnvGym(**env_krargs)
# changed the env interface from stable_baselines3.VecEnv -> gym.Env
assert isinstance(env, gym.Env)
# 2) Create default PPO agent without any external NNs.
# agent = DDPG(env=env, **agent_krargs_ddpg)
agent = PPO(env=env, **agent_krargs_ppo)


# 3) train the agent to learn the distribution of actions using an optimization algorithm
# i.e. maximising the following reward,
#           reward = -(current_state - gt_state)**2
print("training begins")
env.enable_rendering()
agent.learn(total_timesteps=32)
env.last_render()
print("training complete")
