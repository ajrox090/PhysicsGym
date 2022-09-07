from typing import Optional
import gym
from phi.flow import *
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.ddpg import MlpPolicy
# from stable_baselines3.ppo import MlpPolicy

from src.env.ks3_env_gym import KS3EnvGym
from src.runner import RLRunner

runner = RLRunner(path_config="experiment.yml")
rc_env = runner.config['env']
rc_agent = runner.config['agent']

N = rc_env['N']
bound_x = rc_env['bounds']
step_count = rc_env['step_count']
# dt = 1. / step_count
# print(dt)
dt = rc_env['dt']
reward_rms: Optional[RunningMeanStd] = None
final_reward_factor = rc_env['final_reward_factor']
domain_dict = dict(x=N, bounds=Box(x=bound_x))

lr = rc_agent['lr']
batch_size = step_count
num_epochs = rc_agent['num_epochs']
env_krargs = dict(N=N, domain_dict=domain_dict, dt=dt, step_count=step_count,
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

env = KS3EnvGym(**env_krargs)
assert isinstance(env, gym.Env)

# agent = PPO(env=env, **agent_krargs_ppo)
agent = DDPG(env=env, **agent_krargs_ddpg)

print("training begins")
env.enable_rendering()
agent.learn(total_timesteps=step_count)
print("training complete")
env.last_render()

# todo: normal simulation without forces works till 1000 steps,
# todo: but with forces from rl agent, maximum 50 steps, investigate.

"""
    - one way is to generate forces only for small region and 
        then reshaping the actions.
"""
