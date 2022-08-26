from phi.flow import *
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.env.navier_stokes_env_gym import NavierStokesEnvGym


# env
from src.env.phiflow.navier_stokes import NavierStokes

N = 128
step_count = 32
speed = 2.0
# domain_dict = dict(x=N, y=N, bounds=Box(x=N, y=N/2), extrapolation=extrapolation.BOUNDARY)
domain_dict = dict(x=N, y=N, bounds=Box(x=N, y=N), extrapolation=extrapolation.BOUNDARY)
dt = 1. / step_count
final_reward_factor = step_count
reward_rms: Optional[RunningMeanStd] = None

env_krargs = dict(
    N=N, step_count=step_count, domain_dict=domain_dict,
    dt=dt, final_reward_factor=final_reward_factor, reward_rms=reward_rms)

# agent
num_epochs = 100
lr = 1e-4
batch_size = 32
agent_krargs = dict(verbose=0, policy=MlpPolicy,
                    n_steps=step_count,
                    n_epochs=num_epochs,
                    learning_rate=lr,
                    batch_size=batch_size)

# 1) Create an instance of Navier Stoke's environment
env = NavierStokesEnvGym(**env_krargs)

# 2) Create PPO agent
agent = PPO(env=env, **agent_krargs)

# 3) train
print("training begins")
# agent.learn(total_timesteps=32)
print("training complete")

# 4) Testing:

obs = env.reset()

in_state = StaggeredGrid(math.tensor(obs, env.init_state.shape), extrapolation=extrapolation.BOUNDARY)
physics = NavierStokes()
pressure = None
vorticity = None
for i in view('drag,vorticity,velocity,pressure',play=True, namespace=globals()).range():
    in_state, pressure, vorticity = physics.step(in_state, obstacles=env.obstacles,
                                                 boundary_mask=env.boundary_mask, dt=1.0)

    cd = 0.82  # drag coefficient
    drag = cd * pressure * (in_state ** 2) / 2

# for i in view(play=True, namespace=globals()).range():
#     curr_state = CenteredGrid(math.tensor(obs, env.cont_state.shape), obs.shape)
#     act = agent.predict(obs)[0]
#     obs, reward, done, info = env.step(act)
