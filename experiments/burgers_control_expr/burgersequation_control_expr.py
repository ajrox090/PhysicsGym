from tqdm import tqdm
from phi.flow import *
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DDPG
from stable_baselines3.ppo import MlpPolicy

from src.env.BurgersPhysicsGym import BurgersPhysicsGym

# env
N = 8
step_count = 100
domain_dict = dict(x=N, bounds=Box[0:1],
                   extrapolation=extrapolation.PERIODIC)
dt = 1. / step_count
viscosity = 0.01 / (N * np.pi)

env_krargs = dict(N=N, domain_dict=domain_dict, dt=dt, step_count=step_count,
                  diffusivity=viscosity,
                  final_reward_factor=8)

num_epochs = 100
lr = 0.0001
batch_size = step_count

agent_krargs_ppo = dict(verbose=0, policy=MlpPolicy,
                        n_steps=step_count,
                        n_epochs=num_epochs,
                        learning_rate=lr,
                        batch_size=batch_size)

env = BurgersPhysicsGym(**env_krargs)

agent = PPO(env=env, **agent_krargs_ppo)

obs = env.reset()
u = env.init_state
plt.plot(u.data.native("vector,x")[0], label='initial state')
# for _ in view('state, obs', framerate=1, namespace=globals()).range():
for i in tqdm(range(100)):
    u = env.physics.step(u, dt=dt)
    # actions = (np.random.uniform(-1.0, 1.0),)
    # obs, rew, done, info = env.step(np.array(actions))
    # state = env.cont_state

plt.plot(u.data.native("vector,x")[0], label='final state')
plt.xlim(0, 8)
plt.ylim(-3, 3)
plt.legend()
plt.show()

print("training begins")
agent.learn(total_timesteps=100)
print("training complete")
