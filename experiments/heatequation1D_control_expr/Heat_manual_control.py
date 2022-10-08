from tqdm import tqdm

from phi.flow import *
from src.env.HeatPhysicsGym import HeatPhysicsGym

dx = 0.25
domain = 5
N = int(domain / dx)
step_count = 1000
domain_dict = dict(x=N, bounds=Box[0:1],
                   extrapolation=extrapolation.BOUNDARY)
dt = 0.01
viscosity = 0.01 / (N * np.pi)

env_krargs = dict(domain=domain, dx=dx, domain_dict=domain_dict, dt=dt, step_count=step_count,
                  diffusivity=0.3,
                  final_reward_factor=32)

env = HeatPhysicsGym(**env_krargs)
observation = env.reset()
rew = []
# for _ in tqdm(range(500)):
#     actions = np.array([-1])
#     observation, reward, done, info = env.step(actions)


# observation = env.reset()
for _ in tqdm(range(1000)):
    actions = np.array([1])
    observation, reward, done, info = env.step(actions)

# for _ in tqdm(range(1000)):
#     actions = np.array([1])
#     observation, reward, done, info = env.step(actions)
