from tqdm import tqdm

from phi.flow import *
from src.env.KSPhysicsGym import KuramotoSivashinskyPhysicsGym

dx = 0.25
domain = 5
N = int(domain / dx)
step_count = 1000
domain_dict = dict(x=N, bounds=Box[0:1],
                   extrapolation=extrapolation.BOUNDARY)
dt = 0.01

env_krargs = dict(domain=domain, dx=dx, domain_dict=domain_dict, dt=dt, step_count=step_count,
                  final_reward_factor=32)

env = KuramotoSivashinskyPhysicsGym(**env_krargs)
observation = env.reset()
rew = []
for _ in tqdm(range(500)):
    actions = np.array([-1])
    observation, reward, done, info = env.step(actions)


for _ in tqdm(range(1000)):
    actions = np.array([1])
    observation, reward, done, info = env.step(actions)

for _ in tqdm(range(1000)):
    actions = np.array([1])
    observation, reward, done, info = env.step(actions)
