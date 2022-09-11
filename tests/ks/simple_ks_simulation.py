from phi.flow import *
from tqdm import tqdm

from src.env.physics.burgers import Burgers
from src.env.physics.ks import KuramotoSivashinsky


# physics = KuramotoSivashinsky()
physics = Burgers(default_viscosity=0.1)
step_count = 1000
dt = 0.15
N = 128
flame1 = CenteredGrid(Noise(scale=5), x=N, extrapolation=extrapolation.PERIODIC, bounds=Box(x=N))
flame2 = CenteredGrid(flame1 * 0.3, x=N, bounds=Box(x=N), extrapolation=extrapolation.PERIODIC)
trajectory = [flame2.vector['x']]
# for _ in view('flame2', framerate=10, namespace=globals()).range():
for _ in tqdm(range(step_count)):
    flame2 = physics.step(flame2, dt=dt)
    trajectory.append(flame2.vector['x'])

temp_t = field.stack(trajectory, spatial('time'),
                     Box(time=len(trajectory) * dt))  # time=len(trajectory)* dt
vis.show(temp_t.vector[0], aspect='auto', size=(8, 6))
