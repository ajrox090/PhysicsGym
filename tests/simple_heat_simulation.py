from phi.flow import *
from src.env.phiflow.heat import Heat
from src.util.burgers_util import GaussianClash, GaussianForce

N = 128
num_envs = 1
domain_dict = dict(x=N, bounds=Box[0:1])
step_count = 32
diffusivity = 0.01 / (N * np.pi)
# diffusivity = 0
dt = 1. / step_count

v = CenteredGrid(GaussianForce, **domain_dict)
physics = Heat(diffusivity=diffusivity)

for _ in view(play=False, namespace=globals()).range():
    v = physics.step(v, dt=dt)
